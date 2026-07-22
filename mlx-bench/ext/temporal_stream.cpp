// TEMPORAL-PATCH: out-of-tree MLX extension (engine fork for the Mac serving
// bench, PLAN.md Phase 4 escalation). Adds the two graph primitives the
// Python API cannot express, transposing the CUDA fork's zero-host-sync
// design to Metal:
//
//   signal_fetch(x, layer) -> x   (identity dataflow)
//     eval_gpu: ends the current encoder and COMMITS the current command
//     buffer with a C++ completed-handler (Metal completion thread, no GIL):
//     the handler issues the layer's pread(s) from the registered pool fd
//     (offset streams byte-identical to temporal.py's _issue_disk) and then
//     signals the layer's MTLSharedEvent at the next monotonic value.
//
//   wait_fetch(x, sig_dep, layer, value) -> x   (identity dataflow)
//     eval_gpu: ends the current encoder and encodes a command-buffer-level
//     encodeWait(sharedEvent[layer], value): everything ordered after this
//     node executes only once the host handler signalled `value`.
//
// Together they let the WHOLE token submit as one pipelined graph (like the
// ceiling path) while preserving fetch-on-miss causality structurally: the
// handler can only fire when the command buffer containing the layer's
// routing/decision has COMPLETED on the GPU, and the fetched-expert
// contribution is ordered after the wait. Coherency is sound because the
// handshake happens at command-buffer boundaries (the platform's coherency
// points -- see the Stage-A spike that ruled out in-stream spinning).
//
// Registry: pool fd (F_NOCACHE), expert byte-geometry, per-layer cycled
// source counters (7919 stride; n>=2 advances n*7919+1 -- exactly
// temporal.py), logical-byte counters for the audits, per-layer shared
// events (recreated on reset: MTLSharedEvent values are monotonic), and a
// small C++ pread pool for n>=2 batches (QD workers, no GIL anywhere).

#include <array>
#include <atomic>
#include <mach/mach_time.h>
#include <pthread/qos.h>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fcntl.h>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/uio.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "mlx/backend/metal/device.h"
#include "mlx/mlx.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;
namespace nb = nanobind;

namespace temporal_stream {

// ----------------------------- pread pool (C++) -----------------------------
struct PreadPool {
  explicit PreadPool(int nthreads) : stop_(false) {
    for (int i = 0; i < nthreads; ++i) {
      threads_.emplace_back([this] { run(); });
    }
  }
  ~PreadPool() {
    {
      std::lock_guard<std::mutex> lk(m_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : threads_) {
      t.join();
    }
  }
  // Blocking batch: read each (offset, dst, len) then return.
  void read_batch(int fd, const std::vector<std::tuple<uint64_t, uint8_t*, size_t>>& parts) {
    std::atomic<size_t> remaining(parts.size());
    std::mutex dm;
    std::condition_variable dcv;
    {
      std::lock_guard<std::mutex> lk(m_);
      for (auto& p : parts) {
        q_.emplace_back([&, p] {
          auto [off, dst, len] = p;
          ssize_t got = ::pread(fd, dst, len, (off_t)off);
          if (got != (ssize_t)len) {
            err_.store(true);
          }
          if (remaining.fetch_sub(1) == 1) {
            std::lock_guard<std::mutex> dlk(dm);
            dcv.notify_one();
          }
        });
      }
    }
    cv_.notify_all();
    std::unique_lock<std::mutex> dlk(dm);
    dcv.wait(dlk, [&] { return remaining.load() == 0; });
  }
  std::atomic<bool> err_{false};

 private:
  void run() {
    // pin to USER_INTERACTIVE (same rationale as the python engine)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [this] { return stop_ || !q_.empty(); });
        if (stop_ && q_.empty()) {
          return;
        }
        task = std::move(q_.front());
        q_.pop_front();
      }
      task();
    }
  }
  std::mutex m_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> q_;
  std::vector<std::thread> threads_;
  bool stop_;
};

// ------------------------------- registry -----------------------------------
struct Registry {
  static Registry& inst() {
    static Registry r;
    return r;
  }

  void setup(
      const std::string& path,
      uint64_t expert_bytes,
      int n_layers,
      int n_per_fetch,
      int qd,
      int sig_mode = 0,   // 0 = commit+completion-handler, 1 = encodeSignalEvent+service thread
      bool trace = false) {
    teardown();
    sig_mode_ = sig_mode;
    trace_on_ = trace;
    pool_path_ = path;
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
      throw std::runtime_error("[temporal_stream] cannot open pool: " + path);
    }
    ::fcntl(fd_, F_NOCACHE, 1);
    struct stat st;
    ::fstat(fd_, &st);
    eb_ = expert_bytes;
    n_layers_ = n_layers;
    nfetch_ = n_per_fetch;
    uint64_t stride = (uint64_t)n_layers * eb_;
    if ((uint64_t)st.st_size % stride != 0) {
      throw std::runtime_error("[temporal_stream] pool size not a multiple of n_layers*expert_bytes");
    }
    disk_E_ = (uint64_t)st.st_size / stride;
    dcycle_.assign(n_layers_, 0);
    sigval_.assign(n_layers_, 0);
    ring_.assign(2, std::vector<uint8_t>((size_t)std::max(1, nfetch_) * eb_));
    ring_i_ = 0;
    bytes_.store(0);
    fetches_.store(0);
    seq_log_.clear();
    pool_.reset(new PreadPool(qd));
    make_events();
  }

  void make_events() {
    auto* mtl = mx::metal::device(mx::Device::gpu).mtl_device();
    events_.clear();
    for (int i = 0; i < n_layers_; ++i) {
      auto* e = mtl->newSharedEvent();
      if (!e) {
        throw std::runtime_error("[temporal_stream] newSharedEvent failed");
      }
      events_.push_back(NS::TransferPtr(e));
    }
    // v2 (event-signal mode): ONE GPU->CPU signal event + ONE CPU->GPU release
    // event, shared across layers with globally monotonic values (encode order
    // == stream order). The service thread consumes values sequentially.
    sig_event_ = NS::TransferPtr(mtl->newSharedEvent());
    rel_event_ = NS::TransferPtr(mtl->newSharedEvent());
    encode_val_.store(0);
    {
      std::lock_guard<std::mutex> lk(vmap_m_);
      val_layer_.clear();
      pending_val_.assign(n_layers_, 0);
    }
    stop_service();
    if (sig_mode_ == 1 || sig_mode_ == 2) {
      start_service();
    }
    if (sig_mode_ == 3 && io_queue_ == nullptr) {
      NS::Error* err = nullptr;
      auto* url = NS::URL::fileURLWithPath(NS::String::string(
          pool_path_.c_str(), NS::UTF8StringEncoding));
      io_fh_ = mtl->newIOHandle(url, &err);
      auto* qd = MTL::IOCommandQueueDescriptor::alloc()->init();
      qd->setType(MTL::IOCommandQueueTypeSerial);
      qd->setPriority(MTL::IOPriorityHigh);
      io_queue_ = mtl->newIOCommandQueue(qd, &err);
      qd->release();
      if (!io_fh_ || !io_queue_) {
        throw std::runtime_error("[temporal_stream] MTLIO init failed");
      }
      size_t sz = (size_t)std::max(1, nfetch_) * eb_;
      for (int i = 0; i < 2; ++i) {
        io_bufs_.push_back(mtl->newBuffer(sz, MTL::ResourceStorageModeShared));
      }
    }
    io_err_.store(false);
    {
      std::lock_guard<std::mutex> lk(log_m_);
      io_done_log_.clear();
    }
  }

  // v4: called from SignalFetch::eval_gpu (encode/eval time). Builds and
  // commits the layer's IO command buffer: [wait sig@v] -> load(s) ->
  // [signal rel@v]. Offsets are the SAME encode-time cycled counters the
  // other engines use (emulation semantics: sources never depend on routing;
  // the LOAD's causality is event-enforced by the in-stream routing signal).
  void encode_io_fetch(int layer, uint64_t v) {
    auto* icb = io_queue_->commandBuffer();
    icb->wait(sig_event_.get(), v);
    if (nfetch_ > 0) {
      auto* buf = io_bufs_[io_buf_i_];
      io_buf_i_ = (io_buf_i_ + 1) % (int)io_bufs_.size();
      uint64_t L = (uint64_t)layer;
      if (nfetch_ == 1) {
        uint64_t off = (L * disk_E_ + (dcycle_[layer] % disk_E_)) * eb_;
        dcycle_[layer] = (dcycle_[layer] + 7919) % disk_E_;
        icb->loadBuffer(buf, 0, eb_, io_fh_, off);
      } else {
        for (int i = 0; i < nfetch_; ++i) {
          uint64_t off =
              (L * disk_E_ + (dcycle_[layer] + (uint64_t)i * 7919) % disk_E_) * eb_;
          icb->loadBuffer(buf, (uint64_t)i * eb_, eb_, io_fh_, off);
        }
        dcycle_[layer] = (dcycle_[layer] + (uint64_t)nfetch_ * 7919 + 1) % disk_E_;
      }
      bytes_.fetch_add(eb_ * (uint64_t)nfetch_);
      fetches_.fetch_add(1);
    }
    icb->signalEvent(rel_event_.get(), v);
    icb->addCompletedHandler(
        MTL::IOCommandBufferHandlerFunction([this, layer, v](MTL::IOCommandBuffer* b) {
          if (b->status() != MTL::IOStatusComplete) {
            io_err_.store(true);
          }
          std::lock_guard<std::mutex> lk(log_m_);
          io_done_log_.push_back({(uint64_t)layer, v, mach_absolute_time()});
          seq_log_.emplace_back(layer, v);
        }));
    icb->commit();
  }

  void start_service() {
    svc_stop_.store(false);
    svc_ = std::thread([this] {
      pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
      mach_timebase_info_data_t tb;
      mach_timebase_info(&tb);
      uint64_t v = 1;
      while (!svc_stop_.load()) {
        if (sig_mode_ == 2) {
          // v3: SPIN-POLL the shared event's signaledValue (objc property
          // read ~100ns) with ISB pauses -- a thread that never blocks never
          // loses the macOS wake-placement lottery. Safety valve: after 50ms
          // without progress, fall back to ONE blocking timed wait (idle
          // gaps/cooldowns land here; the next signal wakes it and the loop
          // returns to spinning).
          uint64_t spin_start = mach_absolute_time();
          bool got = false;
          while (!svc_stop_.load()) {
            if (sig_event_->signaledValue() >= v) {
              got = true;
              break;
            }
            for (int i = 0; i < 8; ++i) {
              __builtin_arm_isb(15);
            }
            if ((mach_absolute_time() - spin_start) * tb.numer / tb.denom >
                50000000ull) {
              break;                       // 50ms: yield the core
            }
          }
          if (!got) {
            if (!sig_event_->waitUntilSignaledValue(v, 50)) {
              if (warm_req_.exchange(false)) {
                // (spin mode: nothing to warm -- the spin IS the warmth)
              }
              continue;
            }
          }
          uint64_t t0 = trace_on_ ? mach_absolute_time() : 0;
          int layer = -1;
          {
            std::lock_guard<std::mutex> lk(vmap_m_);
            if (v - 1 < val_layer_.size()) {
              layer = val_layer_[v - 1];
            }
          }
          if (layer < 0) {
            continue;   // value observed before its encode bookkeeping: re-spin
          }
          do_fetch(layer);
          uint64_t t1 = trace_on_ ? mach_absolute_time() : 0;
          rel_event_->setSignaledValue(v);
          if (trace_on_) {
            std::lock_guard<std::mutex> lk(log_m_);
            trace_.push_back({(uint64_t)layer, v, t0, t1, mach_absolute_time()});
          }
          {
            std::lock_guard<std::mutex> lk(log_m_);
            seq_log_.emplace_back(layer, v);
          }
          ++v;
          continue;
        }
        if (!sig_event_->waitUntilSignaledValue(v, 50)) {
          // timeout: re-check stop flag; honor a pending keep-warm request
          // (post-cooldown P-core re-promotion, the service-thread analog of
          // temporal._respin -- long-idle wakes land on E-cores and the whole
          // following rep then runs with inflated wake latency)
          if (warm_req_.exchange(false)) {
            uint64_t t0 = mach_absolute_time();
            mach_timebase_info_data_t tb;
            mach_timebase_info(&tb);
            volatile double x = 1.0;
            while ((mach_absolute_time() - t0) * tb.numer / tb.denom < 20000000ull) {
              x = x * 1.0000001 + 1e-9;   // ~20 ms CPU burst
            }
          }
          continue;
        }
        uint64_t t0 = trace_on_ ? mach_absolute_time() : 0;
        int layer;
        {
          std::lock_guard<std::mutex> lk(vmap_m_);
          layer = val_layer_.at(v - 1);
        }
        do_fetch(layer);
        uint64_t t1 = trace_on_ ? mach_absolute_time() : 0;
        rel_event_->setSignaledValue(v);
        if (trace_on_) {
          std::lock_guard<std::mutex> lk(log_m_);
          trace_.push_back({(uint64_t)layer, v, t0, t1, mach_absolute_time()});
        }
        {
          std::lock_guard<std::mutex> lk(log_m_);
          seq_log_.emplace_back(layer, v);
        }
        ++v;
      }
    });
  }

  void stop_service() {
    if (svc_.joinable()) {
      svc_stop_.store(true);
      svc_.join();
    }
  }

  void reset() {
    // Fresh monotonic domains: recreate events, zero counters/cycles.
    std::fill(dcycle_.begin(), dcycle_.end(), 0);
    std::fill(sigval_.begin(), sigval_.end(), 0);
    bytes_.store(0);
    fetches_.store(0);
    {
      std::lock_guard<std::mutex> lk(log_m_);
      seq_log_.clear();
    }
    make_events();
  }

  void teardown() {
    stop_service();
    if (io_queue_) {
      io_queue_->release();
      io_queue_ = nullptr;
    }
    if (io_fh_) {
      io_fh_->release();
      io_fh_ = nullptr;
    }
    for (auto* b : io_bufs_) {
      b->release();
    }
    io_bufs_.clear();
    io_buf_i_ = 0;
    events_.clear();
    sig_event_.reset();
    rel_event_.reset();
    pool_.reset();
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

  // The per-layer fetch, run on the Metal completion thread. Offset streams
  // are byte-identical to temporal.py::_issue_disk (split-1 single pread for
  // n==1; n whole-expert reads through the pool for n>=2).
  void do_fetch(int layer) {
    if (nfetch_ > 0 && fd_ >= 0) {
      auto& ring = ring_[ring_i_];
      ring_i_ = (ring_i_ + 1) % ring_.size();
      uint64_t L = (uint64_t)layer;
      if (nfetch_ == 1) {
        uint64_t off = (L * disk_E_ + (dcycle_[layer] % disk_E_)) * eb_;
        dcycle_[layer] = (dcycle_[layer] + 7919) % disk_E_;
        ssize_t got = ::pread(fd_, ring.data(), eb_, (off_t)off);
        if (got != (ssize_t)eb_) {
          pool_->err_.store(true);
        }
      } else {
        std::vector<std::tuple<uint64_t, uint8_t*, size_t>> parts;
        parts.reserve(nfetch_);
        for (int i = 0; i < nfetch_; ++i) {
          uint64_t off = (L * disk_E_ + (dcycle_[layer] + (uint64_t)i * 7919) % disk_E_) * eb_;
          parts.emplace_back(off, ring.data() + (size_t)i * eb_, eb_);
        }
        dcycle_[layer] = (dcycle_[layer] + (uint64_t)nfetch_ * 7919 + 1) % disk_E_;
        pool_->read_batch(fd_, parts);
      }
      bytes_.fetch_add(eb_ * (uint64_t)nfetch_);
      fetches_.fetch_add(1);
    }
  }

  // v1 commit-mode completion handler body
  void on_signal(int layer, MTL::CommandBuffer* cbuf) {
    uint64_t t0 = trace_on_ ? mach_absolute_time() : 0;
    do_fetch(layer);
    uint64_t t1 = trace_on_ ? mach_absolute_time() : 0;
    uint64_t v = ++sigval_[layer];
    {
      std::lock_guard<std::mutex> lk(log_m_);
      seq_log_.emplace_back(layer, v);
    }
    events_[layer]->setSignaledValue(v);
    if (trace_on_ && cbuf) {
      // GPUStart/EndTime are seconds in the mach timebase domain
      uint64_t gs = (uint64_t)(cbuf->GPUStartTime() * 1e9);
      uint64_t ge = (uint64_t)(cbuf->GPUEndTime() * 1e9);
      std::lock_guard<std::mutex> lk(log_m_);
      trace_.push_back({(uint64_t)layer, v, t0, t1, mach_absolute_time()});
      cbtrace_.push_back({(uint64_t)layer, v, gs, ge});
    }
  }

  // v2 encode-time bookkeeping: assign the next global value to this layer
  uint64_t assign_value(int layer) {
    uint64_t v = encode_val_.fetch_add(1) + 1;
    std::lock_guard<std::mutex> lk(vmap_m_);
    val_layer_.push_back(layer);
    pending_val_[layer] = v;
    return v;
  }
  uint64_t pending_value(int layer) {
    std::lock_guard<std::mutex> lk(vmap_m_);
    return pending_val_[layer];
  }

  void host_signal(int layer, uint64_t value) { // test aid
    events_[layer]->setSignaledValue(value);
  }

  MTL::SharedEvent* event(int layer) {
    return events_[layer].get();
  }

  int fd_ = -1;
  std::string pool_path_;
  uint64_t eb_ = 0, disk_E_ = 0;
  int n_layers_ = 0, nfetch_ = 0;
  std::vector<uint64_t> dcycle_, sigval_;
  std::vector<std::vector<uint8_t>> ring_;
  int ring_i_ = 0;
  std::atomic<uint64_t> bytes_{0}, fetches_{0};
  std::vector<NS::SharedPtr<MTL::SharedEvent>> events_;
  std::unique_ptr<PreadPool> pool_;
  std::mutex log_m_;
  std::vector<std::pair<int, uint64_t>> seq_log_;
  // v2 members
  int sig_mode_ = 0;
  bool trace_on_ = false;
  NS::SharedPtr<MTL::SharedEvent> sig_event_, rel_event_;
  std::atomic<uint64_t> encode_val_{0};
  std::mutex vmap_m_;
  std::vector<int> val_layer_;
  std::vector<uint64_t> pending_val_;
  std::thread svc_;
  std::atomic<bool> svc_stop_{false};
  std::atomic<bool> warm_req_{false};
  // v4 (mtlio): IO queue path -- pre-committed IO command buffers parked on
  // the in-stream routing signal; zero CPU in the fetch loop.
  MTL::IOCommandQueue* io_queue_ = nullptr;
  MTL::IOFileHandle* io_fh_ = nullptr;
  std::vector<MTL::Buffer*> io_bufs_;
  int io_buf_i_ = 0;
  std::atomic<bool> io_err_{false};
  std::vector<std::array<uint64_t, 3>> io_done_log_;  // layer, value, t_done
  std::vector<std::array<uint64_t, 5>> trace_;
  std::vector<std::array<uint64_t, 4>> cbtrace_;
};

// ------------------------------ primitives ----------------------------------
class SignalFetch : public mx::Primitive {
 public:
  SignalFetch(mx::Stream stream, int layer)
      : mx::Primitive(stream), layer_(layer) {}

  void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    throw std::runtime_error("[signal_fetch] GPU only");
  }

  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    outputs[0].copy_shared_buffer(inputs[0]);
    auto& enc = mx::metal::get_command_encoder(stream());
    auto& reg = Registry::inst();
    enc.end_encoding();
    if (reg.sig_mode_ >= 1) {
      // v2/v3/v4: in-stream signal -- fires when the GPU PASSES this point
      // (no CB drain, no per-layer commit; the token stays deeply pipelined).
      uint64_t v = reg.assign_value(layer_);
      enc.get_command_buffer()->encodeSignalEvent(reg.sig_event_.get(), v);
      if (reg.sig_mode_ == 3) {
        // v4: pre-commit the layer's IO command buffer, parked on sig@v; it
        // loads the expert bytes on Metal's IO queue and signals rel@v --
        // zero CPU in the fetch loop.
        reg.encode_io_fetch(layer_, v);
      }
    } else {
      int layer = layer_;
      MTL::CommandBuffer* cbuf = enc.get_command_buffer();  // borrowed; alive
      enc.commit([layer, cbuf]() {                          //  during handlers
        Registry::inst().on_signal(layer, cbuf);
      });
    }
  }

  const char* name() const override {
    return "SignalFetch";
  }

 private:
  int layer_;
};

class WaitFetch : public mx::Primitive {
 public:
  WaitFetch(mx::Stream stream, int layer, uint64_t value)
      : mx::Primitive(stream), layer_(layer), value_(value) {}

  void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    throw std::runtime_error("[wait_fetch] GPU only");
  }

  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    outputs[0].copy_shared_buffer(inputs[0]);
    auto& enc = mx::metal::get_command_encoder(stream());
    auto& reg = Registry::inst();
    enc.end_encoding();
    if (reg.sig_mode_ >= 1) {
      // the paired SignalFetch encoded first (dataflow dependency), so the
      // layer's pending global value is the one to gate on
      enc.get_command_buffer()->encodeWait(
          reg.rel_event_.get(), reg.pending_value(layer_));
    } else {
      enc.get_command_buffer()->encodeWait(Registry::inst().event(layer_), value_);
    }
  }

  const char* name() const override {
    return "WaitFetch";
  }

 private:
  int layer_;
  uint64_t value_;
};

class CommitBoundary : public mx::Primitive {
 public:
  explicit CommitBoundary(mx::Stream stream) : mx::Primitive(stream) {}
  void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    throw std::runtime_error("[commit_boundary] GPU only");
  }
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    // Plain command-buffer boundary (no handler, no event): commands encoded
    // BEFORE this point (the masked hit contributions) land in their own CB
    // with no event wait, so Metal kicks it off immediately after the
    // previous CB -- the hits really execute DURING the fetch. Without this,
    // Metal defers starting any CB whose stream contains an unsatisfied
    // event wait, serializing the hits behind the release (measured 0/3599).
    outputs[0].copy_shared_buffer(inputs[0]);
    auto& enc = mx::metal::get_command_encoder(stream());
    enc.end_encoding();
    enc.commit();
  }
  const char* name() const override {
    return "CommitBoundary";
  }
};

mx::array commit_boundary(const mx::array& x, mx::StreamOrDevice s) {
  return mx::array(x.shape(), x.dtype(),
                   std::make_shared<CommitBoundary>(mx::to_stream(s)), {x});
}

mx::array signal_fetch(const mx::array& x, int layer, mx::StreamOrDevice s) {
  return mx::array(
      x.shape(), x.dtype(),
      std::make_shared<SignalFetch>(mx::to_stream(s), layer), {x});
}

mx::array wait_fetch(
    const mx::array& x,
    const mx::array& sig_dep,
    int layer,
    uint64_t value,
    mx::StreamOrDevice s) {
  return mx::array(
      x.shape(), x.dtype(),
      std::make_shared<WaitFetch>(mx::to_stream(s), layer, value),
      {x, sig_dep});
}

// ------------------------- MTLIO spike (Stage A, v4) -------------------------
// Measures Metal 3 fast-resource-streaming as a fetch path: load latency at
// our offsets, event-chained round trip, and cache behavior. Self-contained;
// uses its own IO/compute queues, not MLX's stream.
static nb::dict mtlio_spike(const std::string& path, int iters, bool aligned,
                            bool chain, int priority, uint64_t seed) {
  nb::dict out;
  auto* pool = NS::AutoreleasePool::alloc()->init();
  auto* dev = mx::metal::device(mx::Device::gpu).mtl_device();
  NS::Error* err = nullptr;
  auto* url = NS::URL::fileURLWithPath(
      NS::String::string(path.c_str(), NS::UTF8StringEncoding));
  MTL::IOFileHandle* fh = dev->newIOHandle(url, &err);
  if (!fh) {
    throw std::runtime_error("newIOHandle failed");
  }
  auto* qd = MTL::IOCommandQueueDescriptor::alloc()->init();
  qd->setType(MTL::IOCommandQueueTypeSerial);
  qd->setPriority((MTL::IOPriority)priority);
  MTL::IOCommandQueue* ioq = dev->newIOCommandQueue(qd, &err);
  if (!ioq) {
    throw std::runtime_error("newIOCommandQueue failed");
  }
  const uint64_t EB = 663552, PG = 16384;
  uint64_t fsz = 0;
  {
    struct stat st;
    ::stat(path.c_str(), &st);
    fsz = (uint64_t)st.st_size;
  }
  auto* buf = dev->newBuffer(EB + 2 * PG, MTL::ResourceStorageModeShared);
  mach_timebase_info_data_t tb;
  mach_timebase_info(&tb);
  auto ns_of = [&](uint64_t t) { return t * tb.numer / tb.denom; };
  uint64_t cyc = seed;
  auto next_off = [&]() {
    cyc = (cyc + 7919) % ((fsz - 2 * PG) / EB);
    return cyc * EB;
  };
  std::vector<double> lat;
  // (i) single-load latency, cycled offsets
  for (int i = 0; i < iters; ++i) {
    uint64_t off = next_off();
    uint64_t o = off, len = EB;
    if (aligned) {
      o = (off / PG) * PG;
      len = ((off - o + EB + PG - 1) / PG) * PG;
    }
    auto* icb = ioq->commandBuffer();
    icb->loadBuffer(buf, 0, len, fh, o);
    uint64_t t0 = mach_absolute_time();
    icb->commit();
    icb->waitUntilCompleted();
    lat.push_back(ns_of(mach_absolute_time() - t0) / 1e3);
    if (icb->status() != MTL::IOStatusComplete) {
      out["io_error"] = true;
    }
  }
  std::sort(lat.begin(), lat.end());
  out["load_med_us"] = lat[lat.size() / 2];
  out["load_p90_us"] = lat[(size_t)(lat.size() * 0.9)];
  out["load_min_us"] = lat.front();
  // warm-repeat: same offset 40x (cache probe)
  {
    uint64_t off = next_off();
    std::vector<double> w;
    for (int i = 0; i < 40; ++i) {
      auto* icb = ioq->commandBuffer();
      icb->loadBuffer(buf, 0, EB, fh, off);
      uint64_t t0 = mach_absolute_time();
      icb->commit();
      icb->waitUntilCompleted();
      w.push_back(ns_of(mach_absolute_time() - t0) / 1e3);
    }
    std::sort(w.begin(), w.end());
    out["warm_repeat_med_us"] = w[w.size() / 2];
  }
  if (chain) {
    // (ii) event-chained round trip: [compute signal E@v] -> [IO waits E@v,
    // loads, signals E@v+1] -> [compute waits E@v+1]; IO CB pre-committed.
    auto* cq = dev->newCommandQueue();
    auto* ev = dev->newSharedEvent();
    std::vector<double> ch;
    uint64_t v = 0;
    for (int i = 0; i < iters; ++i) {
      uint64_t off = next_off();
      auto* icb = ioq->commandBuffer();
      icb->wait(ev, v + 1);
      icb->loadBuffer(buf, 0, EB, fh, off);
      icb->signalEvent(ev, v + 2);
      icb->commit();                       // pre-committed, parked on the wait
      auto* ca = cq->commandBuffer();
      ca->encodeSignalEvent(ev, v + 1);
      auto* cb = cq->commandBuffer();
      cb->encodeWait(ev, v + 2);
      uint64_t t0 = mach_absolute_time();
      ca->commit();
      cb->commit();
      cb->waitUntilCompleted();
      ch.push_back(ns_of(mach_absolute_time() - t0) / 1e3);
      v += 2;
    }
    std::sort(ch.begin(), ch.end());
    out["chain_med_us"] = ch[ch.size() / 2];
    out["chain_p90_us"] = ch[(size_t)(ch.size() * 0.9)];
    out["chain_min_us"] = ch.front();
    cq->release();
    ev->release();
  }
  buf->release();
  ioq->release();
  qd->release();
  fh->release();
  pool->release();
  return out;
}

} // namespace temporal_stream

NB_MODULE(_temporal_stream, m) {
  m.def("mtlio_spike", &temporal_stream::mtlio_spike, nb::arg("path"),
        nb::arg("iters") = 100, nb::arg("aligned") = false,
        nb::arg("chain") = false, nb::arg("priority") = 0,
        nb::arg("seed") = 13);
  m.doc() = "TEMPORAL-PATCH: shared-event fetch stream primitives for the Mac serving bench";
  m.def("setup",
        [](const std::string& path, uint64_t eb, int n_layers, int nfetch,
           int qd, int sig_mode, bool trace) {
          temporal_stream::Registry::inst().setup(
              path, eb, n_layers, nfetch, qd, sig_mode, trace);
        },
        nb::arg("path"), nb::arg("eb"), nb::arg("n_layers"), nb::arg("nfetch"),
        nb::arg("qd"), nb::arg("sig_mode") = 0, nb::arg("trace") = false);
  m.def("trace_ns", []() {
    auto& r = temporal_stream::Registry::inst();
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    auto to_ns = [&](uint64_t t) { return t * tb.numer / tb.denom; };
    std::lock_guard<std::mutex> lk(r.log_m_);
    nb::list out;
    for (auto& e : r.trace_) {   // layer, value, t_start, t_fetch_done, t_signal_set (mach)
      out.append(nb::make_tuple(e[0], e[1], to_ns(e[2]), to_ns(e[3]), to_ns(e[4])));
    }
    nb::list cbs;
    for (auto& e : r.cbtrace_) { // layer, value, gpu_start_ns, gpu_end_ns
      cbs.append(nb::make_tuple(e[0], e[1], e[2], e[3]));
    }
    return nb::make_tuple(out, cbs);
  });
  m.def("reset", []() { temporal_stream::Registry::inst().reset(); });
  m.def("teardown", []() { temporal_stream::Registry::inst().teardown(); });
  m.def("copied_bytes", []() { return temporal_stream::Registry::inst().bytes_.load(); });
  m.def("fetches", []() { return temporal_stream::Registry::inst().fetches_.load(); });
  m.def("io_error", []() {
    auto& r = temporal_stream::Registry::inst();
    return r.pool_ && r.pool_->err_.load();
  });
  m.def("seq_log", []() {
    auto& r = temporal_stream::Registry::inst();
    std::lock_guard<std::mutex> lk(r.log_m_);
    nb::list out;
    for (auto& [l, v] : r.seq_log_) {
      out.append(nb::make_tuple(l, v));
    }
    return out;
  });
  m.def("poll_cost_ns", [](int iters) {
    auto& r = temporal_stream::Registry::inst();
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    uint64_t t0 = mach_absolute_time();
    uint64_t acc = 0;
    for (int i = 0; i < iters; ++i) {
      acc += r.sig_event_->signaledValue();
    }
    uint64_t dt = (mach_absolute_time() - t0) * tb.numer / tb.denom;
    return nb::make_tuple((double)dt / iters, acc);
  });
  m.def("service_warm", []() {
    temporal_stream::Registry::inst().warm_req_.store(true);
  });
  m.def("host_signal", [](int layer, uint64_t value) {
    temporal_stream::Registry::inst().host_signal(layer, value);
  });
  m.def(
      "commit_boundary",
      [](const mx::array& x) { return temporal_stream::commit_boundary(x, {}); },
      nb::arg("x"));
  m.def(
      "signal_fetch",
      [](const mx::array& x, int layer) {
        return temporal_stream::signal_fetch(x, layer, {});
      },
      nb::arg("x"), nb::arg("layer"));
  m.def(
      "wait_fetch",
      [](const mx::array& x, const mx::array& sig_dep, int layer, uint64_t value) {
        return temporal_stream::wait_fetch(x, sig_dep, layer, value, {});
      },
      nb::arg("x"), nb::arg("sig_dep"), nb::arg("layer"), nb::arg("value"));
}
