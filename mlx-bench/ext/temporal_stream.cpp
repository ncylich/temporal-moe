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

#include <atomic>
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
      int qd) {
    teardown();
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
    events_.clear();
    pool_.reset();
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

  // The per-layer fetch, run on the Metal completion thread. Offset streams
  // are byte-identical to temporal.py::_issue_disk (split-1 single pread for
  // n==1; n whole-expert reads through the pool for n>=2).
  void on_signal(int layer) {
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
    // ordering-control log + release
    uint64_t v = ++sigval_[layer];
    {
      std::lock_guard<std::mutex> lk(log_m_);
      seq_log_.emplace_back(layer, v);
    }
    events_[layer]->setSignaledValue(v);
  }

  void host_signal(int layer, uint64_t value) { // test aid
    events_[layer]->setSignaledValue(value);
  }

  MTL::SharedEvent* event(int layer) {
    return events_[layer].get();
  }

  int fd_ = -1;
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
    enc.end_encoding();
    int layer = layer_;
    enc.commit([layer]() { Registry::inst().on_signal(layer); });
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
    enc.end_encoding();
    enc.get_command_buffer()->encodeWait(Registry::inst().event(layer_), value_);
  }

  const char* name() const override {
    return "WaitFetch";
  }

 private:
  int layer_;
  uint64_t value_;
};

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

} // namespace temporal_stream

NB_MODULE(_temporal_stream, m) {
  m.doc() = "TEMPORAL-PATCH: shared-event fetch stream primitives for the Mac serving bench";
  m.def("setup", [](const std::string& path, uint64_t eb, int n_layers, int nfetch, int qd) {
    temporal_stream::Registry::inst().setup(path, eb, n_layers, nfetch, qd);
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
  m.def("host_signal", [](int layer, uint64_t value) {
    temporal_stream::Registry::inst().host_signal(layer, value);
  });
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
