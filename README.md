# FLAME-MoE :fire:​: A Transparent End-to-End Research Platform for Mixture-of-Experts Language Models

> **Note — personal playground fork.** This is a private, personal copy of
> [cmu-flame/FLAME-MoE](https://github.com/cmu-flame/FLAME-MoE), maintained by
> [@ncylich](https://github.com/ncylich) purely for my own experimentation and
> learning. It is **not** affiliated with or endorsed by the original FLAME-MoE
> authors — all credit for the platform and research goes to them
> ([paper](https://www.arxiv.org/abs/2505.20225)). Expect ad-hoc, possibly-broken
> changes; this is a playground, not a maintained release.

**FLAME-MoE** is a transparent, end-to-end research platform for Mixture-of-Experts (MoE) language models. It is designed to facilitate scalable training, evaluation, and experimentation with MoE architectures. [arXiv](https://www.arxiv.org/abs/2505.20225)

## 🔗 Model Checkpoints

Explore our publicly released checkpoints on Hugging Face:

* [FLAME-MoE-1.7B-10.3B](https://huggingface.co/CMU-FLAME/FLAME-MoE-1.7B-10.3B)
* [FLAME-MoE-721M-3.8B](https://huggingface.co/CMU-FLAME/FLAME-MoE-721M-3.8B)
* [FLAME-MoE-419M-2.2B](https://huggingface.co/CMU-FLAME/FLAME-MoE-419M-2.2B)
* [FLAME-MoE-290M-1.3B](https://huggingface.co/CMU-FLAME/FLAME-MoE-290M-1.3B)
* [FLAME-MoE-115M-459M](https://huggingface.co/CMU-FLAME/FLAME-MoE-115M-459M)
* [FLAME-MoE-98M-349M](https://huggingface.co/CMU-FLAME/FLAME-MoE-98M-349M)
* [FLAME-MoE-38M-100M](https://huggingface.co/CMU-FLAME/FLAME-MoE-38M-100M)

---

## 🚀 Getting Started

### 1. Clone the Repository

Ensure you clone the repository **recursively** to include all submodules:

```bash
git clone --recursive https://github.com/cmu-flame/MoE-Research
cd MoE-Research
```

### 2. Set Up the Environment

Set up the Conda environment using the provided script:

```bash
sbatch scripts/miscellaneous/install.sh
```

> **Note:** This assumes you're using a SLURM-managed cluster. Adapt accordingly if running locally.

---

## 📚 Data Preparation

### 3. Download and Tokenize the Dataset

Use the following SLURM jobs to download and tokenize the dataset:

```bash
sbatch scripts/dataset/download.sh
sbatch scripts/dataset/tokenize.sh
```

---

## 🧠 Training

### 4. Train FLAME-MoE Models

Launch training jobs for the desired model configurations:

```bash
bash scripts/release/flame-moe-1.7b.sh
bash scripts/release/flame-moe-721m.sh
bash scripts/release/flame-moe-419m.sh
bash scripts/release/flame-moe-290m.sh
bash scripts/release/flame-moe-115m.sh
bash scripts/release/flame-moe-98m.sh
bash scripts/release/flame-moe-38m.sh
```

---

## 📈 Evaluation

### 5. Evaluate the Model

To evaluate a trained model, set the appropriate job ID and iteration number before submitting the evaluation script:

```bash
export JOBID=...    # Replace with your training job ID
export ITER=...     # Replace with the iteration to evaluate (e.g., 11029)
sbatch scripts/evaluate.sh
```

---

### 6. WandB Workspace

All the loss curves during training for both the scaling law studies and the final releases can be found in the following WandB workspace: https://wandb.ai/haok/flame-moe
