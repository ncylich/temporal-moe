#!/bin/bash
# Reduced stub for the upstream CMU FLAME-MoE SLURM/GCS scripts.
#
# The original set a CMU-owned GCS bucket, SLURM job-scoped scratch paths, and activated a conda
# environment named MoE. None of that is used by the temporal-moe pipeline, which runs single-GPU
# from a local checkout via scripts/env.sh. This file is kept only because 16 upstream scripts
# under scripts/{training,empirical_analysis,ablation}/ and scripts/evaluate.sh still `source` it,
# and deleting it would break them at load time.
#
# Removed deliberately:
#   gcloud config set core/disable_file_logging   side effect, and requires gcloud to be installed
#   conda activate MoE                            that environment does not exist here
#   SSD_MOUNT=/tmp/slurm-$SLURM_JOB_ID            unset SLURM_JOB_ID aborts any caller using set -u
#
# For the pipeline that is actually maintained, see scripts/env.sh.
# Set GCP_MOUNT yourself if you are reviving the upstream GCS path.

export GCP_MOUNT="${GCP_MOUNT:-}"
export SSD_MOUNT="${SSD_MOUNT:-${TMPDIR:-/tmp}/temporal-moe-$$}"

export GCP_DATASET="${GCP_DATASET:-$GCP_MOUNT/dataset}"
export SSD_DATASET="${SSD_DATASET:-$SSD_MOUNT/dataset}"

export GCP_WEIGHTS="${GCP_WEIGHTS:-$GCP_MOUNT/weights}"
export SSD_WEIGHTS="${SSD_WEIGHTS:-$SSD_MOUNT/weights}"
