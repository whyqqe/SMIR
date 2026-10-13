#!/bin/bash
#SBATCH --job-name=smir-venue
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=18:00:00
#SBATCH --output=/project/peilab/why/NEW/smir/artifacts/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/NEW/smir/artifacts/slurm/%x-%j.err
#
# SMIR formal venue: 7B LoRA + semantic-flow chart
# Losses: task + conjugacy + tubular + metric + normal + IV
# vs ContCE / Outcome. Saves adapter+heads only.
#
#   sbatch smir/slurm/21_smir_venue.sbatch

set -euo pipefail
SMIR_ROOT="/project/peilab/why/NEW/smir"
CCRO_ROOT="/project/peilab/why/NEW/ccro"
VMARS_ROOT="/project/peilab/why/V-MARS"
HF_ROOT="${CCRO_ROOT}/artifacts/hf"
DATA="${SMIR_ROOT}/artifacts/data/SMIR-Shadow"
OUT="${SMIR_ROOT}/artifacts/runs/smir_venue"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
STEPS="${STEPS:-2500}"
EVAL_N="${EVAL_N:-150}"
T="${T:-4}"

export VMARS_ROOT
# shellcheck disable=SC1091
source "${VMARS_ROOT}/scripts/env_common.sh"
vmars_setup_modules
# shellcheck disable=SC1091
source "${VMARS_ROOT}/.venv/bin/activate"
vmars_setup_env

export HF_HOME="${HF_ROOT}"
export HUGGINGFACE_HUB_CACHE="${HF_ROOT}/hub"
export HF_HUB_CACHE="${HF_ROOT}/hub"
export TRANSFORMERS_CACHE="${HF_ROOT}/hub"
export TMPDIR="${SMIR_ROOT}/artifacts/tmp"
mkdir -p "${OUT}" "${TMPDIR}" "${SMIR_ROOT}/artifacts/slurm"

if [[ ! -d "${HF_ROOT}/hub/models--Qwen--Qwen2.5-7B-Instruct" ]]; then
  python "${CCRO_ROOT}/llm/download_model.py" --model "${MODEL}" --out-root "${HF_ROOT}"
fi
if [[ ! -f "${DATA}/train.jsonl" ]]; then
  python "${SMIR_ROOT}/data/build_shadow.py" --out-dir "${DATA}" \
    --n-train 8000 --n-val 800 --n-test-id 600 --n-test-ood 600 --seed 0
fi

cd "${SMIR_ROOT}/train"
echo "host=$(hostname) model=${MODEL} steps=${STEPS}"
nvidia-smi -L || true
df -h /project/peilab/why | awk 'NR==2{print}'
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

python run_smir_venue.py \
  --model "${MODEL}" \
  --data-dir "${DATA}" \
  --out-dir "${OUT}" \
  --steps "${STEPS}" \
  --lr 1e-4 \
  --T "${T}" \
  --z-dim 256 \
  --eval-n "${EVAL_N}" \
  --lam-conj 0.5 \
  --lam-tub 0.3 \
  --lam-met 0.2 \
  --lam-norm 0.1 \
  --lam-iv 0.4 \
  --seed 0

echo "==== gates ===="
python -c "import json; d=json.load(open('${OUT}/overall.json')); print(json.dumps({'gates':d['gates'],'comparisons':d['comparisons'],'meta':d['meta']}, indent=2))"
