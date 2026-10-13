#!/bin/bash
#SBATCH --job-name=smir-sf
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=20:00:00
#SBATCH --output=/project/peilab/why/NEW/smir/artifacts/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/NEW/smir/artifacts/slurm/%x-%j.err
#
# Exp-1+2: Phase-S → Phase-F (+ patched certification eval)
#   sbatch smir/slurm/24_smir_sf.sbatch
#   MODE=phase_s_only sbatch smir/slurm/24_smir_sf.sbatch

set -euo pipefail
SMIR_ROOT="/project/peilab/why/NEW/smir"
CCRO_ROOT="/project/peilab/why/NEW/ccro"
VMARS_ROOT="/project/peilab/why/V-MARS"
HF_ROOT="${CCRO_ROOT}/artifacts/hf"
DATA="${SMIR_ROOT}/artifacts/data/SMIR-Shadow"
OUT="${SMIR_ROOT}/artifacts/runs/smir_sf"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MODE="${MODE:-full}"
STEPS_S="${STEPS_S:-1500}"
STEPS_F="${STEPS_F:-2500}"
EVAL_N="${EVAL_N:-200}"
T="${T:-4}"
Z_DIM="${Z_DIM:-128}"
LAM_FLOW="${LAM_FLOW:-1.0}"
LAM_END="${LAM_END:-0.1}"
METRIC_ALIGN="${METRIC_ALIGN:-0}"
FREEZE_E="${FREEZE_E_IN_F:-0}"

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
echo "host=$(hostname) MODE=${MODE} S=${STEPS_S} F=${STEPS_F} LAM_END=${LAM_END}"
nvidia-smi -L || true
df -h /project/peilab/why | awk 'NR==2{print}'

python run_smir_sf.py \
  --model "${MODEL}" \
  --data-dir "${DATA}" \
  --out-dir "${OUT}" \
  --mode "${MODE}" \
  --steps-s "${STEPS_S}" \
  --steps-f "${STEPS_F}" \
  --T "${T}" \
  --z-dim "${Z_DIM}" \
  --lam-flow "${LAM_FLOW}" \
  --lam-end "${LAM_END}" \
  --freeze-e-in-f "${FREEZE_E}" \
  --metric-align "${METRIC_ALIGN}" \
  --eval-n "${EVAL_N}" \
  --seed 0

echo "==== gates ===="
python -c "import json; d=json.load(open('${OUT}/overall.json')); print(json.dumps({'gates':d['gates'],'meta':d['meta']}, indent=2))"
