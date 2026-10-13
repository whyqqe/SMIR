#!/bin/bash
#SBATCH --job-name=smir-main
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
# SMIR main experiment (unified SCF):
#   L = L_task + λ L_flow
#   vs ContCE (λ=0) and Outcome
#
#   sbatch smir/slurm/22_smir_main.sbatch

set -euo pipefail
SMIR_ROOT="/project/peilab/why/NEW/smir"
CCRO_ROOT="/project/peilab/why/NEW/ccro"
VMARS_ROOT="/project/peilab/why/V-MARS"
HF_ROOT="${CCRO_ROOT}/artifacts/hf"
DATA="${SMIR_ROOT}/artifacts/data/SMIR-Shadow"
OUT="${SMIR_ROOT}/artifacts/runs/smir_main"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
STEPS="${STEPS:-3000}"
EVAL_N="${EVAL_N:-200}"
T="${T:-4}"
Z_DIM="${Z_DIM:-128}"
LAM_FLOW="${LAM_FLOW:-1.0}"

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
echo "host=$(hostname) STEPS=${STEPS} T=${T} Z_DIM=${Z_DIM} LAM_FLOW=${LAM_FLOW}"
nvidia-smi -L || true
df -h /project/peilab/why | awk 'NR==2{print}'
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

python run_smir_main.py \
  --model "${MODEL}" \
  --data-dir "${DATA}" \
  --out-dir "${OUT}" \
  --steps "${STEPS}" \
  --lr 1e-4 \
  --T "${T}" \
  --z-dim "${Z_DIM}" \
  --lam-flow "${LAM_FLOW}" \
  --eval-n "${EVAL_N}" \
  --seed 0

echo "==== gates ===="
python -c "import json; d=json.load(open('${OUT}/overall.json')); print(json.dumps({'gates':d['gates'],'comparisons':d['comparisons'],'meta':d['meta']}, indent=2))"
