#!/bin/bash
#SBATCH --job-name=smir-vstar
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/project/peilab/why/NEW/smir/artifacts/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/NEW/smir/artifacts/slurm/%x-%j.err
#
# SMIR-V*: freeze iso E/Q, absolute teacher-force V (biggest gap)
#   sbatch smir/slurm/30_smir_vstar.sbatch

set -euo pipefail
SMIR_ROOT="/project/peilab/why/NEW/smir"
VMARS_ROOT="/project/peilab/why/V-MARS"
DATA="${SMIR_ROOT}/artifacts/data/SMIR-Shadow-Struct"
INIT="${INIT_CKPT:-${SMIR_ROOT}/artifacts/runs/smir_iso_cert/stage_a}"
OUT="${SMIR_ROOT}/artifacts/runs/smir_vstar"
STEPS="${STEPS:-8000}"
BATCH="${BATCH:-64}"
LR="${LR:-3e-4}"
EVAL_N="${EVAL_N:-300}"
LAM_FLOW="${LAM_FLOW:-1.0}"
LAM_END="${LAM_END:-0.5}"
LAM_TASK="${LAM_TASK:-0.15}"

export VMARS_ROOT
# shellcheck disable=SC1091
source "${VMARS_ROOT}/scripts/env_common.sh"
vmars_setup_modules
# shellcheck disable=SC1091
source "${VMARS_ROOT}/.venv/bin/activate"
vmars_setup_env

export TMPDIR="${SMIR_ROOT}/artifacts/tmp"
mkdir -p "${OUT}" "${TMPDIR}" "${SMIR_ROOT}/artifacts/slurm"

if [[ ! -f "${DATA}/train.jsonl" ]]; then
  python "${SMIR_ROOT}/data/build_shadow.py" --out-dir "${DATA}" \
    --n-train 8000 --n-val 800 --n-test-id 600 --n-test-ood 600 --seed 0
fi

cd "${SMIR_ROOT}/train"
echo "host=$(hostname) V* INIT=${INIT} STEPS=${STEPS} LR=${LR}"
nvidia-smi -L || true

INIT_ARG=()
if [[ -f "${INIT}/chart.pt" ]]; then
  INIT_ARG=(--init-ckpt "${INIT}")
else
  echo "[WARN] no ${INIT}/chart.pt — will fallback Stage-A inside runner"
fi

python run_smir_vstar.py \
  --data-dir "${DATA}" \
  "${INIT_ARG[@]}" \
  --out-dir "${OUT}" \
  --steps "${STEPS}" \
  --batch "${BATCH}" \
  --lr "${LR}" \
  --lam-flow "${LAM_FLOW}" \
  --lam-end "${LAM_END}" \
  --lam-task "${LAM_TASK}" \
  --eval-n "${EVAL_N}" \
  --seed 0

python -c "import json; d=json.load(open('${OUT}/overall.json')); print(json.dumps({'gates':d['gates'],'comparisons':d['comparisons'],'meta':d['meta']}, indent=2))"
