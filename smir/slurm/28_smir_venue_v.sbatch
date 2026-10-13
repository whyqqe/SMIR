#!/bin/bash
#SBATCH --job-name=smir-vv
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/project/peilab/why/NEW/smir/artifacts/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/NEW/smir/artifacts/slurm/%x-%j.err
#
# Venue-V: freeze Stage-A E/Q, train V only (multi-step conjugacy)
#   sbatch smir/slurm/28_smir_venue_v.sbatch

set -euo pipefail
SMIR_ROOT="/project/peilab/why/NEW/smir"
VMARS_ROOT="/project/peilab/why/V-MARS"
DATA="${SMIR_ROOT}/artifacts/data/SMIR-Shadow-Struct"
INIT="${INIT_CKPT:-${SMIR_ROOT}/artifacts/runs/smir_cert/stage_a}"
OUT="${SMIR_ROOT}/artifacts/runs/smir_venue_v"
STEPS="${STEPS:-6000}"
BATCH="${BATCH:-64}"
LR="${LR:-1e-4}"
EVAL_N="${EVAL_N:-300}"
LAM_FLOW="${LAM_FLOW:-1.0}"
LAM_END="${LAM_END:-0.2}"
LAM_TASK="${LAM_TASK:-0.2}"

export VMARS_ROOT
# shellcheck disable=SC1091
source "${VMARS_ROOT}/scripts/env_common.sh"
vmars_setup_modules
# shellcheck disable=SC1091
source "${VMARS_ROOT}/.venv/bin/activate"
vmars_setup_env

export TMPDIR="${SMIR_ROOT}/artifacts/tmp"
mkdir -p "${OUT}" "${TMPDIR}" "${SMIR_ROOT}/artifacts/slurm"

if [[ ! -f "${INIT}/chart.pt" ]]; then
  echo "Missing Stage-A ckpt: ${INIT}/chart.pt" >&2
  exit 1
fi
if [[ ! -f "${DATA}/train.jsonl" ]]; then
  echo "Missing data ${DATA}; run 27 pipeline first" >&2
  exit 1
fi

cd "${SMIR_ROOT}/train"
echo "host=$(hostname) INIT=${INIT} STEPS=${STEPS} LR=${LR}"
nvidia-smi -L || true

python run_smir_venue_v.py \
  --data-dir "${DATA}" \
  --init-ckpt "${INIT}" \
  --out-dir "${OUT}" \
  --steps "${STEPS}" \
  --batch "${BATCH}" \
  --lr "${LR}" \
  --lam-flow "${LAM_FLOW}" \
  --lam-end "${LAM_END}" \
  --lam-task "${LAM_TASK}" \
  --eval-n "${EVAL_N}" \
  --seed 0

echo "==== venue-v gates ===="
python -c "import json; d=json.load(open('${OUT}/overall.json')); print(json.dumps({'gates':d['gates'],'comparisons':d['comparisons'],'oracle':d['oracle_upper_bound'],'meta':d['meta']}, indent=2))"
