#!/bin/bash
#SBATCH --job-name=smir-regv
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/project/peilab/why/NEW/smir/artifacts/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/NEW/smir/artifacts/slurm/%x-%j.err
#
# SMIR-RegV: identity R^5 chart + structured register V (highest success)
#   sbatch smir/slurm/31_smir_regv.sbatch
#   PURE=1 sbatch ...   # no residual (max conjugacy)

set -euo pipefail
SMIR_ROOT="/project/peilab/why/NEW/smir"
VMARS_ROOT="/project/peilab/why/V-MARS"
DATA="${SMIR_ROOT}/artifacts/data/SMIR-Shadow-Struct"
OUT="${SMIR_ROOT}/artifacts/runs/smir_regv"
STEPS_A="${STEPS_A:-4000}"
STEPS_B="${STEPS_B:-3000}"
BATCH="${BATCH:-64}"
EVAL_N="${EVAL_N:-300}"
PURE="${PURE:-0}"

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
echo "host=$(hostname) RegV STEPS_A=${STEPS_A} STEPS_B=${STEPS_B} PURE=${PURE}"
nvidia-smi -L || true

EXTRA=()
if [[ "${PURE}" == "1" ]]; then
  EXTRA+=(--no-residual)
fi

python run_smir_regv.py \
  --data-dir "${DATA}" \
  --out-dir "${OUT}" \
  --steps-a "${STEPS_A}" \
  --steps-b "${STEPS_B}" \
  --batch "${BATCH}" \
  --eval-n "${EVAL_N}" \
  --seed 0 \
  "${EXTRA[@]}"

python -c "import json; d=json.load(open('${OUT}/overall.json')); print(json.dumps({'gates':d['gates'],'comparisons':d['comparisons'],'meta':d['meta']}, indent=2))"
