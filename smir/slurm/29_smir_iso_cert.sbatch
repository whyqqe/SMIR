#!/bin/bash
#SBATCH --job-name=smir-iso
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
# Iso-Cert: near-isometric chart + relative conjugacy
#   sbatch smir/slurm/29_smir_iso_cert.sbatch

set -euo pipefail
SMIR_ROOT="/project/peilab/why/NEW/smir"
VMARS_ROOT="/project/peilab/why/V-MARS"
DATA="${SMIR_ROOT}/artifacts/data/SMIR-Shadow-Struct"
OUT="${SMIR_ROOT}/artifacts/runs/smir_iso_cert"
STEPS_A="${STEPS_A:-3000}"
STEPS_B="${STEPS_B:-5000}"
BATCH_A="${BATCH_A:-64}"
BATCH_B="${BATCH_B:-32}"
LR_A="${LR_A:-3e-3}"
LR_B="${LR_B:-1e-4}"
EVAL_N="${EVAL_N:-300}"
LAM_FLOW="${LAM_FLOW:-1.0}"
LAM_END="${LAM_END:-0.3}"
LAM_TASK="${LAM_TASK:-0.3}"

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
NTRAIN=$(wc -l < "${DATA}/train.jsonl")
if [[ "${NTRAIN}" -lt 8000 ]]; then
  python "${SMIR_ROOT}/data/build_shadow.py" --out-dir "${DATA}" \
    --n-train 8000 --n-val 800 --n-test-id 600 --n-test-ood 600 --seed 0
fi

cd "${SMIR_ROOT}/train"
echo "host=$(hostname) Iso-Cert STEPS_A=${STEPS_A} STEPS_B=${STEPS_B} LR_B=${LR_B}"
nvidia-smi -L || true

python run_smir_iso_cert.py \
  --data-dir "${DATA}" \
  --out-dir "${OUT}" \
  --steps-a "${STEPS_A}" \
  --steps-b "${STEPS_B}" \
  --batch-a "${BATCH_A}" \
  --batch-b "${BATCH_B}" \
  --lr-a "${LR_A}" \
  --lr-b "${LR_B}" \
  --lam-flow "${LAM_FLOW}" \
  --lam-end "${LAM_END}" \
  --lam-task "${LAM_TASK}" \
  --eval-n "${EVAL_N}" \
  --seed 0

python -c "import json; d=json.load(open('${OUT}/overall.json')); print(json.dumps({'gates':d['gates'],'comparisons':d['comparisons'],'meta':d['meta']}, indent=2))"
