#!/bin/bash
#SBATCH --job-name=smir-cert
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
# SMIR-Cert remaining-goals pipeline (structured chart; no 7B re-run)
#   sbatch smir/slurm/27_smir_cert_pipeline.sbatch

set -euo pipefail
SMIR_ROOT="/project/peilab/why/NEW/smir"
VMARS_ROOT="/project/peilab/why/V-MARS"
DATA="${SMIR_ROOT}/artifacts/data/SMIR-Shadow-Struct"
OUT="${SMIR_ROOT}/artifacts/runs/smir_cert"
STEPS_A="${STEPS_A:-3000}"
STEPS_B="${STEPS_B:-4000}"
BATCH_A="${BATCH_A:-64}"
BATCH_B="${BATCH_B:-32}"
EVAL_N="${EVAL_N:-300}"
T="${T:-4}"
Z_DIM="${Z_DIM:-64}"
LAM_FLOW="${LAM_FLOW:-1.0}"
LAM_END="${LAM_END:-0.5}"
LR_A="${LR_A:-3e-3}"
LR_B="${LR_B:-1e-3}"

export VMARS_ROOT
# shellcheck disable=SC1091
source "${VMARS_ROOT}/scripts/env_common.sh"
vmars_setup_modules
# shellcheck disable=SC1091
source "${VMARS_ROOT}/.venv/bin/activate"
vmars_setup_env

export TMPDIR="${SMIR_ROOT}/artifacts/tmp"
mkdir -p "${OUT}" "${TMPDIR}" "${SMIR_ROOT}/artifacts/slurm"

# Rebuild if missing or leftover smoke-test size (<8000)
NTRAIN=0
if [[ -f "${DATA}/train.jsonl" ]]; then
  NTRAIN=$(wc -l < "${DATA}/train.jsonl")
fi
if [[ ! -f "${DATA}/train.jsonl" || "${NTRAIN}" -lt 8000 || "${FORCE_REBUILD:-0}" == "1" ]]; then
  echo "Building structured shadow data (ntrain=${NTRAIN})..."
  python "${SMIR_ROOT}/data/build_shadow.py" --out-dir "${DATA}" \
    --n-train 8000 --n-val 800 --n-test-id 600 --n-test-ood 600 --seed 0
fi

python -c "import json; from pathlib import Path; p=Path('${DATA}/train.jsonl'); r=json.loads(p.open().readline()); n=sum(1 for _ in p.open());
assert 'continuation' in r and 'state_final' in r, r.keys(); assert n>=8000, n; print('data_ok', n)"

cd "${SMIR_ROOT}/train"
echo "host=$(hostname) STEPS_A=${STEPS_A} STEPS_B=${STEPS_B}"
nvidia-smi -L || true

python run_smir_cert.py \
  --data-dir "${DATA}" \
  --out-dir "${OUT}" \
  --steps-a "${STEPS_A}" \
  --steps-b "${STEPS_B}" \
  --batch-a "${BATCH_A}" \
  --batch-b "${BATCH_B}" \
  --lr-a "${LR_A}" \
  --lr-b "${LR_B}" \
  --T "${T}" \
  --z-dim "${Z_DIM}" \
  --lam-flow "${LAM_FLOW}" \
  --lam-end "${LAM_END}" \
  --eval-n "${EVAL_N}" \
  --seed 0

echo "==== cert gates ===="
python -c "import json; d=json.load(open('${OUT}/overall.json')); print(json.dumps({'gates':d['gates'],'comparisons':d['comparisons'],'meta':d['meta']}, indent=2))"
