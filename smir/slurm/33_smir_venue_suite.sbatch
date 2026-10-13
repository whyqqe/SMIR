#!/bin/bash
#SBATCH --job-name=smir-suite
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=/project/peilab/why/NEW/smir/artifacts/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/NEW/smir/artifacts/slurm/%x-%j.err
#
# Multi-experiment venue suite:
#   long-chain data → E1 RegV-OOD → E2 SoftV → E3 MLP-TF → E4 MLP-Task → summary
#
#   sbatch smir/slurm/33_smir_venue_suite.sbatch

set -euo pipefail
SMIR_ROOT="/project/peilab/why/NEW/smir"
VMARS_ROOT="/project/peilab/why/V-MARS"
DATA="${SMIR_ROOT}/artifacts/data/SMIR-Shadow-Long"
OUT="${SMIR_ROOT}/artifacts/runs/smir_venue_suite"
STEPS_E1="${STEPS_E1:-7000}"
STEPS_E2="${STEPS_E2:-6000}"
STEPS_E3="${STEPS_E3:-6000}"
STEPS_E4="${STEPS_E4:-4000}"
EVAL_N="${EVAL_N:-300}"
SKIP="${SKIP:-}"

export VMARS_ROOT
# shellcheck disable=SC1091
source "${VMARS_ROOT}/scripts/env_common.sh"
vmars_setup_modules
# shellcheck disable=SC1091
source "${VMARS_ROOT}/.venv/bin/activate"
vmars_setup_env

export TMPDIR="${SMIR_ROOT}/artifacts/tmp"
mkdir -p "${OUT}" "${TMPDIR}" "${SMIR_ROOT}/artifacts/slurm" "${DATA}"

echo "==== Build long-chain shadow data ===="
python "${SMIR_ROOT}/data/build_shadow.py" \
  --out-dir "${DATA}" \
  --n-train 10000 --n-val 800 --n-test-id 600 --n-test-ood 600 \
  --train-lens "2,3,4,5" --ood-lens "4,5" --seed 1

cd "${SMIR_ROOT}/train"
echo "==== Venue suite ===="
nvidia-smi -L || true

EXTRA=()
if [[ -n "${SKIP}" ]]; then
  EXTRA+=(--skip "${SKIP}")
fi

python run_smir_venue_suite.py \
  --data-dir "${DATA}" \
  --out-dir "${OUT}" \
  --steps-e1 "${STEPS_E1}" \
  --steps-e2 "${STEPS_E2}" \
  --steps-e3 "${STEPS_E3}" \
  --steps-e4 "${STEPS_E4}" \
  --eval-n "${EVAL_N}" \
  --T 5 \
  --seed 0 \
  "${EXTRA[@]}"

python -c "import json; d=json.load(open('${OUT}/summary.json')); print(json.dumps({'claims':d['claims'],'experiments':d['experiments'],'best':d.get('best',{}).get('name')}, indent=2))"
