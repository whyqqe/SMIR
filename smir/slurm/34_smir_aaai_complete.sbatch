#!/bin/bash
#SBATCH --job-name=smir-aaai
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
# AAAI remaining-work complete pipeline (high success):
#   1) ensure Shadow-Long data
#   2) λ_flow grid {0.3,0.5,1.0} curriculum Hard-V main
#   3) pack Table1/Table2 + 7B negative
#
#   sbatch smir/slurm/34_smir_aaai_complete.sbatch

set -euo pipefail
SMIR_ROOT="/project/peilab/why/NEW/smir"
VMARS_ROOT="/project/peilab/why/V-MARS"
DATA="${SMIR_ROOT}/artifacts/data/SMIR-Shadow-Long"
GRID="${SMIR_ROOT}/artifacts/runs/smir_aaai_grid"
PACK="${SMIR_ROOT}/artifacts/runs/smir_aaai_pack"
STEPS_1="${STEPS_1:-3000}"
STEPS_2="${STEPS_2:-5000}"
EVAL_N="${EVAL_N:-300}"
P_DIST="${P_DIST:-0.75}"

export VMARS_ROOT
# shellcheck disable=SC1091
source "${VMARS_ROOT}/scripts/env_common.sh"
vmars_setup_modules
# shellcheck disable=SC1091
source "${VMARS_ROOT}/.venv/bin/activate"
vmars_setup_env

export TMPDIR="${SMIR_ROOT}/artifacts/tmp"
mkdir -p "${GRID}" "${PACK}" "${TMPDIR}" "${SMIR_ROOT}/artifacts/slurm"

if [[ ! -f "${DATA}/train.jsonl" ]]; then
  echo "==== build Shadow-Long ===="
  python "${SMIR_ROOT}/data/build_shadow.py" \
    --out-dir "${DATA}" \
    --n-train 10000 --n-val 800 --n-test-id 600 --n-test-ood 600 \
    --train-lens "2,3,4,5" --ood-lens "4,5" --seed 1
fi

cd "${SMIR_ROOT}/train"
echo "host=$(hostname) AAAI complete STEPS_1=${STEPS_1} STEPS_2=${STEPS_2} P_DIST=${P_DIST}"
nvidia-smi -L || true

pass_any=0
for LAM in 0.3 0.5 1.0; do
  OUT="${GRID}/lam_${LAM}"
  mkdir -p "${OUT}"
  echo "==== AAAI-Main lam_flow=${LAM} ===="
  set +e
  python run_smir_aaai_main.py \
    --data-dir "${DATA}" \
    --out-dir "${OUT}" \
    --steps-1 "${STEPS_1}" \
    --steps-2 "${STEPS_2}" \
    --p-dist "${P_DIST}" \
    --lam-flow "${LAM}" \
    --eval-n "${EVAL_N}" \
    --T 5 \
    --seed 0
  rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    pass_any=1
  fi
  echo "lam=${LAM} exit=${rc}"
done

echo "==== Pack tables + LLM negative ===="
set +e
python run_smir_aaai_pack.py \
  --runs-root "${SMIR_ROOT}/artifacts/runs" \
  --grid-root "${GRID}" \
  --out-dir "${PACK}"
pack_rc=$?
set -e

python - <<'PY'
import json
from pathlib import Path
p=Path("/project/peilab/why/NEW/smir/artifacts/runs/smir_aaai_pack/aaai_pack.json")
print(json.dumps(json.load(open(p)), indent=2) if p.exists() else "missing pack")
PY

if [[ ${pack_rc} -ne 0 ]]; then
  echo "[WARN] AAAI pack not fully ready (no λ fully passed)"
  exit 2
fi
echo "[OK] AAAI complete pipeline finished"
