#!/bin/bash
#SBATCH --job-name=smir-pub
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/project/peilab/why/NEW/smir/artifacts/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/NEW/smir/artifacts/slurm/%x-%j.err
#
# Highest-success publication pipeline (Thm5/6/7 + Prop8):
#   E1 Hard (reuse AAAI ckpt on seed0) → Discrete/Soft gated → Joint neg → diag → charts
#
#   sbatch smir/slurm/36_smir_pub_core.sbatch

set -euo pipefail
SMIR_ROOT="/project/peilab/why/NEW/smir"
VMARS_ROOT="/project/peilab/why/V-MARS"
DATA="${SMIR_ROOT}/artifacts/data/SMIR-Shadow-Long"
GRID="${SMIR_ROOT}/artifacts/runs/smir_pub_core"
PACK="${SMIR_ROOT}/artifacts/runs/smir_pub_pack"
E1_CKPT="${E1_CKPT:-${SMIR_ROOT}/artifacts/runs/smir_aaai_grid/lam_0.5/final}"

SEEDS="${SEEDS:-0,1,2}"
EXPS="${EXPS:-hard_e1,discrete_gated,soft_gated,soft_joint,diag,charts}"
STEPS_E1_1="${STEPS_E1_1:-3000}"
STEPS_E1_2="${STEPS_E1_2:-5000}"
STEPS_V="${STEPS_V:-3500}"
EVAL_N="${EVAL_N:-300}"
P_DIST="${P_DIST:-0.75}"
LAM_FLOW="${LAM_FLOW:-0.5}"
GATE_F="${GATE_F:-0.55}"

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
  python "${SMIR_ROOT}/data/build_shadow.py" \
    --out-dir "${DATA}" \
    --n-train 10000 --n-val 800 --n-test-id 600 --n-test-ood 600 \
    --train-lens "2,3,4,5" --ood-lens "4,5" --seed 1
fi

cd "${SMIR_ROOT}/train"
echo "host=$(hostname) PUB-CORE SEEDS=${SEEDS} EXPS=${EXPS} E1_CKPT=${E1_CKPT}"
nvidia-smi -L || true

IFS=',' read -r -a SEED_ARR <<< "${SEEDS}"
for SEED in "${SEED_ARR[@]}"; do
  OUT="${GRID}/seed_${SEED}"
  mkdir -p "${OUT}"
  echo "==== seed=${SEED} ===="
  set +e
  python run_smir_pub_core.py \
    --data-dir "${DATA}" \
    --out-dir "${OUT}" \
    --exps "${EXPS}" \
    --seed "${SEED}" \
    --e1-ckpt "${E1_CKPT}" \
    --reuse-seed 0 \
    --steps-e1-1 "${STEPS_E1_1}" \
    --steps-e1-2 "${STEPS_E1_2}" \
    --steps-v "${STEPS_V}" \
    --p-dist "${P_DIST}" \
    --lam-flow "${LAM_FLOW}" \
    --gate-f "${GATE_F}" \
    --eval-n "${EVAL_N}" \
    --T 5
  echo "seed=${SEED} exit=$?"
  set -e
done

echo "==== Pack ===="
set +e
python run_smir_pub_pack.py \
  --grid-root "${GRID}" \
  --out-dir "${PACK}"
pack_rc=$?
set -e

mkdir -p "${SMIR_ROOT}/results"
if [[ -f "${PACK}/pub_pack.json" ]]; then
  cp "${PACK}/pub_pack.json" "${SMIR_ROOT}/results/pub_pack.json"
fi

python - <<'PY'
import json
from pathlib import Path
p=Path("/project/peilab/why/NEW/smir/artifacts/runs/smir_pub_pack/pub_pack.json")
print(json.dumps(json.load(open(p)).get("claims"), indent=2) if p.exists() else "missing")
PY

if [[ ${pack_rc} -ne 0 ]]; then
  echo "[WARN] pub pack core incomplete"
  exit 2
fi
echo "[OK] publication core pipeline finished"
