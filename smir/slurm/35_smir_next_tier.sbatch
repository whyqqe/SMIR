#!/bin/bash
#SBATCH --job-name=smir-next
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
# Next-tier complete pipeline (NeurIPS/ICML track):
#   g* probe + Hard/Soft-anneal/Discrete/MLP ablations × seeds + pack
#
#   sbatch smir/slurm/35_smir_next_tier.sbatch
#   SEEDS=0,1 EXPS=hard,soft sbatch smir/slurm/35_smir_next_tier.sbatch

set -euo pipefail
SMIR_ROOT="/project/peilab/why/NEW/smir"
VMARS_ROOT="/project/peilab/why/V-MARS"
DATA="${SMIR_ROOT}/artifacts/data/SMIR-Shadow-Long"
GRID="${SMIR_ROOT}/artifacts/runs/smir_next_tier"
PACK="${SMIR_ROOT}/artifacts/runs/smir_next_pack"

SEEDS="${SEEDS:-0,1,2}"
EXPS="${EXPS:-hard,soft,discrete,mlp_tf,mlp_task}"
STEPS_A="${STEPS_A:-2500}"
STEPS_B="${STEPS_B:-3500}"
STEPS_C="${STEPS_C:-2000}"
EVAL_N="${EVAL_N:-300}"
P_DIST="${P_DIST:-0.80}"
LAM_FLOW="${LAM_FLOW:-0.5}"
LAM_CONJ="${LAM_CONJ:-0.5}"
LAM_ROUTE="${LAM_ROUTE:-0.5}"
BIAS_START="${BIAS_START:-5.0}"
BIAS_END="${BIAS_END:-0.0}"

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
echo "host=$(hostname) NEXT-TIER SEEDS=${SEEDS} EXPS=${EXPS}"
nvidia-smi -L || true

echo "==== g* probe ===="
python run_smir_gstar_probe.py \
  --data-dir "${DATA}" \
  --out "${GRID}/gstar_probe.json" \
  --n 400 --seed 0

IFS=',' read -r -a SEED_ARR <<< "${SEEDS}"
core_ok=0
for SEED in "${SEED_ARR[@]}"; do
  OUT="${GRID}/seed_${SEED}"
  mkdir -p "${OUT}"
  echo "==== seed=${SEED} exps=${EXPS} ===="
  set +e
  python run_smir_next_tier.py \
    --data-dir "${DATA}" \
    --out-dir "${OUT}" \
    --exps "${EXPS}" \
    --seed "${SEED}" \
    --steps-a "${STEPS_A}" \
    --steps-b "${STEPS_B}" \
    --steps-c "${STEPS_C}" \
    --p-dist "${P_DIST}" \
    --lam-flow "${LAM_FLOW}" \
    --lam-conj "${LAM_CONJ}" \
    --lam-route "${LAM_ROUTE}" \
    --bias-start "${BIAS_START}" \
    --bias-end "${BIAS_END}" \
    --eval-n "${EVAL_N}" \
    --T 5
  rc=$?
  set -e
  echo "seed=${SEED} exit=${rc}"
  if [[ $rc -eq 0 ]]; then
    core_ok=1
  fi
done

echo "==== Pack next-tier ===="
set +e
python run_smir_next_pack.py \
  --grid-root "${GRID}" \
  --runs-root "${SMIR_ROOT}/artifacts/runs" \
  --gstar "${GRID}/gstar_probe.json" \
  --out-dir "${PACK}"
pack_rc=$?
set -e

python - <<'PY'
import json
from pathlib import Path
p=Path("/project/peilab/why/NEW/smir/artifacts/runs/smir_next_pack/next_tier_pack.json")
print(json.dumps(json.load(open(p)), indent=2) if p.exists() else "missing pack")
PY

# also copy compact pack into results/ for git
mkdir -p "${SMIR_ROOT}/results"
if [[ -f "${PACK}/next_tier_pack.json" ]]; then
  cp "${PACK}/next_tier_pack.json" "${SMIR_ROOT}/results/next_tier_pack.json"
fi

if [[ ${pack_rc} -ne 0 ]]; then
  echo "[WARN] next-tier pack core not fully passed"
  exit 2
fi
echo "[OK] next-tier pipeline finished"
