#!/bin/bash
#SBATCH --job-name=smir-nips
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
# NeurIPS/ICML-oriented continuation theory suite (new_theory.md):
#   A feasibility | B identifiability | C bottleneck | D scope/graph/charts
#
#   sbatch smir/slurm/39_smir_nips.sbatch

set -euo pipefail
SMIR_ROOT="/project/peilab/why/NEW/smir"
VMARS_ROOT="/project/peilab/why/V-MARS"
REG_DATA="${SMIR_ROOT}/artifacts/data/SMIR-Shadow-Long"
GRAPH_DATA="${SMIR_ROOT}/artifacts/data/SMIR-Graph-Shadow"
GRID="${SMIR_ROOT}/artifacts/runs/smir_nips"
PACK="${SMIR_ROOT}/artifacts/runs/smir_nips_pack"

# Default: single seed for first pass; override with SEEDS=0,1,2 for multiseed.
SEEDS="${SEEDS:-0}"
PHASES="${PHASES:-A,B,C,D}"
STEPS_1="${STEPS_1:-2500}"
STEPS_2="${STEPS_2:-3500}"
STEPS_V="${STEPS_V:-2500}"
EVAL_N="${EVAL_N:-300}"
SKIP_GRAPH="${SKIP_GRAPH:-0}"

export VMARS_ROOT
# shellcheck disable=SC1091
source "${VMARS_ROOT}/scripts/env_common.sh"
vmars_setup_modules
# shellcheck disable=SC1091
source "${VMARS_ROOT}/.venv/bin/activate"
vmars_setup_env

export TMPDIR="${SMIR_ROOT}/artifacts/tmp"
mkdir -p "${GRID}" "${PACK}" "${TMPDIR}" "${SMIR_ROOT}/artifacts/slurm"

if [[ ! -f "${REG_DATA}/train.jsonl" ]]; then
  python "${SMIR_ROOT}/data/build_shadow.py" \
    --out-dir "${REG_DATA}" \
    --n-train 10000 --n-val 800 --n-test-id 600 --n-test-ood 600 \
    --train-lens "2,3,4,5" --ood-lens "4,5" --seed 1
fi
if [[ ! -f "${GRAPH_DATA}/train.jsonl" ]]; then
  python "${SMIR_ROOT}/data/build_graph_shadow.py" \
    --out-dir "${GRAPH_DATA}" \
    --n-train 8000 --n-val 800 --n-test-id 600 --n-test-ood 600 \
    --train-lens "3,4,5,6" --ood-lens "6,7,8" --seed 1
fi

cd "${SMIR_ROOT}/train"
echo "host=$(hostname) NIPS-SUITE SEEDS=${SEEDS} PHASES=${PHASES}"
nvidia-smi -L || true

IFS=',' read -r -a SEED_ARR <<< "${SEEDS}"
for SEED in "${SEED_ARR[@]}"; do
  OUT="${GRID}/seed_${SEED}"
  mkdir -p "${OUT}"
  echo "==== NIPS seed=${SEED} ===="
  set +e
  EXTRA=()
  if [[ "${SKIP_GRAPH}" == "1" ]]; then
    EXTRA+=(--skip-graph)
  fi
  python run_smir_nips_suite.py \
    --reg-data "${REG_DATA}" \
    --graph-data "${GRAPH_DATA}" \
    --out-dir "${OUT}" \
    --seed "${SEED}" \
    --phases "${PHASES}" \
    --steps-1 "${STEPS_1}" \
    --steps-2 "${STEPS_2}" \
    --steps-v "${STEPS_V}" \
    --eval-n "${EVAL_N}" \
    "${EXTRA[@]}"
  echo "seed=${SEED} exit=$?"
  set -e
done

echo "==== Pack ===="
set +e
python run_smir_nips_pack.py --grid-root "${GRID}" --out-dir "${PACK}"
pack_rc=$?
set -e

mkdir -p "${SMIR_ROOT}/results"
if [[ -f "${PACK}/nips_pack.json" ]]; then
  cp "${PACK}/nips_pack.json" "${SMIR_ROOT}/results/nips_pack.json"
fi

python - <<'PY'
import json
from pathlib import Path
p=Path("/project/peilab/why/NEW/smir/artifacts/runs/smir_nips_pack/nips_pack.json")
print(json.dumps(json.load(open(p)).get("claims"), indent=2) if p.exists() else "missing")
PY

if [[ ${pack_rc} -ne 0 ]]; then
  echo "[WARN] NIPS pack core incomplete"
  exit 2
fi
echo "[OK] NIPS continuation-theory suite finished"
