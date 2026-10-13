#!/bin/bash
#SBATCH --job-name=smir-cont
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
# Preliminary feasibility of new_theory.md:
#   single L_continuation objective + §11 evaluation battery
#   register: hard / discrete / mlp ; graph: ghard / gdisc
#
#   sbatch smir/slurm/38_smir_continuation.sbatch

set -euo pipefail
SMIR_ROOT="/project/peilab/why/NEW/smir"
VMARS_ROOT="/project/peilab/why/V-MARS"
REG_DATA="${SMIR_ROOT}/artifacts/data/SMIR-Shadow-Long"
GRAPH_DATA="${SMIR_ROOT}/artifacts/data/SMIR-Graph-Shadow"
GRID="${SMIR_ROOT}/artifacts/runs/smir_continuation"
PACK="${SMIR_ROOT}/artifacts/runs/smir_continuation_pack"

SEEDS="${SEEDS:-0,1,2}"
KINDS="${KINDS:-hard,discrete,mlp}"
GRAPH_KINDS="${GRAPH_KINDS:-ghard,gdisc}"
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
  echo "==== build register shadow ===="
  python "${SMIR_ROOT}/data/build_shadow.py" \
    --out-dir "${REG_DATA}" \
    --n-train 10000 --n-val 800 --n-test-id 600 --n-test-ood 600 \
    --train-lens "2,3,4,5" --ood-lens "4,5" --seed 1
fi

if [[ ! -f "${GRAPH_DATA}/train.jsonl" ]]; then
  echo "==== build graph shadow ===="
  python "${SMIR_ROOT}/data/build_graph_shadow.py" \
    --out-dir "${GRAPH_DATA}" \
    --n-train 8000 --n-val 800 --n-test-id 600 --n-test-ood 600 \
    --train-lens "3,4,5,6" --ood-lens "6,7,8" --seed 1
fi

cd "${SMIR_ROOT}/train"
echo "host=$(hostname) CONTINUATION SEEDS=${SEEDS} KINDS=${KINDS} GRAPH_KINDS=${GRAPH_KINDS}"
nvidia-smi -L || true

IFS=',' read -r -a SEED_ARR <<< "${SEEDS}"
for SEED in "${SEED_ARR[@]}"; do
  OUT="${GRID}/seed_${SEED}"
  mkdir -p "${OUT}"
  echo "==== continuation seed=${SEED} ===="
  set +e
  EXTRA=()
  if [[ "${SKIP_GRAPH}" == "1" ]]; then
    EXTRA+=(--skip-graph)
  fi
  python run_smir_continuation.py \
    --reg-data "${REG_DATA}" \
    --graph-data "${GRAPH_DATA}" \
    --out-dir "${OUT}" \
    --seed "${SEED}" \
    --kinds "${KINDS}" \
    --graph-kinds "${GRAPH_KINDS}" \
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
python run_smir_continuation_pack.py \
  --grid-root "${GRID}" \
  --out-dir "${PACK}"
pack_rc=$?
set -e

mkdir -p "${SMIR_ROOT}/results"
if [[ -f "${PACK}/continuation_pack.json" ]]; then
  cp "${PACK}/continuation_pack.json" "${SMIR_ROOT}/results/continuation_pack.json"
fi

python - <<'PY'
import json
from pathlib import Path
p=Path("/project/peilab/why/NEW/smir/artifacts/runs/smir_continuation_pack/continuation_pack.json")
print(json.dumps(json.load(open(p)).get("claims"), indent=2) if p.exists() else "missing")
PY

if [[ ${pack_rc} -ne 0 ]]; then
  echo "[WARN] continuation pack incomplete"
  exit 2
fi
echo "[OK] continuation theory feasibility pipeline finished"
