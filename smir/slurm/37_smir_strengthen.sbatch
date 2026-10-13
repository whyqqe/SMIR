#!/bin/bash
#SBATCH --job-name=smir-str
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
# Defect-fix strengthen pipeline:
#   1) Graph second domain (multi-seed)
#   2) Chart adapter weak-positive (legal chart recovery vs random text)
#   3) Pack with register pub-core results
#
#   sbatch smir/slurm/37_smir_strengthen.sbatch

set -euo pipefail
SMIR_ROOT="/project/peilab/why/NEW/smir"
VMARS_ROOT="/project/peilab/why/V-MARS"
REG_DATA="${SMIR_ROOT}/artifacts/data/SMIR-Shadow-Long"
GRAPH_DATA="${SMIR_ROOT}/artifacts/data/SMIR-Graph-Shadow"
GRAPH_RUNS="${SMIR_ROOT}/artifacts/runs/smir_graph_pub"
REG_RUNS="${SMIR_ROOT}/artifacts/runs/smir_pub_core"
PACK="${SMIR_ROOT}/artifacts/runs/smir_strengthen_pack"
ADAPTER_DIR="${PACK}/adapter"
E1_CKPT="${E1_CKPT:-${SMIR_ROOT}/artifacts/runs/smir_aaai_grid/lam_0.5/final}"

SEEDS="${SEEDS:-0,1,2}"
STEPS_1="${STEPS_1:-2500}"
STEPS_2="${STEPS_2:-3500}"
STEPS_V="${STEPS_V:-3000}"
ADAPTER_STEPS="${ADAPTER_STEPS:-2000}"
EVAL_N="${EVAL_N:-300}"

export VMARS_ROOT
# shellcheck disable=SC1091
source "${VMARS_ROOT}/scripts/env_common.sh"
vmars_setup_modules
# shellcheck disable=SC1091
source "${VMARS_ROOT}/.venv/bin/activate"
vmars_setup_env

export TMPDIR="${SMIR_ROOT}/artifacts/tmp"
mkdir -p "${GRAPH_RUNS}" "${PACK}" "${ADAPTER_DIR}" "${TMPDIR}" "${SMIR_ROOT}/artifacts/slurm"

if [[ ! -f "${GRAPH_DATA}/train.jsonl" ]]; then
  echo "==== build Graph Shadow ===="
  python "${SMIR_ROOT}/data/build_graph_shadow.py" \
    --out-dir "${GRAPH_DATA}" \
    --n-train 8000 --n-val 800 --n-test-id 600 --n-test-ood 600 \
    --train-lens "3,4,5,6" --ood-lens "6,7,8" --seed 1
fi

cd "${SMIR_ROOT}/train"
echo "host=$(hostname) STRENGTHEN SEEDS=${SEEDS}"
nvidia-smi -L || true

IFS=',' read -r -a SEED_ARR <<< "${SEEDS}"
for SEED in "${SEED_ARR[@]}"; do
  OUT="${GRAPH_RUNS}/seed_${SEED}"
  mkdir -p "${OUT}"
  echo "==== graph seed=${SEED} ===="
  set +e
  python run_smir_graph_pub.py \
    --data-dir "${GRAPH_DATA}" \
    --out-dir "${OUT}" \
    --seed "${SEED}" \
    --steps-1 "${STEPS_1}" \
    --steps-2 "${STEPS_2}" \
    --steps-v "${STEPS_V}" \
    --eval-n "${EVAL_N}" \
    --T 8
  echo "graph seed=${SEED} exit=$?"
  set -e
done

echo "==== chart adapter weak-positive ===="
set +e
python run_smir_chart_adapter.py \
  --data-dir "${REG_DATA}" \
  --hard-ckpt "${E1_CKPT}" \
  --out-dir "${ADAPTER_DIR}" \
  --steps "${ADAPTER_STEPS}" \
  --eval-n "${EVAL_N}" \
  --seed 0
echo "adapter exit=$?"
set -e

echo "==== pack ===="
set +e
python run_smir_strengthen_pack.py \
  --reg-root "${REG_RUNS}" \
  --graph-root "${GRAPH_RUNS}" \
  --adapter "${ADAPTER_DIR}/chart_adapter.json" \
  --out-dir "${PACK}"
pack_rc=$?
set -e

mkdir -p "${SMIR_ROOT}/results"
if [[ -f "${PACK}/strengthen_pack.json" ]]; then
  cp "${PACK}/strengthen_pack.json" "${SMIR_ROOT}/results/strengthen_pack.json"
fi

python - <<'PY'
import json
from pathlib import Path
p=Path("/project/peilab/why/NEW/smir/artifacts/runs/smir_strengthen_pack/strengthen_pack.json")
print(json.dumps(json.load(open(p)).get("claims"), indent=2) if p.exists() else "missing")
PY

if [[ ${pack_rc} -ne 0 ]]; then
  echo "[WARN] strengthen pack incomplete (may wait for register pub-core seeds)"
  exit 2
fi
echo "[OK] strengthen pipeline finished"
