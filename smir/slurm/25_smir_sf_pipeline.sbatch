#!/bin/bash
#SBATCH --job-name=smir-pipe
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=/project/peilab/why/NEW/smir/artifacts/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/NEW/smir/artifacts/slurm/%x-%j.err
#
# Full pipeline: Exp-0 diag → Exp-1 Phase-S/F → Exp-2 patched eval
# (no ContCE/Outcome/plain-SCF re-run)
#
#   sbatch smir/slurm/25_smir_sf_pipeline.sbatch

set -euo pipefail
SMIR_ROOT="/project/peilab/why/NEW/smir"
CCRO_ROOT="/project/peilab/why/NEW/ccro"
VMARS_ROOT="/project/peilab/why/V-MARS"
HF_ROOT="${CCRO_ROOT}/artifacts/hf"
DATA="${SMIR_ROOT}/artifacts/data/SMIR-Shadow"
SCF_CKPT="${SCF_CKPT:-${SMIR_ROOT}/artifacts/runs/smir_main/scf}"
OUT="${SMIR_ROOT}/artifacts/runs/smir_sf"
DIAG_OUT="${OUT}/diag_scf.json"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
STEPS_S="${STEPS_S:-1500}"
STEPS_F="${STEPS_F:-2500}"
EVAL_N="${EVAL_N:-200}"
T="${T:-4}"
Z_DIM="${Z_DIM:-128}"
LAM_FLOW="${LAM_FLOW:-1.0}"
LAM_END="${LAM_END:-0.1}"
METRIC_ALIGN="${METRIC_ALIGN:-0}"
N_DIAG="${N_DIAG:-200}"
SKIP_DIAG="${SKIP_DIAG:-0}"

export VMARS_ROOT
# shellcheck disable=SC1091
source "${VMARS_ROOT}/scripts/env_common.sh"
vmars_setup_modules
# shellcheck disable=SC1091
source "${VMARS_ROOT}/.venv/bin/activate"
vmars_setup_env

export HF_HOME="${HF_ROOT}"
export HUGGINGFACE_HUB_CACHE="${HF_ROOT}/hub"
export HF_HUB_CACHE="${HF_ROOT}/hub"
export TRANSFORMERS_CACHE="${HF_ROOT}/hub"
export TMPDIR="${SMIR_ROOT}/artifacts/tmp"
mkdir -p "${OUT}" "${TMPDIR}" "${SMIR_ROOT}/artifacts/slurm"

if [[ ! -d "${HF_ROOT}/hub/models--Qwen--Qwen2.5-7B-Instruct" ]]; then
  python "${CCRO_ROOT}/llm/download_model.py" --model "${MODEL}" --out-root "${HF_ROOT}"
fi
if [[ ! -f "${DATA}/train.jsonl" ]]; then
  python "${SMIR_ROOT}/data/build_shadow.py" --out-dir "${DATA}" \
    --n-train 8000 --n-val 800 --n-test-id 600 --n-test-ood 600 --seed 0
fi

cd "${SMIR_ROOT}/train"
echo "======== PIPELINE host=$(hostname) ========"
nvidia-smi -L || true
df -h /project/peilab/why | awk 'NR==2{print}'

# ---- Exp-0 ----
if [[ "${SKIP_DIAG}" != "1" ]]; then
  if [[ -f "${SCF_CKPT}/heads.pt" ]]; then
    echo "======== Exp-0: diagnose SCF ckpt ========"
    python run_smir_diag.py \
      --model "${MODEL}" \
      --ckpt "${SCF_CKPT}" \
      --data-dir "${DATA}" \
      --out "${DIAG_OUT}" \
      --n "${N_DIAG}"
    python -c "import json; d=json.load(open('${DIAG_OUT}')); print('[Exp-0 verdict]', json.dumps(d['verdict']))"
  else
    echo "[WARN] no SCF ckpt at ${SCF_CKPT}; skip Exp-0"
  fi
else
  echo "[skip] Exp-0"
fi

# ---- Exp-1 + Exp-2 ----
echo "======== Exp-1/2: Phase-S → Phase-F ========"
python run_smir_sf.py \
  --model "${MODEL}" \
  --data-dir "${DATA}" \
  --out-dir "${OUT}" \
  --mode full \
  --steps-s "${STEPS_S}" \
  --steps-f "${STEPS_F}" \
  --T "${T}" \
  --z-dim "${Z_DIM}" \
  --lam-flow "${LAM_FLOW}" \
  --lam-end "${LAM_END}" \
  --freeze-e-in-f 0 \
  --metric-align "${METRIC_ALIGN}" \
  --eval-n "${EVAL_N}" \
  --seed 0

# ---- post-diag on new full ckpt ----
if [[ -f "${OUT}/full/heads.pt" ]]; then
  echo "======== Exp-0b: diagnose new full ckpt ========"
  python run_smir_diag.py \
    --model "${MODEL}" \
    --ckpt "${OUT}/full" \
    --data-dir "${DATA}" \
    --out "${OUT}/diag_full.json" \
    --n "${N_DIAG}"
fi

echo "======== PIPELINE SUMMARY ========"
python - <<'PY'
import json
from pathlib import Path
out = Path("/project/peilab/why/NEW/smir/artifacts/runs/smir_sf")
summary = {}
for name in ["diag_scf.json", "overall.json", "diag_full.json"]:
    p = out / name
    if not p.exists():
        continue
    d = json.load(open(p))
    if name.startswith("diag"):
        summary[name] = {"verdict": d.get("verdict"), "readout": d.get("readout"), "e_separation": d.get("e_separation")}
    else:
        summary[name] = {"gates": d.get("gates"), "meta": d.get("meta")}
print(json.dumps(summary, indent=2))
PY
