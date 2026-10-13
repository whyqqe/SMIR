#!/bin/bash
#SBATCH --job-name=smir-diag
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/project/peilab/why/NEW/smir/artifacts/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/NEW/smir/artifacts/slurm/%x-%j.err
#
# Exp-0: diagnose frozen SCF checkpoint (E vs Q vs tubular)
#   sbatch smir/slurm/23_smir_diag.sbatch

set -euo pipefail
SMIR_ROOT="/project/peilab/why/NEW/smir"
CCRO_ROOT="/project/peilab/why/NEW/ccro"
VMARS_ROOT="/project/peilab/why/V-MARS"
HF_ROOT="${CCRO_ROOT}/artifacts/hf"
DATA="${SMIR_ROOT}/artifacts/data/SMIR-Shadow"
CKPT="${CKPT:-${SMIR_ROOT}/artifacts/runs/smir_main/scf}"
OUT="${OUT:-${SMIR_ROOT}/artifacts/runs/smir_sf/diag_scf.json}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
N="${N:-200}"

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
mkdir -p "${TMPDIR}" "$(dirname "${OUT}")" "${SMIR_ROOT}/artifacts/slurm"

if [[ ! -f "${CKPT}/heads.pt" ]]; then
  echo "Missing ckpt: ${CKPT}/heads.pt" >&2
  exit 1
fi
if [[ ! -f "${DATA}/train.jsonl" ]]; then
  python "${SMIR_ROOT}/data/build_shadow.py" --out-dir "${DATA}" \
    --n-train 8000 --n-val 800 --n-test-id 600 --n-test-ood 600 --seed 0
fi

cd "${SMIR_ROOT}/train"
echo "host=$(hostname) CKPT=${CKPT}"
nvidia-smi -L || true
python run_smir_diag.py \
  --model "${MODEL}" \
  --ckpt "${CKPT}" \
  --data-dir "${DATA}" \
  --out "${OUT}" \
  --n "${N}"

echo "==== diag done ===="
python -c "import json; print(json.dumps(json.load(open('${OUT}'))['verdict'], indent=2))"
