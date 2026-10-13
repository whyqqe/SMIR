#!/bin/bash
#SBATCH --job-name=smir-data
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/project/peilab/why/NEW/smir/artifacts/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/NEW/smir/artifacts/slurm/%x-%j.err

set -euo pipefail
SMIR_ROOT="/project/peilab/why/NEW/smir"
VMARS_ROOT="/project/peilab/why/V-MARS"
OUT="${SMIR_ROOT}/artifacts/data/SMIR-Shadow-Struct"

export VMARS_ROOT
# shellcheck disable=SC1091
source "${VMARS_ROOT}/scripts/env_common.sh"
vmars_setup_modules
# shellcheck disable=SC1091
source "${VMARS_ROOT}/.venv/bin/activate"
vmars_setup_env

mkdir -p "${OUT}" "${SMIR_ROOT}/artifacts/slurm"
python "${SMIR_ROOT}/data/build_shadow.py" --out-dir "${OUT}" \
  --n-train 8000 --n-val 800 --n-test-id 600 --n-test-ood 600 --seed 0
python -c "import json; from pathlib import Path; r=json.loads(Path('${OUT}/train.jsonl').open().readline()); print('keys_ok', all(k in r for k in ['continuation','state_final','s0','step_action']))"
