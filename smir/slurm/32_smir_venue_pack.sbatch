#!/bin/bash
#SBATCH --job-name=smir-pack
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/project/peilab/why/NEW/smir/artifacts/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/NEW/smir/artifacts/slurm/%x-%j.err
#
# Next important step: RegV+ (pass certificate) + contrast table pack
#   sbatch smir/slurm/32_smir_venue_pack.sbatch

set -euo pipefail
SMIR_ROOT="/project/peilab/why/NEW/smir"
VMARS_ROOT="/project/peilab/why/V-MARS"
DATA="${SMIR_ROOT}/artifacts/data/SMIR-Shadow-Struct"
OUT="${SMIR_ROOT}/artifacts/runs/smir_regv_plus"
PACK="${SMIR_ROOT}/artifacts/runs/smir_venue_pack"
STEPS_A="${STEPS_A:-6000}"
STEPS_B="${STEPS_B:-2000}"
EVAL_N="${EVAL_N:-300}"

export VMARS_ROOT
# shellcheck disable=SC1091
source "${VMARS_ROOT}/scripts/env_common.sh"
vmars_setup_modules
# shellcheck disable=SC1091
source "${VMARS_ROOT}/.venv/bin/activate"
vmars_setup_env

export TMPDIR="${SMIR_ROOT}/artifacts/tmp"
mkdir -p "${OUT}" "${PACK}" "${TMPDIR}" "${SMIR_ROOT}/artifacts/slurm"

if [[ ! -f "${DATA}/train.jsonl" ]]; then
  python "${SMIR_ROOT}/data/build_shadow.py" --out-dir "${DATA}" \
    --n-train 8000 --n-val 800 --n-test-id 600 --n-test-ood 600 --seed 0
fi

cd "${SMIR_ROOT}/train"
echo "==== RegV+ ===="
python run_smir_regv_plus.py \
  --data-dir "${DATA}" \
  --out-dir "${OUT}" \
  --steps-a "${STEPS_A}" \
  --steps-b "${STEPS_B}" \
  --eval-n "${EVAL_N}" \
  --seed 0 || true

echo "==== Contrast pack ===="
python run_smir_contrast.py \
  --runs-root "${SMIR_ROOT}/artifacts/runs" \
  --out "${PACK}/contrast_table.json"

python - <<'PY'
import json
from pathlib import Path
out=Path("/project/peilab/why/NEW/smir/artifacts/runs/smir_regv_plus/overall.json")
pack=Path("/project/peilab/why/NEW/smir/artifacts/runs/smir_venue_pack/contrast_table.json")
summary={"regv_plus_gates": None, "contrast": None}
if out.exists():
    d=json.load(open(out)); summary["regv_plus_gates"]=d.get("gates")
if pack.exists():
    summary["contrast"]=json.load(open(pack))
print(json.dumps(summary, indent=2))
# fail job only if regv+ missing or not pass
if not out.exists() or not summary["regv_plus_gates"] or not summary["regv_plus_gates"].get("pass_regv_plus"):
    raise SystemExit(2)
print("[OK] venue pack: RegV+ passed + contrast table written")
PY
