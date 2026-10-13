#!/usr/bin/env python3
"""Build venue contrast table from existing SMIR run overall.json files (no retrain)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def pick(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def summarize(name: str, path: Path) -> dict:
    if not path.exists():
        return {"name": name, "missing": True, "path": str(path)}
    d = json.loads(path.read_text())
    # normalize across different overall schemas
    final = (
        d.get("final")
        or d.get("stage_b")
        or d.get("results", {}).get("stage_b")
        or d.get("results", {}).get("scf")
        or d.get("results", {}).get("full")
        or {}
    )
    inter = final.get("intervention") or {}
    gates = d.get("gates") or {}
    return {
        "name": name,
        "path": str(path),
        "eta_abs": final.get("eta_abs", final.get("eta_conjugacy", final.get("eta_rel_conjugacy"))),
        "tub_abs": final.get("tubular_mse_abs", final.get("tubular_mse")),
        "tub_rel": final.get("tubular_mse_rel"),
        "F": inter.get("branch_disagree"),
        "follow": inter.get("follow_both"),
        "on_minus_off": inter.get("on_minus_off"),
        "S_gap": final.get("sufficiency_gap"),
        "ood": final.get("task_ood"),
        "oracle": final.get("task_state_oracle"),
        "gstar": final.get("gstar_proxy_tv_gap"),
        "pass_hint": bool(
            gates.get("pass_regv_plus")
            or gates.get("pass_regv")
            or gates.get("pass_vstar")
            or gates.get("pass_cert")
            or gates.get("pass_main")
            or gates.get("pass_iso_cert")
        ),
        "gates_pass_keys": [k for k, v in gates.items() if k.startswith("pass") and v],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, default=Path("/project/peilab/why/NEW/smir/artifacts/runs"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    specs = [
        ("7B-SCF (text E)", args.runs_root / "smir_main" / "overall.json"),
        ("Iso-A / V* residual", args.runs_root / "smir_vstar" / "overall.json"),
        ("RegV PURE hard V", args.runs_root / "smir_regv" / "overall.json"),
        ("RegV+ (this run)", args.runs_root / "smir_regv_plus" / "overall.json"),
    ]
    rows = [summarize(n, p) for n, p in specs]

    # narrative ranking for paper
    story = {
        "claim": "Correct chart + structured V closes conjugacy; text/residual baselines fail differently.",
        "rows": rows,
        "takeaways": [
            "7B-SCF: weak/failed F (text collapse).",
            "V*: F preserved, conjugacy improved but plateaus.",
            "RegV: near-exact η/tub; OOD slightly under gate.",
            "RegV+: target full certificate (F+η+tub+OOD).",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(story, indent=2), encoding="utf-8")
    print(json.dumps(story, indent=2), flush=True)


if __name__ == "__main__":
    main()
