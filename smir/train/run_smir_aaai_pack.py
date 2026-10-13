#!/usr/bin/env python3
"""Pack AAAI Table1/Table2 + LLM negative result from existing runs + λ-grid mains."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text())


def from_final(d, prefer=("final", "stage_b", "results")):
    if d is None:
        return {}
    if "final" in d and isinstance(d["final"], dict) and "intervention" in d["final"]:
        return d["final"]
    if "stage_b" in d and isinstance(d["stage_b"], dict):
        return d["stage_b"]
    r = d.get("results") or {}
    for k in ("stage_b", "scf", "full", "E1_RegV_OOD"):
        if k in r and isinstance(r[k], dict):
            return r[k]
    # suite summary experiment entry
    return d


def row(name, m, gates=None, note=""):
    if not m:
        return {"name": name, "missing": True}
    I = m.get("intervention") or {}
    return {
        "name": name,
        "F": I.get("branch_disagree"),
        "follow": I.get("follow_both"),
        "on_minus_off": I.get("on_minus_off"),
        "eta_abs": m.get("eta_abs", m.get("eta_conjugacy")),
        "tub_abs": m.get("tubular_mse_abs", m.get("tubular_mse")),
        "ood": m.get("task_ood"),
        "oracle": m.get("task_state_oracle"),
        "S_gap": m.get("sufficiency_gap"),
        "note": note,
        "pass": bool((gates or {}).get("pass_aaai_main") or (gates or {}).get("pass_regv")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, default=Path("/project/peilab/why/NEW/smir/artifacts/runs"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--grid-root", type=Path, required=True, help="dir containing lam_* /overall.json")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    R = args.runs_root

    # pick best λ from grid
    grid = []
    for p in sorted(args.grid_root.glob("lam_*/overall.json")):
        d = load(p)
        if not d:
            continue
        grid.append(
            {
                "path": str(p),
                "lam_flow": d.get("meta", {}).get("lam_flow"),
                "score": d.get("score", 0),
                "gates": d.get("gates"),
                "final": d.get("final"),
            }
        )
    best = (
        max(
            grid,
            key=lambda x: (
                bool((x.get("gates") or {}).get("pass_aaai_main")),
                float(x.get("score") or 0),
            ),
        )
        if grid
        else None
    )

    # LLM negative from smir_main
    main7 = load(R / "smir_main" / "overall.json")
    scf = None
    if main7:
        scf = (main7.get("results") or {}).get("scf") or {}
    llm_neg = {
        "name": "7B-SCF text encoder (negative)",
        "protocol": "same shadow labels; last-token E; Q(z_T)",
        "F": (scf.get("intervention") or {}).get("branch_disagree") if scf else None,
        "eta_conjugacy": scf.get("eta_conjugacy") if scf else None,
        "tubular_mse": scf.get("tubular_mse") if scf else None,
        "ood": scf.get("task_ood") if scf else None,
        "claim": "Standard LLM last-token chart fails intervention faithfulness (F≈0); tubular unstable.",
    }

    # Table2 contrast
    suite = load(R / "smir_venue_suite" / "summary.json")
    suite_exps = {e["name"]: e for e in (suite or {}).get("experiments", [])}

    def suite_row(name, note=""):
        e = suite_exps.get(name)
        if not e:
            return {"name": name, "missing": True}
        return {
            "name": name,
            "F": e.get("F"),
            "eta_abs": e.get("eta"),
            "tub_abs": e.get("tub"),
            "ood": e.get("ood"),
            "S_gap": e.get("S_gap"),
            "note": note,
        }

    table1 = row(
        "AAAI-Main (best λ)",
        best["final"] if best else None,
        best.get("gates") if best else None,
        note=f"lam_flow={best.get('lam_flow')}" if best else "",
    )
    table2 = [
        row(
            "7B-SCF",
            scf,
            note="text E collapse",
        ),
        row(
            "V*",
            (load(R / "smir_vstar" / "overall.json") or {}).get("final"),
            note="iso chart + residual V plateau",
        ),
        row(
            "RegV-PURE",
            (load(R / "smir_regv" / "overall.json") or {}).get("results", {}).get("stage_b")
            or (load(R / "smir_regv" / "overall.json") or {}).get("stage_b"),
            note="hard V short/medium",
        ),
        suite_row("E2_SoftV", "learnable soft routing"),
        suite_row("E3_MLP_TeacherForce", "MLP + teacher"),
        suite_row("E4_MLP_TaskOnly", "ablation: no teacher"),
        table1,
    ]

    # also list all grid members
    grid_rows = [
        {
            "lam_flow": g["lam_flow"],
            "score": g["score"],
            "pass": bool((g.get("gates") or {}).get("pass_aaai_main")),
            "F": (g.get("final") or {}).get("intervention", {}).get("branch_disagree"),
            "ood": (g.get("final") or {}).get("task_ood"),
            "eta": (g.get("final") or {}).get("eta_abs"),
            "tub": (g.get("final") or {}).get("tubular_mse_abs"),
            "S_gap": (g.get("final") or {}).get("sufficiency_gap"),
        }
        for g in grid
    ]

    pack = {
        "meta": {
            "purpose": "AAAI remaining-work pack: Table1 main + Table2 contrast + LLM negative",
            "success_rule": "F>=0.5 & OOD>=0.8 & eta<0.02 & tub<0.05 & S_gap>=0.15",
        },
        "table1_main": table1,
        "table2_contrast": table2,
        "llm_negative": llm_neg,
        "lambda_grid": grid_rows,
        "best_lambda": best["lam_flow"] if best else None,
        "aaai_ready_experimentally": bool(best and best.get("gates", {}).get("pass_aaai_main")),
    }
    (args.out_dir / "aaai_pack.json").write_text(json.dumps(pack, indent=2), encoding="utf-8")
    print(json.dumps(pack, indent=2), flush=True)
    if not pack["aaai_ready_experimentally"]:
        print("[WARN] no λ member fully passed AAAI main gates", flush=True)
        raise SystemExit(2)
    print("[OK] AAAI experimental pack ready", flush=True)


if __name__ == "__main__":
    main()
