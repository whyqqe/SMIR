#!/usr/bin/env python3
"""Pack next-tier multiseed results + g* probe + prior AAAI/LLM negatives."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.pstdev(vals)


def summarize_seed_runs(root: Path, exp: str):
    rows = []
    for p in sorted(root.glob(f"seed_*/{exp}/overall.json")):
        d = load(p)
        if not d:
            continue
        f = d.get("final") or {}
        I = f.get("intervention") or {}
        FI = f.get("flow_intervention") or {}
        rows.append(
            {
                "seed": d.get("seed"),
                "F_oracle": I.get("branch_disagree"),
                "F_flow": FI.get("branch_disagree"),
                "ood": f.get("task_ood"),
                "eta": f.get("eta_abs"),
                "tub": f.get("tubular_mse_abs"),
                "S_gap": f.get("sufficiency_gap"),
                "routing_acc": f.get("routing_acc"),
                "pass": bool(
                    (d.get("gates") or {}).get("pass_m2")
                    or (d.get("gates") or {}).get("pass_m1")
                ),
                "score": d.get("score"),
            }
        )
    if not rows:
        return {"name": exp, "missing": True}
    def agg(key):
        m, s = mean_std([r[key] for r in rows])
        return {"mean": m, "std": s}

    return {
        "name": exp,
        "n_seeds": len(rows),
        "F_oracle": agg("F_oracle"),
        "F_flow": agg("F_flow"),
        "ood": agg("ood"),
        "eta": agg("eta"),
        "tub": agg("tub"),
        "S_gap": agg("S_gap"),
        "routing_acc": agg("routing_acc"),
        "pass_any": any(r["pass"] for r in rows),
        "pass_all": all(r["pass"] for r in rows),
        "seeds": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-root", type=Path, required=True)
    ap.add_argument("--runs-root", type=Path, default=Path("/project/peilab/why/NEW/smir/artifacts/runs"))
    ap.add_argument("--gstar", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    table = {
        "hard": summarize_seed_runs(args.grid_root, "hard"),
        "soft": summarize_seed_runs(args.grid_root, "soft"),
        "discrete": summarize_seed_runs(args.grid_root, "discrete"),
        "mlp_tf": summarize_seed_runs(args.grid_root, "mlp_tf"),
        "mlp_task": summarize_seed_runs(args.grid_root, "mlp_task"),
    }

    aaai = load(args.runs_root / "smir_aaai_pack" / "aaai_pack.json") or load(
        Path("/project/peilab/why/NEW/smir/results/aaai_pack.json")
    )
    gstar = load(args.gstar) if args.gstar else load(args.grid_root / "gstar_probe.json")

    soft = table["soft"]
    hard = table["hard"]
    disc = table["discrete"]
    soft_ok = bool(soft.get("pass_any")) and not soft.get("missing")
    hard_ok = bool(hard.get("pass_any")) and not hard.get("missing")
    disc_ok = bool(disc.get("pass_any")) and not disc.get("missing")

    pack = {
        "meta": {
            "purpose": "Next-tier pack: M1 Hard multiseed + M2 Soft anneal / Discrete + ablations + g* probe",
            "m2_rule": "Flow-F>=0.45 & OOD>=0.80 & eta<0.05 & tub<0.20 & S_gap>=0.12",
        },
        "table_main": table,
        "gstar_probe": gstar,
        "aaai_main_ref": (aaai or {}).get("table1_main"),
        "llm_negative_ref": (aaai or {}).get("llm_negative"),
        "claims": {
            "pass_m1_hard": hard_ok,
            "pass_m2_soft": soft_ok,
            "pass_m2_discrete": disc_ok,
            "pass_next_tier_core": hard_ok and (soft_ok or disc_ok),
            "ablation_task_only_worse_eta": (
                not table["mlp_task"].get("missing")
                and not table["mlp_tf"].get("missing")
                and (table["mlp_task"].get("eta", {}).get("mean") or 0)
                > (table["mlp_tf"].get("eta", {}).get("mean") or 0)
            ),
        },
    }
    (args.out_dir / "next_tier_pack.json").write_text(json.dumps(pack, indent=2), encoding="utf-8")
    print(json.dumps(pack, indent=2), flush=True)
    if not pack["claims"]["pass_next_tier_core"]:
        print("[WARN] next-tier core not ready", flush=True)
        raise SystemExit(2)
    print("[OK] next-tier pack ready", flush=True)


if __name__ == "__main__":
    main()
