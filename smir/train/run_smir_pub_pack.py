#!/usr/bin/env python3
"""Pack publication-core multiseed results into one venue table JSON."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.pstdev(xs)


def agg_exp(root: Path, name: str):
    rows = []
    for p in sorted(root.glob(f"seed_*/{name}/overall.json")):
        d = load(p)
        if not d:
            continue
        f = d.get("final") or {}
        I = f.get("intervention") or {}
        FI = f.get("flow_intervention") or {}
        g = d.get("gates") or {}
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
                "pass": bool(g.get("pass_m1") or g.get("pass_m2")),
            }
        )
    if not rows:
        # diag/charts style
        for p in sorted(root.glob(f"seed_*/{name}/overall.json")):
            d = load(p)
            if d and "claims" in d:
                return {"name": name, "n_seeds": 1, "claims": d.get("claims"), "rows": d.get("rows"), "special": True}
        return {"name": name, "missing": True}

    def A(key):
        m, s = mean_std([r[key] for r in rows])
        return {"mean": m, "std": s}

    return {
        "name": name,
        "n_seeds": len(rows),
        "F_oracle": A("F_oracle"),
        "F_flow": A("F_flow"),
        "ood": A("ood"),
        "eta": A("eta"),
        "tub": A("tub"),
        "S_gap": A("S_gap"),
        "pass_any": any(r["pass"] for r in rows),
        "pass_all": all(r["pass"] for r in rows),
        "seeds": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--aaai-ref", type=Path, default=Path("/project/peilab/why/NEW/smir/results/aaai_pack.json"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    table = {
        k: agg_exp(args.grid_root, k)
        for k in ["hard_e1", "discrete_gated", "soft_gated", "soft_joint", "diag", "charts"]
    }

    # gather seed-level claims
    pub_flags = []
    for p in sorted(args.grid_root.glob("seed_*/summary.json")):
        d = load(p)
        if d:
            pub_flags.append(d.get("claims") or {})

    pass_m1 = any(f.get("pass_m1") for f in pub_flags) or bool(table["hard_e1"].get("pass_any"))
    pass_m2 = any(f.get("pass_m2_discrete") or f.get("pass_m2_soft") for f in pub_flags) or bool(
        table["discrete_gated"].get("pass_any") or table["soft_gated"].get("pass_any")
    )
    beat = any(f.get("gated_beats_joint") for f in pub_flags)
    diag_ok = bool((table.get("diag") or {}).get("claims", {}).get("pass_thm5_necessity"))
    charts_ok = bool((table.get("charts") or {}).get("claims", {}).get("pass_thm6_style"))

    pack = {
        "meta": {
            "purpose": "Publication core pack (Thm5/6/7 + Prop8)",
            "m2_rule": "Flow-F>=0.45 & OOD>=0.80 & eta<0.05 & tub<0.20",
        },
        "table": table,
        "aaai_ref": load(args.aaai_ref),
        "claims": {
            "pass_m1": pass_m1,
            "pass_m2": pass_m2,
            "gated_beats_joint": beat,
            "pass_diag": diag_ok,
            "pass_charts": charts_ok,
            "pass_pub_core": pass_m1 and pass_m2,
            "pass_pub_strong": pass_m1 and pass_m2 and (beat or diag_ok or charts_ok),
        },
    }
    (args.out_dir / "pub_pack.json").write_text(json.dumps(pack, indent=2), encoding="utf-8")
    print(json.dumps(pack, indent=2), flush=True)
    if not pack["claims"]["pass_pub_core"]:
        print("[WARN] pub core not ready", flush=True)
        raise SystemExit(2)
    print("[OK] pub pack ready", flush=True)


if __name__ == "__main__":
    main()
