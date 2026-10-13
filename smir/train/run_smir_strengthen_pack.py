#!/usr/bin/env python3
"""Pack defect-fix experiments: register pub-core + graph domain2 + chart adapter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def brief_reg(root: Path):
    out = {}
    for seed_dir in sorted(root.glob("seed_*")):
        s = load(seed_dir / "summary.json")
        if s:
            out[seed_dir.name] = s.get("claims")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reg-root", type=Path, default=Path("/project/peilab/why/NEW/smir/artifacts/runs/smir_pub_core"))
    ap.add_argument("--graph-root", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--aaai-ref", type=Path, default=Path("/project/peilab/why/NEW/smir/results/aaai_pack.json"))
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    reg_claims = brief_reg(args.reg_root)
    reg_any_m1 = any((c or {}).get("pass_m1") for c in reg_claims.values()) or False
    reg_any_m2 = any((c or {}).get("pass_m2_discrete") or (c or {}).get("pass_m2_soft") for c in reg_claims.values()) or False
    # also peek seed_0 files directly
    s0 = load(args.reg_root / "seed_0" / "summary.json")
    if s0:
        reg_any_m1 = reg_any_m1 or bool(s0.get("claims", {}).get("pass_m1"))
        reg_any_m2 = reg_any_m2 or bool(
            s0.get("claims", {}).get("pass_m2_discrete") or s0.get("claims", {}).get("pass_m2_soft")
        )

    graph_summaries = {}
    g_m1 = g_m2 = g_charts = False
    for seed_dir in sorted(args.graph_root.glob("seed_*")):
        s = load(seed_dir / "summary.json")
        if not s:
            continue
        graph_summaries[seed_dir.name] = s.get("claims")
        c = s.get("claims") or {}
        g_m1 = g_m1 or bool(c.get("pass_m1"))
        g_m2 = g_m2 or bool(c.get("pass_m2_discrete"))
        g_charts = g_charts or bool(c.get("pass_charts"))

    adapter = load(args.adapter) if args.adapter else load(args.out_dir / "chart_adapter.json")
    aaai = load(args.aaai_ref)

    pack = {
        "meta": {
            "purpose": "Strengthen pack addressing review defects: domain2 + multiseed reg + weak-positive adapter",
            "defects_targeted": [
                "single_domain",
                "statistical_ sufficiency",
                "llm_only_negative",
            ],
        },
        "register_pub_claims_by_seed": reg_claims,
        "graph_claims_by_seed": graph_summaries,
        "chart_adapter": adapter,
        "aaai_main_ref": (aaai or {}).get("table1_main"),
        "llm_negative_ref": (aaai or {}).get("llm_negative"),
        "claims": {
            "pass_register_m1": reg_any_m1,
            "pass_register_m2": reg_any_m2,
            "pass_graph_m1": g_m1,
            "pass_graph_m2": g_m2,
            "pass_graph_charts": g_charts,
            "pass_weak_positive_adapter": bool((adapter or {}).get("claims", {}).get("pass_weak_positive")),
            "pass_cross_domain": reg_any_m1 and g_m1,
            "pass_strengthen_core": reg_any_m1 and reg_any_m2 and g_m1 and g_m2,
        },
    }
    (args.out_dir / "strengthen_pack.json").write_text(json.dumps(pack, indent=2), encoding="utf-8")
    print(json.dumps(pack, indent=2), flush=True)
    if not pack["claims"]["pass_strengthen_core"]:
        print("[WARN] strengthen core incomplete", flush=True)
        raise SystemExit(2)
    print("[OK] strengthen pack ready", flush=True)


if __name__ == "__main__":
    main()
