#!/usr/bin/env python3
"""Pack multi-seed continuation-theory feasibility runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summaries(root: Path):
    out = {}
    if not root.exists():
        return out
    for d in sorted(root.glob("seed_*")):
        p = d / "summary.json"
        if p.exists():
            out[d.name] = json.load(open(p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    seeds = load_summaries(args.grid_root)
    n = len(seeds)
    any_feas = any((s.get("claims") or {}).get("pass_theory_feasibility") for s in seeds.values())
    n_hard = sum(1 for s in seeds.values() if (s.get("claims") or {}).get("pass_reg_hard_cont"))
    n_disc = sum(1 for s in seeds.values() if (s.get("claims") or {}).get("pass_reg_discrete_cont"))
    n_graph = sum(1 for s in seeds.values() if (s.get("claims") or {}).get("pass_graph_hard_cont"))
    n_mlp = sum(1 for s in seeds.values() if (s.get("claims") or {}).get("pass_reg_mlp_cont"))
    n_struct = sum(1 for s in seeds.values() if (s.get("claims") or {}).get("structured_beats_mlp"))

    pack = {
        "meta": {
            "purpose": "Preliminary feasibility of new_theory.md continuation consistency",
            "n_seeds": n,
            "objective": "L_continuation_only",
        },
        "by_seed": {k: v.get("claims") for k, v in seeds.items()},
        "by_seed_results": {k: v.get("results") for k, v in seeds.items()},
        "claims": {
            "any_seed_feasibility": any_feas,
            "n_pass_reg_hard": n_hard,
            "n_pass_reg_discrete": n_disc,
            "n_pass_reg_mlp": n_mlp,
            "n_pass_graph_hard": n_graph,
            "n_structured_beats_mlp": n_struct,
            "pass_continuation_feasibility": any_feas and n_hard >= 1 and (n_disc >= 1 or n_graph >= 1),
            "pass_multiseed_hard": n_hard >= 2 if n >= 2 else n_hard >= 1,
        },
    }
    out = args.out_dir / "continuation_pack.json"
    out.write_text(json.dumps(pack, indent=2), encoding="utf-8")
    print(json.dumps(pack["claims"], indent=2))
    if not pack["claims"]["pass_continuation_feasibility"]:
        raise SystemExit(2)
    print("[OK] continuation pack passed")


if __name__ == "__main__":
    main()
