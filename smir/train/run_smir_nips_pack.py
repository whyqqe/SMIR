#!/usr/bin/env python3
"""Pack multi-seed NIPS continuation-theory suite."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    seeds = {}
    for d in sorted(args.grid_root.glob("seed_*")):
        p = d / "summary.json"
        if p.exists():
            seeds[d.name] = json.load(open(p))

    n = len(seeds)
    n_core = sum(1 for s in seeds.values() if (s.get("claims") or {}).get("pass_nips_core"))
    n_strong = sum(1 for s in seeds.values() if (s.get("claims") or {}).get("pass_nips_strong"))
    n_A = sum(1 for s in seeds.values() if (s.get("claims") or {}).get("A/pass_phase_A"))
    n_B = sum(1 for s in seeds.values() if (s.get("claims") or {}).get("B/pass_phase_B"))
    n_C = sum(1 for s in seeds.values() if (s.get("claims") or {}).get("C/pass_phase_C"))
    n_D = sum(1 for s in seeds.values() if (s.get("claims") or {}).get("D/pass_phase_D"))

    pack = {
        "meta": {
            "purpose": "NeurIPS/ICML-oriented continuation theory validation (A+B+C+D)",
            "theory": "new_theory.md",
            "n_seeds": n,
        },
        "by_seed_claims": {k: v.get("claims") for k, v in seeds.items()},
        "by_seed_brief": {
            k: {
                "A1": (v.get("A") or {}).get("A1"),
                "A2": (v.get("A") or {}).get("A2"),
                "A3": (v.get("A") or {}).get("A3"),
                "B2": (v.get("B") or {}).get("B2"),
                "C": (v.get("C") or {}).get("claims"),
                "D": (v.get("D") or {}).get("claims"),
            }
            for k, v in seeds.items()
        },
        "claims": {
            "n_pass_nips_core": n_core,
            "n_pass_nips_strong": n_strong,
            "n_pass_A": n_A,
            "n_pass_B": n_B,
            "n_pass_C": n_C,
            "n_pass_D": n_D,
            "pass_nips_pack_core": n_core >= 1 and (n_core >= 2 if n >= 2 else True),
            "pass_nips_pack_strong": n_strong >= 1 and n_B >= 1 and n_C >= 1,
        },
    }
    out = args.out_dir / "nips_pack.json"
    out.write_text(json.dumps(pack, indent=2), encoding="utf-8")
    print(json.dumps(pack["claims"], indent=2))
    if not pack["claims"]["pass_nips_pack_core"]:
        raise SystemExit(2)
    print("[OK] NIPS pack core passed")


if __name__ == "__main__":
    main()
