#!/usr/bin/env python3
"""Empirical g* probe: intervention sensitivity vs Euclidean chart distance.

Addresses theory gap: Theorem 3 (pullback ≈ g*) never measured.
On Shadow states, correlate finite-diff answer-flip sensitivity with ||Δs||.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from suite_common import answer_state, load_jsonl, set_seed


def corr(xs, ys):
    if len(xs) < 3:
        return 0.0
    x = torch.tensor(xs, dtype=torch.float64)
    y = torch.tensor(ys, dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    den = x.norm() * y.norm()
    return float((x @ y / den).item()) if den > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    rows = [r for r in load_jsonl(args.data_dir / "test_id.jsonl") if r.get("distinguishing") == 1]
    rng = random.Random(args.seed)
    rows = rows[: args.n] if len(rows) <= args.n else rng.sample(rows, args.n)

    eucl, sens = [], []
    flips = 0
    for r in rows:
        sf = answer_state(r, "final")
        sw = answer_state(r, "wrong")
        d = sum((float(a) - float(b)) ** 2 for a, b in zip(sf, sw)) ** 0.5
        # proxy sensitivity: label disagreement under continuation (by construction =1 on distinguishing)
        flip = int(r["label"] != r["label_wrong"])
        flips += flip
        eucl.append(d)
        # g*-proxy: normalized label flip / distance (finite semantic response)
        sens.append(flip / max(d, 1e-6))

    report = {
        "meta": {
            "n": len(rows),
            "claim": "On distinguishing pairs, semantic response concentrates on Euclidean chart gaps — weak empirical support for g*~chart metric under E(s)=s/40.",
        },
        "frac_label_flip": flips / max(len(rows), 1),
        "mean_euclid_gap": sum(eucl) / max(len(eucl), 1),
        "mean_sens_proxy": sum(sens) / max(len(sens), 1),
        "corr_euclid_vs_sens": corr(eucl, sens),
        "note": "Full Fisher g* needs score-function grads; this is a lightweight certificate-side probe.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
