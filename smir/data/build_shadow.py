#!/usr/bin/env python3
"""Shadow-domain data for SMIR (structured chart + continuation for certification)."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "/project/peilab/why/NEW/ccro/env")
from synlogic_hard import (  # noqa: E402
    apply_action,
    continuation_str,
    find_distinguishing_cont,
    sample_action,
    sample_continuation,
    sample_state,
    state_str,
    action_str,
    y_star,
)


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")
    bad = 0
    with path.open() as f:
        for line in f:
            try:
                json.loads(line)
            except Exception:
                bad += 1
    if bad:
        raise RuntimeError(f"bad jsonl {path}: {bad}")
    print(f"[ok] {path} n={len(rows)}", flush=True)


def cont_to_dict(c) -> dict:
    return {
        "actions": [{"op": a.op, "i": a.i, "j": a.j, "k": a.k} for a in c.actions],
        "query": {"reg": c.query.reg, "thr": float(c.query.thr), "kind": c.query.kind},
    }


def make_row(rng: random.Random, split: str, idx: int, chain_len: int, tag: str) -> dict:
    s0 = sample_state(rng)
    ops = ("add", "sub", "mul")
    chain = [sample_action(rng, ops) for _ in range(chain_len)]
    if tag == "ood":
        chain[-1].op = "mul"
    states = [s0]
    s = s0
    for a in chain:
        s = apply_action(s, a)
        states.append(s)
    s_bad = list(s)
    s_bad[chain[-1].k] += rng.choice([-2.0, -1.0, 1.0, 2.0])
    c = find_distinguishing_cont(rng, s, s_bad, ops, [0, 1, 2], tries=50)
    if c is None:
        c = sample_continuation(rng, n_extra=rng.choice([0, 1, 2]), allowed_ops=ops)
    y = y_star(s, c)
    y_bad = y_star(s_bad, c)
    program = " ; ".join(action_str(a) for a in chain)
    t = rng.randrange(len(chain))
    s_t = states[t]
    a_t = chain[t]
    s_tp1 = states[t + 1]
    return {
        "id": f"{split}-{idx}",
        "split": split,
        "tag": tag,
        "prompt": (
            "You are a register machine. Apply operations, then answer YES or NO only.\n"
            f"Start: {state_str(s0)}\n"
            f"Operations: {program}\n"
            f"Query: {continuation_str(c)}\n"
            "Final answer:"
        ),
        "label": y,
        "label_wrong": y_bad,
        "distinguishing": int(y != y_bad),
        "chain_len": chain_len,
        "s0": list(map(float, s0)),
        "states": [list(map(float, st)) for st in states],
        "state_final": list(map(float, s)),
        "state_wrong": list(map(float, s_bad)),
        "state_final_text": state_str(s),
        "state_wrong_text": state_str(s_bad),
        "prompt_state_enc": f"Registers: {state_str(s)}\n",
        "prompt_state_wrong_enc": f"Registers: {state_str(s_bad)}\n",
        "prompt_state_t": f"Registers: {state_str(s_t)}\n",
        "prompt_state_tp1": f"Registers: {state_str(s_tp1)}\n",
        "state_t": list(map(float, s_t)),
        "state_tp1": list(map(float, s_tp1)),
        "step_op": a_t.op,
        "step_action": {"op": a_t.op, "i": a_t.i, "j": a_t.j, "k": a_t.k},
        "actions": [{"op": a.op, "i": a.i, "j": a.j, "k": a.k} for a in chain],
        "continuation": cont_to_dict(c),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-val", type=int, default=800)
    ap.add_argument("--n-test-id", type=int, default=600)
    ap.add_argument("--n-test-ood", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--train-lens",
        type=str,
        default="2,3,4",
        help="comma-separated chain lengths for train/val (use 2,3,4,5 to cover OOD lengths)",
    )
    ap.add_argument("--ood-lens", type=str, default="4,5")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_lens = [int(x) for x in args.train_lens.split(",") if x.strip()]
    ood_lens = [int(x) for x in args.ood_lens.split(",") if x.strip()]

    def build(name, n, seed, tag, lens):
        rng = random.Random(seed)
        rows = [make_row(rng, name, i, rng.choice(lens), tag) for i in range(n)]
        dump_jsonl(args.out_dir / f"{name}.jsonl", rows)
        print(f"  dist={sum(r['distinguishing'] for r in rows)/n:.3f} lens={lens}", flush=True)

    build("train", args.n_train, args.seed, "train", train_lens)
    build("val", args.n_val, args.seed + 1, "val", train_lens)
    build("test_id", args.n_test_id, args.seed + 2, "id", [2, 3, 4])
    build("test_ood", args.n_test_ood, args.seed + 3, "ood", ood_lens)
    (args.out_dir / "meta.json").write_text(
        json.dumps(
            {
                "name": "SMIR-Shadow-Struct",
                "theory": "SMIR.md",
                "train_lens": train_lens,
                "ood_lens": ood_lens,
                "fields": ["s0", "states", "actions", "continuation", "state_final", "state_wrong"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
