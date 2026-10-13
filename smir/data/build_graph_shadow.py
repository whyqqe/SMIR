#!/usr/bin/env python3
"""Second semantic domain: 5-node graph distance shadow (ProsQA-style).

State s ∈ ℝ^5 stores node distances from a source (large = unreachable).
Actions:
  set  — set source i to 0
  relax — s[j] = min(s[j], s[i] + 1)
  null — no-op
Query: whether s[reg] <= thr (kind=gt inverted via thr compare as leq via kind handling).

JSON schema matches register shadow so trainers can reuse the same loaders.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

N = 5
UNREACH = 20.0


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")
    print(f"[ok] {path} n={len(rows)}", flush=True)


def apply_action(s, a):
    s = list(s)
    op, i, j = a["op"], int(a["i"]), int(a["j"])
    if op == "set":
        s[i] = 0.0
    elif op == "relax":
        s[j] = min(s[j], s[i] + 1.0)
    return s


def y_star(s, query):
    # label 1 iff distance[reg] <= thr
    return int(float(s[int(query["reg"])]) <= float(query["thr"]))


def sample_graph_actions(rng, chain_len, ood=False):
    src = rng.randrange(N)
    actions = [{"op": "set", "i": src, "j": 0, "k": 0}]
    # random undirected edges pool
    edges = [(u, v) for u in range(N) for v in range(N) if u != v]
    rng.shuffle(edges)
    # ensure connectivity-ish by cycling edges
    for t in range(chain_len - 1):
        u, v = edges[t % len(edges)]
        if ood and t == chain_len - 2:
            # longer effective hop: reverse direction unusual pattern
            u, v = v, u
        actions.append({"op": "relax", "i": u, "j": v, "k": 0})
    return actions, src


def make_row(rng, split, idx, chain_len, tag):
    actions, src = sample_graph_actions(rng, chain_len, ood=(tag == "ood"))
    s0 = [UNREACH] * N
    states = [list(s0)]
    s = list(s0)
    for a in actions:
        s = apply_action(s, a)
        states.append(list(s))
    # wrong: bump one node's distance
    s_bad = list(s)
    k = rng.randrange(N)
    s_bad[k] = min(UNREACH, s_bad[k] + rng.choice([1.0, 2.0, 3.0]))
    # find distinguishing query
    distinguishing = 0
    query = {"reg": 0, "thr": 1.0, "kind": "gt"}
    for _ in range(40):
        reg = rng.randrange(N)
        thr = float(rng.choice([0, 1, 2, 3, 4, 5]))
        q = {"reg": reg, "thr": thr, "kind": "gt"}
        # use leq semantics in y_star; kind kept for encoder compatibility
        y = y_star(s, q)
        yb = y_star(s_bad, q)
        query = q
        if y != yb:
            distinguishing = 1
            break
    y = y_star(s, query)
    yb = y_star(s_bad, query)
    t = rng.randrange(len(actions))
    return {
        "id": f"{split}-{idx}",
        "split": split,
        "tag": tag,
        "domain": "graph",
        "prompt": (
            f"Graph distances on {N} nodes. Source set then relax edges.\n"
            f"Start: {s0}\nActions: {actions}\n"
            f"Query: dist[{query['reg']}] <= {query['thr']}?\nFinal:"
        ),
        "label": y,
        "label_wrong": yb,
        "distinguishing": int(y != yb) if distinguishing else int(y != yb),
        "chain_len": chain_len,
        "s0": list(map(float, s0)),
        "states": [list(map(float, st)) for st in states],
        "state_final": list(map(float, s)),
        "state_wrong": list(map(float, s_bad)),
        "state_final_text": str(s),
        "state_wrong_text": str(s_bad),
        "prompt_state_enc": f"Dist: {s}\n",
        "prompt_state_wrong_enc": f"Dist: {s_bad}\n",
        "state_t": list(map(float, states[t])),
        "state_tp1": list(map(float, states[t + 1])),
        "step_op": actions[t]["op"],
        "step_action": actions[t],
        "actions": actions,
        "continuation": {"actions": [], "query": query},
        "source": src,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-val", type=int, default=800)
    ap.add_argument("--n-test-id", type=int, default=600)
    ap.add_argument("--n-test-ood", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--train-lens", type=str, default="3,4,5,6")
    ap.add_argument("--ood-lens", type=str, default="6,7,8")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_lens = [int(x) for x in args.train_lens.split(",") if x.strip()]
    ood_lens = [int(x) for x in args.ood_lens.split(",") if x.strip()]

    def build(name, n, seed, tag, lens):
        rng = random.Random(seed)
        rows = [make_row(rng, name, i, rng.choice(lens), tag) for i in range(n)]
        dump_jsonl(args.out_dir / f"{name}.jsonl", rows)
        dist = sum(r["distinguishing"] for r in rows) / max(n, 1)
        print(f"  dist={dist:.3f} lens={lens}", flush=True)

    build("train", args.n_train, args.seed, "train", train_lens)
    build("val", args.n_val, args.seed + 1, "val", train_lens)
    build("test_id", args.n_test_id, args.seed + 2, "id", [3, 4, 5])
    build("test_ood", args.n_test_ood, args.seed + 3, "ood", ood_lens)
    (args.out_dir / "meta.json").write_text(
        json.dumps(
            {
                "name": "SMIR-Graph-Shadow",
                "domain": "graph_distance",
                "train_lens": train_lens,
                "ood_lens": ood_lens,
                "n_nodes": N,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
