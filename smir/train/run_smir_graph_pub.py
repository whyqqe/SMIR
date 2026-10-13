#!/usr/bin/env python3
"""Graph-domain publication legs: second semantic domain (defect #1).

Kinds: ghard / gdisc (gated) / gsoft-joint negative / diag / charts.
Same metrics as register pub-core.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_suite import SuiteSMIR, GOP2ID, graph_action_tensors
from suite_common import chain_abs, encode_answer, evaluate_bundle, load_jsonl, set_seed


def split_by_len(rows, maxlen):
    return [r for r in rows if int(r.get("chain_len", 0)) <= maxlen]


def transfer_q(src, dst):
    dst.q_enc.load_state_dict(src.q_enc.state_dict())
    dst.answer.load_state_dict(src.answer.state_dict())


def gates_m1(m):
    I = m["intervention"]
    return {
        "F_oracle": I["branch_disagree"],
        "F_flow": m["flow_intervention"]["branch_disagree"],
        "ood": m["task_ood"],
        "eta": m["eta_abs"],
        "tub": m["tubular_mse_abs"],
        "S_gap": m["sufficiency_gap"],
        "pass_m1": (
            I["branch_disagree"] >= 0.45
            and m["task_ood"] >= 0.75
            and m["eta_abs"] < 0.05
            and m["tubular_mse_abs"] < 0.1
            and m["sufficiency_gap"] >= 0.10
        ),
    }


def gates_m2(m):
    F = m["flow_intervention"]["branch_disagree"]
    return {
        "F_oracle": m["intervention"]["branch_disagree"],
        "F_flow": F,
        "ood": m["task_ood"],
        "eta": m["eta_abs"],
        "tub": m["tubular_mse_abs"],
        "S_gap": m["sufficiency_gap"],
        "pass_m2": (
            F >= 0.40
            and m["task_ood"] >= 0.75
            and m["eta_abs"] < 0.1
            and m["tubular_mse_abs"] < 0.25
            and m["sufficiency_gap"] >= 0.08
        ),
    }


def train_q(model, rows_all, rows_short, steps, lr, device, bs, seed, p_dist, lam_flow, prefer_short, tag):
    rng = random.Random(seed)
    dist_all = [r for r in rows_all if r.get("distinguishing") == 1] or rows_all
    dist_short = [r for r in rows_short if r.get("distinguishing") == 1] or rows_short
    pool = rows_short if prefer_short else rows_all
    dist_pool = dist_short if prefer_short else dist_all
    for p in model.V.parameters():
        p.requires_grad = False
    params = list(model.q_enc.parameters()) + list(model.answer.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    model.train()
    for step in range(1, steps + 1):
        batch = [rng.choice(dist_pool if rng.random() < p_dist else pool) for _ in range(bs)]
        loss = torch.zeros((), device=device)
        ok = f_ok = f_n = 0
        for r in batch:
            q = r["continuation"]["query"]
            y = torch.tensor([r["label"]], device=device)
            z = encode_answer(model, r, "final", device)
            lo = model.logits(z, q)
            loss = loss + F.cross_entropy(lo, y)
            ok += int(lo.argmax(-1).item() == r["label"])
            if r.get("distinguishing") == 1:
                yw = torch.tensor([r["label_wrong"]], device=device)
                zw = encode_answer(model, r, "wrong", device)
                loss = loss + F.cross_entropy(model.logits(zw, q), yw)
                f_ok += int(lo.argmax(-1).item() != model.logits(zw, q).argmax(-1).item())
                f_n += 1
            Tc = max(len(r["actions"]), 1)
            z0 = model.encode_state(r["s0"], device)
            zT = model.flow(z0, r["actions"], T=Tc)
            cont = r["continuation"].get("actions") or []
            z_ans = model.flow(zT, cont, T=len(cont)) if cont else zT
            loss = loss + lam_flow * F.cross_entropy(model.logits(z_ans, q), y)
            loss = loss + 1e-3 * chain_abs(model, r, device)
        loss = loss / bs
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == 1:
            print(f"[{tag}] step={step} L={loss.item():.4f} acc={ok/bs:.3f} batchF={f_ok/max(f_n,1):.3f}", flush=True)


def train_v(model, rows_all, rows_short, steps, lr, device, bs, seed, p_dist, lam_flow, lam_conj, lam_route, train_q, tag):
    rng = random.Random(seed)
    dist_all = [r for r in rows_all if r.get("distinguishing") == 1] or rows_all
    pool, dist_pool = rows_all, dist_all
    for p in model.V.parameters():
        p.requires_grad = True
    for p in list(model.q_enc.parameters()) + list(model.answer.parameters()):
        p.requires_grad = bool(train_q)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    model.train()
    for step in range(1, steps + 1):
        batch = [rng.choice(dist_pool if rng.random() < p_dist else pool) for _ in range(bs)]
        loss = torch.zeros((), device=device)
        for r in batch:
            q = r["continuation"]["query"]
            y = torch.tensor([r["label"]], device=device)
            z = encode_answer(model, r, "final", device)
            loss = loss + F.cross_entropy(model.logits(z, q), y)
            if r.get("distinguishing") == 1:
                loss = loss + F.cross_entropy(
                    model.logits(encode_answer(model, r, "wrong", device), q),
                    torch.tensor([r["label_wrong"]], device=device),
                )
            Tc = max(len(r["actions"]), 1)
            z0 = model.encode_state(r["s0"], device)
            zT = model.flow(z0, r["actions"], T=Tc)
            cont = r["continuation"].get("actions") or []
            z_ans = model.flow(zT, cont, T=len(cont)) if cont else zT
            loss = loss + lam_flow * F.cross_entropy(model.logits(z_ans, q), y)
            if lam_conj > 0:
                loss = loss + lam_conj * chain_abs(model, r, device)
            if lam_route > 0 and hasattr(model.V, "route"):
                for a, st in zip(r["actions"], r["states"]):
                    tid = GOP2ID.get(a["op"], 2)
                    if tid > 1:
                        continue
                    zt = model.encode_state(st, device)
                    op, i, j, k = graph_action_tensors(a, device)
                    if hasattr(model.V, "route_logits"):
                        logits = model.V.route_logits(zt, op, i, j, k)
                    else:
                        op_oh = F.one_hot(op.clamp(0, 2), 3).float()
                        i_oh = F.one_hot(i.clamp(0, 4), 5).float()
                        j_oh = F.one_hot(j.clamp(0, 4), 5).float()
                        logits = model.V.route(torch.cat([zt, op_oh, i_oh, j_oh], dim=-1))
                    loss = loss + lam_route * F.cross_entropy(logits, torch.tensor([tid], device=device))
        loss = loss / bs
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 50 == 0 or step == 1:
            print(f"[{tag}] step={step} L={loss.item():.4f}", flush=True)


class ChartWrap(nn.Module):
    def __init__(self, base, mode, seed=0):
        super().__init__()
        self.base = base
        self.mode = mode
        self.kind = base.kind
        self.is_graph = bool(getattr(base, "is_graph", str(base.kind).startswith("g")))
        self.z_dim = base.z_dim
        self.T = base.T
        g = torch.Generator().manual_seed(seed + 7)
        if mode == "proj":
            P = torch.randn(5, 5, generator=g)
            P[:, -1] = 0
            self.register_buffer("P", P / (P.norm() + 1e-6) * (5 ** 0.5))
        self.scale = {"legal": 1.0, "scale2": 2.0}.get(mode, 1.0)

    def encode_state(self, state, device=None):
        z = self.base.encode_state(state, device)
        if self.mode in {"legal", "scale2"}:
            return z * self.scale
        if self.mode == "proj":
            return z @ self.P.to(z.device)
        if self.mode == "zero":
            return torch.zeros_like(z)
        return z

    def encode_query(self, query, device=None):
        return self.base.encode_query(query, device)

    def one_step(self, z, action):
        return self.base.one_step(z, action)

    def flow(self, z0, actions, T=None):
        return self.base.flow(z0, actions, T=T)

    def logits(self, z, query):
        return self.base.logits(z, query)

    @property
    def V(self):
        return self.base.V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps-1", type=int, default=2500)
    ap.add_argument("--steps-2", type=int, default=3500)
    ap.add_argument("--steps-v", type=int, default=3000)
    ap.add_argument("--eval-n", type=int, default=300)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    t0 = time.time()
    train = load_jsonl(args.data_dir / "train.jsonl")
    test_id = load_jsonl(args.data_dir / "test_id.jsonl")
    test_ood = load_jsonl(args.data_dir / "test_ood.jsonl")
    short = split_by_len(train, 4)
    print(f"[GraphPub] device={device} seed={args.seed} train={len(train)}", flush=True)

    # E1 ghard
    hard = SuiteSMIR(kind="ghard", T=args.T).to(device)
    train_q(hard, train, short, args.steps_1, 2e-3, device, 64, args.seed, 0.85, 0.2, True, "gE1a")
    train_q(hard, train, short, args.steps_2, 1e-3, device, 64, args.seed + 1, 0.75, 0.5, False, "gE1b")
    m1 = evaluate_bundle(hard, test_id, test_ood, args.T, args.eval_n, device)
    g1 = gates_m1(m1)
    (args.out_dir / "hard_e1").mkdir(parents=True, exist_ok=True)
    hard.save(args.out_dir / "hard_e1" / "final")
    (args.out_dir / "hard_e1" / "overall.json").write_text(
        json.dumps({"kind": "graph_hard_e1", "seed": args.seed, "final": m1, "gates": g1}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"graph_hard": g1}, indent=2), flush=True)

    # discrete gated
    disc = SuiteSMIR(kind="gdisc", T=args.T).to(device)
    transfer_q(hard, disc)
    train_v(disc, train, short, args.steps_v, 1e-3, device, 64, args.seed + 10, 0.85, 0.3, 0.5, 1.0, False, "gdiscV")
    train_v(disc, train, short, args.steps_v // 2, 5e-4, device, 64, args.seed + 11, 0.85, 0.5, 0.3, 0.5, True, "gdiscQ")
    md = evaluate_bundle(disc, test_id, test_ood, args.T, args.eval_n, device)
    gd = gates_m2(md)
    disc.save(args.out_dir / "discrete_gated" / "final")
    (args.out_dir / "discrete_gated").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "discrete_gated" / "overall.json").write_text(
        json.dumps({"kind": "graph_discrete_gated", "seed": args.seed, "final": md, "gates": gd}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"graph_discrete": gd}, indent=2), flush=True)

    # soft joint negative
    soft = SuiteSMIR(kind="gsoft", T=args.T, soft_bias=2.0).to(device)
    train_v(soft, train, short, args.steps_1 + args.steps_v // 2, 1e-3, device, 64, args.seed, 0.65, 0.5, 0.5, 0.2, True, "gsoftJ")
    if hasattr(soft.V, "set_op_bias"):
        soft.V.set_op_bias(0.0)
    ms = evaluate_bundle(soft, test_id, test_ood, args.T, args.eval_n, device)
    gs = gates_m2(ms)
    (args.out_dir / "soft_joint").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "soft_joint" / "overall.json").write_text(
        json.dumps({"kind": "graph_soft_joint", "seed": args.seed, "final": ms, "gates": gs}, indent=2), encoding="utf-8"
    )

    # charts
    rows = {}
    for mode in ["legal", "scale2", "proj", "zero"]:
        wrap = ChartWrap(hard, mode, seed=args.seed)
        mm = evaluate_bundle(wrap, test_id, test_ood, args.T, args.eval_n, device)
        rows[mode] = {
            "F_oracle": mm["intervention"]["branch_disagree"],
            "ood": mm["task_ood"],
            "tub": mm["tubular_mse_abs"],
        }
    claims_c = {
        "proj_hurts": rows["proj"]["F_oracle"] < rows["legal"]["F_oracle"] - 0.1,
        "zero_hurts": rows["zero"]["F_oracle"] < 0.25,
        "pass_thm6_style": rows["proj"]["F_oracle"] < rows["legal"]["F_oracle"] - 0.1,
    }
    (args.out_dir / "charts").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "charts" / "overall.json").write_text(
        json.dumps({"kind": "graph_charts", "seed": args.seed, "rows": rows, "claims": claims_c}, indent=2), encoding="utf-8"
    )

    # diag break Q
    class NullV(nn.Module):
        def forward(self, z, op, i, j, k):
            return z

    broken_v = SuiteSMIR(kind="ghard", T=args.T).to(device)
    broken_v.load_state_dict(hard.state_dict())
    broken_v.V = NullV()
    mv = evaluate_bundle(broken_v, test_id, test_ood, args.T, args.eval_n, device)
    broken_q = SuiteSMIR(kind="ghard", T=args.T).to(device)
    broken_q.load_state_dict(hard.state_dict())
    for p in broken_q.answer.parameters():
        p.data.normal_(0, 1)
    mq = evaluate_bundle(broken_q, test_id, test_ood, args.T, args.eval_n, device)
    me = evaluate_bundle(ChartWrap(hard, "proj", args.seed), test_id, test_ood, args.T, args.eval_n, device)
    diag = {
        "intact": m1["intervention"]["branch_disagree"],
        "break_E": me["intervention"]["branch_disagree"],
        "break_V_ood": mv["task_ood"],
        "break_Q": mq["intervention"]["branch_disagree"],
        "pass_thm5_necessity": (
            me["intervention"]["branch_disagree"] < m1["intervention"]["branch_disagree"] - 0.08
            and mq["intervention"]["branch_disagree"] < 0.3
        ),
    }
    (args.out_dir / "diag").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "diag" / "overall.json").write_text(json.dumps({"kind": "graph_diag", "seed": args.seed, "claims": diag}, indent=2), encoding="utf-8")

    summary = {
        "meta": {"domain": "graph", "seed": args.seed, "seconds": round(time.time() - t0, 2), "host": os.uname().nodename},
        "claims": {
            "pass_m1": bool(g1["pass_m1"]),
            "pass_m2_discrete": bool(gd["pass_m2"]),
            "gated_beats_joint": gd["F_flow"] > gs["F_flow"] + 0.05,
            "pass_charts": bool(claims_c["pass_thm6_style"]),
            "pass_diag": bool(diag["pass_thm5_necessity"]),
            "pass_domain2": bool(g1["pass_m1"]) and bool(gd["pass_m2"]),
        },
        "hard": g1,
        "discrete": gd,
        "soft_joint": gs,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["claims"]["pass_domain2"]:
        raise SystemExit(2)
    print("[OK] graph domain core passed", flush=True)


if __name__ == "__main__":
    main()
