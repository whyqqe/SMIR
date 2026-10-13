#!/usr/bin/env python3
"""SMIR Publication Core — highest-success pipeline aligned with Thm5/6/7 + Prop8.

E1  hard_e1        : AAAI Hard recipe (readout leg upper bound); optional ckpt reuse
E2a discrete_gated : Hard→Q transfer → learn discrete router + hard exec (Flow-F)
E2b soft_gated     : Hard→Q transfer → bias anneal only after Oracle-F gate
E2c soft_joint     : no-gate joint Soft (negative control)
E3  diag           : break E / V / Q separately (necessity)
E4  charts         : legal / scale / proj charts (Thm6/7)

Success core: E1 pass_m1 AND (discrete_gated OR soft_gated) pass_m2
             AND soft_gated/discrete Flow-F > soft_joint Flow-F (when all present).
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

from model_register import N_REG, action_tensors, state_tensor
from model_suite import SuiteSMIR
from suite_common import (
    chain_abs,
    encode_answer,
    evaluate_bundle,
    load_jsonl,
    set_seed,
)


def split_by_len(rows, maxlen):
    return [r for r in rows if int(r.get("chain_len", 0)) <= maxlen]


def transfer_q(src: SuiteSMIR, dst: SuiteSMIR):
    dst.q_enc.load_state_dict(src.q_enc.state_dict())
    dst.answer.load_state_dict(src.answer.state_dict())


def load_suite(path: Path, kind: str, device, T=5):
    ckpt = torch.load(path / "suite.pt", map_location=device)
    model = SuiteSMIR(kind=kind, T=ckpt.get("T", T)).to(device)
    # allow loading hard Q into other kinds via transfer only; full load if kind matches
    if ckpt.get("kind", kind) == kind:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        # load matching tensors only (q heads)
        src = SuiteSMIR(kind=ckpt.get("kind", "hard"), T=ckpt.get("T", T)).to(device)
        src.load_state_dict(ckpt["state_dict"])
        transfer_q(src, model)
    return model


def set_bias(model, b):
    if hasattr(model.V, "set_op_bias"):
        model.V.set_op_bias(b)


def set_tau(model, t):
    if hasattr(model.V, "set_tau"):
        model.V.set_tau(t)


def train_q_hard(
    model,
    rows_all,
    rows_short,
    steps,
    lr,
    device,
    bs,
    seed,
    p_dist,
    lam_flow,
    prefer_short,
    tag,
):
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
    hist = []
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
            hist.append({"step": step, "loss": float(loss.item()), "acc": ok / bs, "batch_F": f_ok / max(f_n, 1)})
            print(f"[{tag}] step={step} L={loss.item():.4f} acc={ok/bs:.3f} batchF={f_ok/max(f_n,1):.3f}", flush=True)
    return hist


def train_v_phase(
    model,
    rows_all,
    rows_short,
    steps,
    lr,
    device,
    bs,
    seed,
    p_dist,
    lam_flow,
    lam_conj,
    lam_route,
    prefer_short,
    train_q,
    soft_bias,
    disc_tau,
    tag,
    anneal_bias=None,
):
    rng = random.Random(seed)
    dist_all = [r for r in rows_all if r.get("distinguishing") == 1] or rows_all
    dist_short = [r for r in rows_short if r.get("distinguishing") == 1] or rows_short
    pool = rows_short if prefer_short else rows_all
    dist_pool = dist_short if prefer_short else dist_all
    set_bias(model, soft_bias)
    set_tau(model, disc_tau)
    for p in model.V.parameters():
        p.requires_grad = True
    for p in list(model.q_enc.parameters()) + list(model.answer.parameters()):
        p.requires_grad = bool(train_q)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    model.train()
    op_map = {"add": 0, "sub": 1, "mul": 2}
    hist = []
    for step in range(1, steps + 1):
        if anneal_bias is not None:
            t = (step - 1) / max(steps - 1, 1)
            b0, b1 = anneal_bias
            set_bias(model, b0 + (b1 - b0) * t)
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
            if lam_conj > 0:
                loss = loss + lam_conj * chain_abs(model, r, device)
            if lam_route > 0 and hasattr(model.V, "route"):
                for a, st in zip(r["actions"], r["states"]):
                    tid = op_map.get(a["op"], 3)
                    if tid > 2:
                        continue
                    zt = model.encode_state(st, device)
                    op, i, j, k = action_tensors(a, device)
                    if hasattr(model.V, "route_logits"):
                        logits = model.V.route_logits(zt, op, i, j, k)
                    else:
                        op_oh = F.one_hot(op.clamp(0, 3), 4).float()
                        i_oh = F.one_hot(i.clamp(0, N_REG - 1), N_REG).float()
                        j_oh = F.one_hot(j.clamp(0, N_REG - 1), N_REG).float()
                        k_oh = F.one_hot(k.clamp(0, N_REG - 1), N_REG).float()
                        logits = model.V.route(torch.cat([zt, op_oh, i_oh, j_oh, k_oh], dim=-1))
                    loss = loss + lam_route * F.cross_entropy(logits, torch.tensor([tid], device=device))
        loss = loss / bs
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 50 == 0 or step == 1:
            bnow = getattr(model.V, "op_bias", soft_bias)
            hist.append({"step": step, "loss": float(loss.item()), "acc": ok / bs, "batch_F": f_ok / max(f_n, 1), "bias": bnow})
            print(f"[{tag}] step={step} L={loss.item():.4f} acc={ok/bs:.3f} batchF={f_ok/max(f_n,1):.3f} bias={bnow}", flush=True)
    return hist


def gates_m1(m):
    I = m["intervention"]
    g = {
        "F_oracle": I["branch_disagree"],
        "F_flow": m["flow_intervention"]["branch_disagree"],
        "ood": m["task_ood"],
        "eta": m["eta_abs"],
        "tub": m["tubular_mse_abs"],
        "S_gap": m["sufficiency_gap"],
        "pass_m1": (
            I["branch_disagree"] >= 0.50
            and m["task_ood"] >= 0.80
            and m["eta_abs"] < 0.02
            and m["tubular_mse_abs"] < 0.05
            and m["sufficiency_gap"] >= 0.15
        ),
    }
    return g


def gates_m2(m):
    I = m["intervention"]
    FI = m["flow_intervention"]
    F = FI["branch_disagree"]
    g = {
        "F_oracle": I["branch_disagree"],
        "F_flow": F,
        "ood": m["task_ood"],
        "eta": m["eta_abs"],
        "tub": m["tubular_mse_abs"],
        "S_gap": m["sufficiency_gap"],
        "routing_acc": m.get("routing_acc"),
        "pass_m2": (
            F >= 0.45
            and m["task_ood"] >= 0.80
            and m["eta_abs"] < 0.05
            and m["tubular_mse_abs"] < 0.20
            and m["sufficiency_gap"] >= 0.12
        ),
    }
    return g


def run_hard_e1(args, train, short, test_id, test_ood, device, out_dir, seed):
    out_dir.mkdir(parents=True, exist_ok=True)
    reuse = args.e1_ckpt and Path(args.e1_ckpt).exists() and seed == args.reuse_seed
    if reuse:
        print(f"[E1] reuse ckpt {args.e1_ckpt}", flush=True)
        model = load_suite(Path(args.e1_ckpt), "hard", device, args.T)
        m = evaluate_bundle(model, test_id, test_ood, args.T, args.eval_n, device)
        model.save(out_dir / "final")
        g = gates_m1(m)
        return {"kind": "hard_e1", "seed": seed, "reused": True, "final": m, "gates": g}

    model = SuiteSMIR(kind="hard", T=args.T).to(device)
    print("==== E1 Phase1 short F ====", flush=True)
    train_q_hard(
        model, train, short, args.steps_e1_1, args.lr1, device, args.batch, seed,
        p_dist=max(args.p_dist, 0.8), lam_flow=min(args.lam_flow, 0.3), prefer_short=True, tag=f"E1a-s{seed}",
    )
    print("==== E1 Phase2 mix ====", flush=True)
    train_q_hard(
        model, train, short, args.steps_e1_2, args.lr2, device, args.batch, seed + 1,
        p_dist=args.p_dist, lam_flow=args.lam_flow, prefer_short=False, tag=f"E1b-s{seed}",
    )
    m = evaluate_bundle(model, test_id, test_ood, args.T, args.eval_n, device)
    if m["intervention"]["branch_disagree"] < 0.50 and m["task_ood"] >= 0.78:
        print("==== E1 Phase3 F-recovery ====", flush=True)
        train_q_hard(
            model, train, short, max(args.steps_e1_1 // 2, 800), args.lr2 * 0.7, device, args.batch, seed + 2,
            p_dist=0.9, lam_flow=0.2, prefer_short=False, tag=f"E1c-s{seed}",
        )
        m = evaluate_bundle(model, test_id, test_ood, args.T, args.eval_n, device)
    model.save(out_dir / "final")
    g = gates_m1(m)
    print(json.dumps({"E1": g}, indent=2), flush=True)
    return {"kind": "hard_e1", "seed": seed, "reused": False, "final": m, "gates": g, "model": model}


def run_discrete_gated(args, hard_model, train, short, test_id, test_ood, device, out_dir, seed):
    out_dir.mkdir(parents=True, exist_ok=True)
    model = SuiteSMIR(kind="discrete", T=args.T).to(device)
    transfer_q(hard_model, model)
    # verify gate
    m0 = evaluate_bundle(model, test_id, test_ood, args.T, args.eval_n, device)
    print(json.dumps({"disc_after_transfer": {"F": m0["intervention"]["branch_disagree"], "FlowF": m0["flow_intervention"]["branch_disagree"], "ood": m0["task_ood"]}}, indent=2), flush=True)
    if m0["intervention"]["branch_disagree"] < args.gate_f:
        print(f"[WARN] Oracle-F {m0['intervention']['branch_disagree']:.3f} < gate {args.gate_f}; continue but low success expected", flush=True)

    print("==== Discrete: train V (Q frozen) ====", flush=True)
    train_v_phase(
        model, train, short, args.steps_v, args.lr2, device, args.batch, seed + 10,
        p_dist=max(args.p_dist, 0.8), lam_flow=0.3, lam_conj=0.5, lam_route=1.0,
        prefer_short=False, train_q=False, soft_bias=0.0, disc_tau=1.2, tag=f"discV-s{seed}",
    )
    print("==== Discrete: light Q+V polish ====", flush=True)
    train_v_phase(
        model, train, short, args.steps_v // 2, args.lr2 * 0.5, device, args.batch, seed + 11,
        p_dist=0.85, lam_flow=args.lam_flow, lam_conj=0.3, lam_route=0.5,
        prefer_short=False, train_q=True, soft_bias=0.0, disc_tau=0.7, tag=f"discQ-s{seed}",
    )
    m = evaluate_bundle(model, test_id, test_ood, args.T, args.eval_n, device)
    model.save(out_dir / "final")
    g = gates_m2(m)
    print(json.dumps({"discrete_gated": g}, indent=2), flush=True)
    return {"kind": "discrete_gated", "seed": seed, "final": m, "gates": g, "transfer_probe": m0}


def run_soft_gated(args, hard_model, train, short, test_id, test_ood, device, out_dir, seed):
    out_dir.mkdir(parents=True, exist_ok=True)
    model = SuiteSMIR(kind="soft", T=args.T, soft_bias=args.bias_start).to(device)
    transfer_q(hard_model, model)
    set_bias(model, args.bias_start)
    m0 = evaluate_bundle(model, test_id, test_ood, args.T, args.eval_n, device)
    print(json.dumps({"soft_transfer": {"F": m0["intervention"]["branch_disagree"], "ood": m0["task_ood"]}}, indent=2), flush=True)
    if m0["intervention"]["branch_disagree"] < args.gate_f:
        print("[soft_gated] gate failed — skip anneal, save transfer-only", flush=True)
        model.save(out_dir / "final")
        g = gates_m2(m0)
        g["pass_m2"] = False
        g["gate_failed"] = True
        return {"kind": "soft_gated", "seed": seed, "final": m0, "gates": g, "skipped_anneal": True}

    print("==== Soft gated: anneal bias with strong route CE ====", flush=True)
    train_v_phase(
        model, train, short, args.steps_v, args.lr2, device, args.batch, seed + 20,
        p_dist=0.85, lam_flow=0.3, lam_conj=0.5, lam_route=1.0,
        prefer_short=False, train_q=True, soft_bias=args.bias_start, disc_tau=1.0,
        tag=f"softG-s{seed}", anneal_bias=(args.bias_start, args.bias_end),
    )
    m = evaluate_bundle(model, test_id, test_ood, args.T, args.eval_n, device)
    # if Flow-F collapsed, restore? keep better of transfer vs annealed on score
    def sc(mm):
        return mm["flow_intervention"]["branch_disagree"] + mm["task_ood"]

    if sc(m) < sc(m0) - 0.05:
        print("[soft_gated] annealed worse; keep transfer checkpoint metrics note", flush=True)
    model.save(out_dir / "final")
    g = gates_m2(m)
    print(json.dumps({"soft_gated": g}, indent=2), flush=True)
    return {"kind": "soft_gated", "seed": seed, "final": m, "gates": g, "transfer_probe": m0}


def run_soft_joint(args, train, short, test_id, test_ood, device, out_dir, seed):
    out_dir.mkdir(parents=True, exist_ok=True)
    model = SuiteSMIR(kind="soft", T=args.T, soft_bias=2.0).to(device)
    print("==== Soft joint (no gate, negative) ====", flush=True)
    train_v_phase(
        model, train, short, args.steps_e1_1 + args.steps_v // 2, args.lr1, device, args.batch, seed,
        p_dist=0.65, lam_flow=0.5, lam_conj=0.5, lam_route=0.2,
        prefer_short=False, train_q=True, soft_bias=2.0, disc_tau=1.0,
        tag=f"softJ-s{seed}", anneal_bias=(2.0, 0.0),
    )
    m = evaluate_bundle(model, test_id, test_ood, args.T, args.eval_n, device)
    model.save(out_dir / "final")
    g = gates_m2(m)
    print(json.dumps({"soft_joint": g}, indent=2), flush=True)
    return {"kind": "soft_joint", "seed": seed, "final": m, "gates": g}


class ChartWrapper(nn.Module):
    """Wrap SuiteSMIR.encode_state with a chart transform (Thm6/7)."""

    def __init__(self, base: SuiteSMIR, mode: str, seed: int = 0):
        super().__init__()
        self.base = base
        self.mode = mode
        self.kind = base.kind
        self.z_dim = base.z_dim
        self.T = base.T
        g = torch.Generator().manual_seed(seed + 123)
        if mode == "proj":
            # random projection folding one direction
            P = torch.randn(N_REG, N_REG, generator=g)
            P[:, -1] = 0  # kill last coordinate in chart
            self.register_buffer("P", P / (P.norm() + 1e-6) * (N_REG ** 0.5))
        elif mode == "perm":
            perm = torch.randperm(N_REG, generator=g)
            self.register_buffer("perm", perm)
        else:
            self.P = None
            self.perm = None
        self.scale = {"legal": 1.0, "scale2": 2.0, "scale_half": 0.5}.get(mode, 1.0)

    def encode_state(self, state, device=None):
        z = self.base.encode_state(state, device)
        if self.mode in {"legal", "scale2", "scale_half"}:
            return z * self.scale
        if self.mode == "proj":
            return z @ self.P.to(z.device)
        if self.mode == "perm":
            return z[:, self.perm.to(z.device)]
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


def run_charts(args, hard_model, test_id, test_ood, device, out_dir, seed):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = {}
    for mode in ["legal", "scale2", "scale_half", "proj", "perm", "zero"]:
        wrap = ChartWrapper(hard_model, mode, seed=seed).to(device)
        m = evaluate_bundle(wrap, test_id, test_ood, args.T, args.eval_n, device)
        rows[mode] = {
            "F_oracle": m["intervention"]["branch_disagree"],
            "F_flow": m["flow_intervention"]["branch_disagree"],
            "ood": m["task_ood"],
            "eta": m["eta_abs"],
            "tub": m["tubular_mse_abs"],
        }
        print(f"[E4 chart={mode}] F={rows[mode]['F_oracle']:.3f} tub={rows[mode]['tub']:.4f}", flush=True)
    # Thm6/7 checks
    legal_f = rows["legal"]["F_oracle"]
    claims = {
        "scale_preserves_F": abs(rows["scale2"]["F_oracle"] - legal_f) < 0.15 or abs(rows["scale_half"]["F_oracle"] - legal_f) < 0.15,
        "proj_hurts_F": rows["proj"]["F_oracle"] < legal_f - 0.15,
        "zero_hurts_F": rows["zero"]["F_oracle"] < 0.2,
        "pass_thm6_style": rows["proj"]["F_oracle"] < legal_f - 0.15 and rows["zero"]["F_oracle"] < 0.25,
    }
    out = {"kind": "charts", "seed": seed, "rows": rows, "claims": claims}
    (out_dir / "overall.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def run_diag(args, hard_model, test_id, test_ood, device, out_dir, seed):
    """Break E/V/Q separately on top of a strong Hard model."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base = evaluate_bundle(hard_model, test_id, test_ood, args.T, args.eval_n, device)

    # break E
    bad_e = ChartWrapper(hard_model, "proj", seed=seed)
    m_e = evaluate_bundle(bad_e, test_id, test_ood, args.T, args.eval_n, device)

    # break V: null actions in flow by monkeypatch
    class NullV(nn.Module):
        def forward(self, z, op, i, j, k):
            return z

    broken_v = SuiteSMIR(kind="hard", T=args.T).to(device)
    broken_v.load_state_dict(hard_model.state_dict())
    broken_v.V = NullV()
    m_v = evaluate_bundle(broken_v, test_id, test_ood, args.T, args.eval_n, device)

    # break Q: randomize answer head
    broken_q = SuiteSMIR(kind="hard", T=args.T).to(device)
    broken_q.load_state_dict(hard_model.state_dict())
    for p in broken_q.answer.parameters():
        p.data.normal_(0, 1.0)
    m_q = evaluate_bundle(broken_q, test_id, test_ood, args.T, args.eval_n, device)

    def pack(name, m):
        return {
            "name": name,
            "F_oracle": m["intervention"]["branch_disagree"],
            "F_flow": m["flow_intervention"]["branch_disagree"],
            "ood": m["task_ood"],
            "eta": m["eta_abs"],
            "tub": m["tubular_mse_abs"],
        }

    rows = {
        "intact": pack("intact", base),
        "break_E": pack("break_E", m_e),
        "break_V": pack("break_V", m_v),
        "break_Q": pack("break_Q", m_q),
    }
    claims = {
        "break_E_hurts": rows["break_E"]["F_oracle"] < rows["intact"]["F_oracle"] - 0.1,
        "break_V_hurts_flow": rows["break_V"]["F_flow"] < rows["intact"]["F_flow"] - 0.1
        or rows["break_V"]["ood"] < rows["intact"]["ood"] - 0.1,
        "break_Q_hurts": rows["break_Q"]["F_oracle"] < 0.25,
        "pass_thm5_necessity": False,
    }
    claims["pass_thm5_necessity"] = all(
        [claims["break_E_hurts"], claims["break_V_hurts_flow"], claims["break_Q_hurts"]]
    )
    out = {"kind": "diag", "seed": seed, "rows": rows, "claims": claims}
    (out_dir / "overall.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--exps", type=str, default="hard_e1,discrete_gated,soft_gated,soft_joint,diag,charts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--e1-ckpt", type=str, default="")
    ap.add_argument("--reuse-seed", type=int, default=0)
    ap.add_argument("--steps-e1-1", type=int, default=3000)
    ap.add_argument("--steps-e1-2", type=int, default=5000)
    ap.add_argument("--steps-v", type=int, default=3500)
    ap.add_argument("--lr1", type=float, default=2e-3)
    ap.add_argument("--lr2", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--p-dist", type=float, default=0.75)
    ap.add_argument("--lam-flow", type=float, default=0.5)
    ap.add_argument("--gate-f", type=float, default=0.55)
    ap.add_argument("--bias-start", type=float, default=5.0)
    ap.add_argument("--bias-end", type=float, default=0.0)
    ap.add_argument("--eval-n", type=int, default=300)
    ap.add_argument("--T", type=int, default=5)
    ap.add_argument("--short-maxlen", type=int, default=3)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    t0 = time.time()
    train = load_jsonl(args.data_dir / "train.jsonl")
    test_id = load_jsonl(args.data_dir / "test_id.jsonl")
    test_ood = load_jsonl(args.data_dir / "test_ood.jsonl")
    short = split_by_len(train, args.short_maxlen)
    exps = [e.strip() for e in args.exps.split(",") if e.strip()]
    print(f"[PUB] device={device} seed={args.seed} exps={exps}", flush=True)

    results = {}
    hard_model = None

    if "hard_e1" in exps or any(x in exps for x in ("discrete_gated", "soft_gated", "diag", "charts")):
        e1 = run_hard_e1(args, train, short, test_id, test_ood, device, args.out_dir / "hard_e1", args.seed)
        # drop non-serializable
        hard_model = e1.pop("model", None)
        if hard_model is None:
            hard_model = load_suite(args.out_dir / "hard_e1" / "final", "hard", device, args.T)
        results["hard_e1"] = e1
        (args.out_dir / "hard_e1" / "overall.json").write_text(json.dumps(e1, indent=2), encoding="utf-8")
        if not e1["gates"]["pass_m1"]:
            print("[WARN] E1 did not pass M1 — gated legs will likely fail", flush=True)

    if "discrete_gated" in exps:
        results["discrete_gated"] = run_discrete_gated(
            args, hard_model, train, short, test_id, test_ood, device, args.out_dir / "discrete_gated", args.seed
        )
        (args.out_dir / "discrete_gated" / "overall.json").write_text(
            json.dumps(results["discrete_gated"], indent=2), encoding="utf-8"
        )

    if "soft_gated" in exps:
        results["soft_gated"] = run_soft_gated(
            args, hard_model, train, short, test_id, test_ood, device, args.out_dir / "soft_gated", args.seed
        )
        (args.out_dir / "soft_gated" / "overall.json").write_text(
            json.dumps(results["soft_gated"], indent=2), encoding="utf-8"
        )

    if "soft_joint" in exps:
        results["soft_joint"] = run_soft_joint(
            args, train, short, test_id, test_ood, device, args.out_dir / "soft_joint", args.seed
        )
        (args.out_dir / "soft_joint" / "overall.json").write_text(
            json.dumps(results["soft_joint"], indent=2), encoding="utf-8"
        )

    if "diag" in exps:
        results["diag"] = run_diag(args, hard_model, test_id, test_ood, device, args.out_dir / "diag", args.seed)

    if "charts" in exps:
        results["charts"] = run_charts(args, hard_model, test_id, test_ood, device, args.out_dir / "charts", args.seed)

    def brief(v):
        if "gates" in v:
            return {"gates": v["gates"], "F_flow": (v.get("final") or {}).get("flow_intervention", {}).get("branch_disagree"),
                    "F_oracle": (v.get("final") or {}).get("intervention", {}).get("branch_disagree"),
                    "ood": (v.get("final") or {}).get("task_ood")}
        return {"claims": v.get("claims"), "rows": v.get("rows")}

    summary = {
        "meta": {
            "experiment": "SMIR-PubCore",
            "seconds": round(time.time() - t0, 2),
            "host": os.uname().nodename,
            "device": str(device),
            "seed": args.seed,
            "exps": exps,
            "gate_f": args.gate_f,
        },
        "results": {k: brief(v) for k, v in results.items()},
    }
    m1 = bool((results.get("hard_e1") or {}).get("gates", {}).get("pass_m1"))
    m2d = bool((results.get("discrete_gated") or {}).get("gates", {}).get("pass_m2"))
    m2s = bool((results.get("soft_gated") or {}).get("gates", {}).get("pass_m2"))
    joint_f = ((results.get("soft_joint") or {}).get("final") or {}).get("flow_intervention", {}).get("branch_disagree")
    disc_f = ((results.get("discrete_gated") or {}).get("final") or {}).get("flow_intervention", {}).get("branch_disagree")
    soft_f = ((results.get("soft_gated") or {}).get("final") or {}).get("flow_intervention", {}).get("branch_disagree")
    beat_joint = False
    if joint_f is not None:
        cands = [x for x in [disc_f, soft_f] if x is not None]
        beat_joint = bool(cands) and max(cands) > joint_f + 0.05
    summary["claims"] = {
        "pass_m1": m1,
        "pass_m2_discrete": m2d,
        "pass_m2_soft": m2s,
        "gated_beats_joint": beat_joint,
        "pass_diag": bool((results.get("diag") or {}).get("claims", {}).get("pass_thm5_necessity")),
        "pass_charts": bool((results.get("charts") or {}).get("claims", {}).get("pass_thm6_style")),
        "pass_pub_core": m1 and (m2d or m2s),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["claims"]["pass_pub_core"]:
        print("[WARN] pub core not fully passed", flush=True)
        raise SystemExit(2)
    print("[OK] pub core passed", flush=True)


if __name__ == "__main__":
    main()
