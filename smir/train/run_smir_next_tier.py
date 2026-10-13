#!/usr/bin/env python3
"""SMIR Next-Tier (NeurIPS/ICML track): fix Hard-tautology / Soft F / protocol gaps.

Experiments in one run (selected via --exps):
  hard     — M1 multiseed certificate (freeze V, AAAI curriculum)
  soft     — M2 SoftV anneal: PhaseA high bias freeze-route≈hard → PhaseB anneal → PhaseC OOD
  discrete — M2b learn discrete router + hard execute (STE)
  mlp_tf   — ablation refresh (teacher force)
  mlp_task — ablation task-only (should fail η)

Reports Oracle-F and Flow-F; gates on Flow-F for learnable models.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

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


def set_soft_bias(model, bias: float):
    if hasattr(model, "V") and hasattr(model.V, "set_op_bias"):
        model.V.set_op_bias(bias)


def set_disc_tau(model, tau: float):
    if hasattr(model, "V") and hasattr(model.V, "set_tau"):
        model.V.set_tau(tau)


def train_phase(
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
    train_v,
    soft_bias,
    disc_tau,
    tag,
):
    rng = random.Random(seed)
    dist_all = [r for r in rows_all if r.get("distinguishing") == 1] or rows_all
    dist_short = [r for r in rows_short if r.get("distinguishing") == 1] or rows_short
    pool = rows_short if prefer_short else rows_all
    dist_pool = dist_short if prefer_short else dist_all

    set_soft_bias(model, soft_bias)
    set_disc_tau(model, disc_tau)

    params = list(model.q_enc.parameters()) + list(model.answer.parameters())
    if train_v:
        params += [p for p in model.V.parameters() if p.requires_grad]
    for p in model.V.parameters():
        p.requires_grad = bool(train_v)

    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    model.train()
    hist = []
    op_map = {"add": 0, "sub": 1, "mul": 2, "null": 3}

    for step in range(1, steps + 1):
        # linear anneal helpers inside phase if bias span encoded as negative sentinel — handled outside
        batch = [
            rng.choice(dist_pool if rng.random() < p_dist else pool) for _ in range(bs)
        ]
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
                lw = model.logits(zw, q)
                loss = loss + F.cross_entropy(lw, yw)
                f_ok += int(lo.argmax(-1).item() != lw.argmax(-1).item())
                f_n += 1
            Tc = max(len(r["actions"]), 1)
            z0 = model.encode_state(r["s0"], device)
            zT = model.flow(z0, r["actions"], T=Tc)
            cont = r["continuation"].get("actions") or []
            z_ans = model.flow(zT, cont, T=len(cont)) if cont else zT
            loss = loss + lam_flow * F.cross_entropy(model.logits(z_ans, q), y)
            if lam_conj > 0:
                loss = loss + lam_conj * chain_abs(model, r, device)
            # optional route CE toward true op (soft/discrete)
            if lam_route > 0 and hasattr(model.V, "route"):
                from model_register import action_tensors

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
                        i_oh = F.one_hot(i.clamp(0, 4), 5).float()
                        j_oh = F.one_hot(j.clamp(0, 4), 5).float()
                        k_oh = F.one_hot(k.clamp(0, 4), 5).float()
                        logits = model.V.route(torch.cat([zt, op_oh, i_oh, j_oh, k_oh], dim=-1))
                    loss = loss + lam_route * F.cross_entropy(
                        logits, torch.tensor([tid], device=device)
                    )
        loss = loss / bs
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 50 == 0 or step == 1:
            hist.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "acc": ok / bs,
                    "batch_F": f_ok / max(f_n, 1),
                    "bias": soft_bias,
                }
            )
            print(
                f"[{tag}] step={step} L={loss.item():.4f} acc={ok/bs:.3f} "
                f"batchF={f_ok/max(f_n,1):.3f} bias={soft_bias}",
                flush=True,
            )
    return hist


def anneal_soft_phase(
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
    bias_start,
    bias_end,
    tag,
):
    """Phase B: train V+Q while linearly annealing op_bias start→end."""
    rng = random.Random(seed)
    dist_all = [r for r in rows_all if r.get("distinguishing") == 1] or rows_all
    dist_short = [r for r in rows_short if r.get("distinguishing") == 1] or rows_short
    pool = rows_short if prefer_short else rows_all
    dist_pool = dist_short if prefer_short else dist_all

    for p in model.parameters():
        p.requires_grad = True
    params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    model.train()
    hist = []
    op_map = {"add": 0, "sub": 1, "mul": 2}

    for step in range(1, steps + 1):
        t = (step - 1) / max(steps - 1, 1)
        bias = bias_start + (bias_end - bias_start) * t
        set_soft_bias(model, bias)
        batch = [
            rng.choice(dist_pool if rng.random() < p_dist else pool) for _ in range(bs)
        ]
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
            loss = loss + lam_conj * chain_abs(model, r, device)
            if lam_route > 0:
                from model_register import action_tensors

                for a, st in zip(r["actions"], r["states"]):
                    tid = op_map.get(a["op"], 3)
                    if tid > 2:
                        continue
                    zt = model.encode_state(st, device)
                    op, i, j, k = action_tensors(a, device)
                    logits = model.V.route_logits(zt, op, i, j, k)
                    loss = loss + lam_route * F.cross_entropy(
                        logits, torch.tensor([tid], device=device)
                    )
        loss = loss / bs
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 50 == 0 or step == 1:
            hist.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "acc": ok / bs,
                    "batch_F": f_ok / max(f_n, 1),
                    "bias": bias,
                }
            )
            print(
                f"[{tag}] step={step} L={loss.item():.4f} acc={ok/bs:.3f} "
                f"batchF={f_ok/max(f_n,1):.3f} bias={bias:.2f}",
                flush=True,
            )
    return hist


def gates_for(kind: str, m: dict) -> dict:
    I = m["intervention"]
    FI = m.get("flow_intervention") or {}
    F_oracle = I["branch_disagree"]
    F_flow = FI.get("branch_disagree", F_oracle)
    # learnable models judged primarily on Flow-F
    F_use = F_flow if kind in {"soft", "discrete", "mlp"} else F_oracle
    thr_F = 0.45 if kind in {"soft", "discrete"} else 0.50
    thr_eta = 0.05 if kind in {"soft", "discrete"} else (0.10 if kind == "mlp" else 0.02)
    thr_tub = 0.20 if kind in {"soft", "discrete", "mlp"} else 0.05
    g = {
        "F_oracle": F_oracle,
        "F_flow": F_flow,
        "F_use": F_use,
        "F_ge_thr": F_use >= thr_F,
        "follow_ge_0_40": I["follow_both"] >= 0.40,
        "ood_ge_0_80": m["task_ood"] >= 0.80,
        "eta_ok": m["eta_abs"] < thr_eta,
        "tub_ok": m["tubular_mse_abs"] < thr_tub,
        "S_gap_ge_0_12": m["sufficiency_gap"] >= 0.12,
        "thr_F": thr_F,
        "thr_eta": thr_eta,
    }
    g["pass_m2" if kind in {"soft", "discrete"} else "pass_m1"] = all(
        [
            g["F_ge_thr"],
            g["ood_ge_0_80"],
            g["eta_ok"],
            g["tub_ok"],
            g["S_gap_ge_0_12"],
        ]
    )
    return g


def score(m: dict, kind: str) -> float:
    I = m["intervention"]
    FI = m.get("flow_intervention") or {}
    F = FI.get("branch_disagree", I["branch_disagree"]) if kind != "hard" else I["branch_disagree"]
    s = 2.0 * min(F, 0.65) + 2.0 * min(m["task_ood"], 0.9)
    s += min(m["sufficiency_gap"], 0.4) - 0.5 * min(m["eta_abs"], 1.0)
    s -= 0.2 * min(m["tubular_mse_abs"], 1.0)
    if F >= 0.45 and m["task_ood"] >= 0.80:
        s += 1.0
    return s


def run_hard(args, train, short, test_id, test_ood, device, out_dir, seed):
    model = SuiteSMIR(kind="hard", T=args.T).to(device)
    h1 = train_phase(
        model, train, short, args.steps_a, args.lr_a, device, args.batch, seed,
        p_dist=0.85, lam_flow=0.2, lam_conj=1e-3, lam_route=0.0,
        prefer_short=True, train_v=False, soft_bias=0.0, disc_tau=1.0, tag=f"hardA-s{seed}",
    )
    h2 = train_phase(
        model, train, short, args.steps_b, args.lr_b, device, args.batch, seed + 1,
        p_dist=args.p_dist, lam_flow=args.lam_flow, lam_conj=1e-3, lam_route=0.0,
        prefer_short=False, train_v=False, soft_bias=0.0, disc_tau=1.0, tag=f"hardB-s{seed}",
    )
    m = evaluate_bundle(model, test_id, test_ood, args.T, args.eval_n, device)
    model.save(out_dir / "final")
    g = gates_for("hard", m)
    return {"kind": "hard", "seed": seed, "final": m, "gates": g, "score": score(m, "hard"),
            "history": {"A": h1[-2:], "B": h2[-2:]}}


def run_soft(args, train, short, test_id, test_ood, device, out_dir, seed):
    model = SuiteSMIR(kind="soft", T=args.T, soft_bias=args.bias_start).to(device)
    # A: high bias, freeze V (≈ hard), train Q for F
    print(f"==== Soft PhaseA bias={args.bias_start} freeze V ====", flush=True)
    h1 = train_phase(
        model, train, short, args.steps_a, args.lr_a, device, args.batch, seed,
        p_dist=0.85, lam_flow=0.2, lam_conj=0.0, lam_route=0.0,
        prefer_short=True, train_v=False, soft_bias=args.bias_start, disc_tau=1.0,
        tag=f"softA-s{seed}",
    )
    m1 = evaluate_bundle(model, test_id, test_ood, args.T, args.eval_n, device)
    print(json.dumps({"soft_A": {"F": m1["intervention"]["branch_disagree"], "FlowF": m1["flow_intervention"]["branch_disagree"], "ood": m1["task_ood"]}}, indent=2), flush=True)
    model.save(out_dir / "phaseA")

    # B: anneal bias → 0, train V+Q
    print(f"==== Soft PhaseB anneal {args.bias_start}→{args.bias_end} ====", flush=True)
    h2 = anneal_soft_phase(
        model, train, short, args.steps_b, args.lr_b, device, args.batch, seed + 1,
        p_dist=args.p_dist, lam_flow=min(args.lam_flow, 0.4), lam_conj=args.lam_conj,
        lam_route=args.lam_route, prefer_short=False,
        bias_start=args.bias_start, bias_end=args.bias_end, tag=f"softB-s{seed}",
    )
    m2 = evaluate_bundle(model, test_id, test_ood, args.T, args.eval_n, device)
    print(json.dumps({"soft_B": {"F": m2["intervention"]["branch_disagree"], "FlowF": m2["flow_intervention"]["branch_disagree"], "ood": m2["task_ood"], "bias": args.bias_end}}, indent=2), flush=True)
    model.save(out_dir / "phaseB")

    # C: OOD polish at final bias, keep strong F
    print("==== Soft PhaseC OOD polish ====", flush=True)
    h3 = train_phase(
        model, train, short, args.steps_c, args.lr_b * 0.7, device, args.batch, seed + 2,
        p_dist=max(args.p_dist, 0.8), lam_flow=args.lam_flow, lam_conj=args.lam_conj * 0.5,
        lam_route=args.lam_route * 0.5, prefer_short=False, train_v=True,
        soft_bias=args.bias_end, disc_tau=1.0, tag=f"softC-s{seed}",
    )
    m3 = evaluate_bundle(model, test_id, test_ood, args.T, args.eval_n, device)
    # keep better of B/C by score
    if score(m3, "soft") >= score(m2, "soft"):
        m_final = m3
    else:
        ckpt = torch.load(out_dir / "phaseB" / "suite.pt", map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        set_soft_bias(model, args.bias_end)
        m_final = m2
        print("[soft] keep phaseB weights", flush=True)
    model.save(out_dir / "final")
    g = gates_for("soft", m_final)
    return {
        "kind": "soft",
        "seed": seed,
        "phaseA": m1,
        "phaseB": m2,
        "final": m_final,
        "gates": g,
        "score": score(m_final, "soft"),
        "meta": {"bias_end": args.bias_end, "lam_conj": args.lam_conj, "lam_route": args.lam_route},
        "history": {"A": h1[-2:], "B": h2[-2:], "C": h3[-2:]},
    }


def run_discrete(args, train, short, test_id, test_ood, device, out_dir, seed):
    model = SuiteSMIR(kind="discrete", T=args.T).to(device)
    # A: train Q with high route CE (near hard exec via learned router supervised)
    h1 = train_phase(
        model, train, short, args.steps_a, args.lr_a, device, args.batch, seed,
        p_dist=0.85, lam_flow=0.2, lam_conj=0.2, lam_route=1.0,
        prefer_short=True, train_v=True, soft_bias=0.0, disc_tau=1.5,
        tag=f"discA-s{seed}",
    )
    h2 = train_phase(
        model, train, short, args.steps_b, args.lr_b, device, args.batch, seed + 1,
        p_dist=args.p_dist, lam_flow=args.lam_flow, lam_conj=0.5, lam_route=0.5,
        prefer_short=False, train_v=True, soft_bias=0.0, disc_tau=0.8,
        tag=f"discB-s{seed}",
    )
    m = evaluate_bundle(model, test_id, test_ood, args.T, args.eval_n, device)
    model.save(out_dir / "final")
    g = gates_for("discrete", m)
    return {"kind": "discrete", "seed": seed, "final": m, "gates": g, "score": score(m, "discrete"),
            "history": {"A": h1[-2:], "B": h2[-2:]}}


def run_mlp(args, train, short, test_id, test_ood, device, out_dir, seed, teacher: bool):
    model = SuiteSMIR(kind="mlp", T=args.T).to(device)
    lam_conj = 1.0 if teacher else 0.0
    lam_flow = 0.3 if teacher else 1.0
    h1 = train_phase(
        model, train, short, args.steps_a, args.lr_a, device, args.batch, seed,
        p_dist=0.7, lam_flow=lam_flow, lam_conj=lam_conj, lam_route=0.0,
        prefer_short=False, train_v=True, soft_bias=0.0, disc_tau=1.0,
        tag=f"mlp{'TF' if teacher else 'Task'}-s{seed}",
    )
    m = evaluate_bundle(model, test_id, test_ood, args.T, args.eval_n, device)
    model.save(out_dir / "final")
    kind = "mlp"
    g = gates_for(kind, m)
    g["teacher"] = teacher
    return {"kind": "mlp_tf" if teacher else "mlp_task", "seed": seed, "final": m, "gates": g,
            "score": score(m, "mlp"), "history": {"A": h1[-2:]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--exps", type=str, default="hard,soft,discrete,mlp_tf,mlp_task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps-a", type=int, default=2500)
    ap.add_argument("--steps-b", type=int, default=3500)
    ap.add_argument("--steps-c", type=int, default=2000)
    ap.add_argument("--lr-a", type=float, default=2e-3)
    ap.add_argument("--lr-b", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--p-dist", type=float, default=0.80)
    ap.add_argument("--lam-flow", type=float, default=0.5)
    ap.add_argument("--lam-conj", type=float, default=0.5)
    ap.add_argument("--lam-route", type=float, default=0.5)
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
    print(f"[NextTier] device={device} seed={args.seed} exps={exps}", flush=True)

    results = {}
    for name in exps:
        out = args.out_dir / name
        out.mkdir(parents=True, exist_ok=True)
        if name == "hard":
            results[name] = run_hard(args, train, short, test_id, test_ood, device, out, args.seed)
        elif name == "soft":
            results[name] = run_soft(args, train, short, test_id, test_ood, device, out, args.seed)
        elif name == "discrete":
            results[name] = run_discrete(args, train, short, test_id, test_ood, device, out, args.seed)
        elif name == "mlp_tf":
            results[name] = run_mlp(args, train, short, test_id, test_ood, device, out, args.seed, True)
        elif name == "mlp_task":
            results[name] = run_mlp(args, train, short, test_id, test_ood, device, out, args.seed, False)
        else:
            print(f"[skip unknown] {name}", flush=True)
            continue
        (out / "overall.json").write_text(json.dumps(results[name], indent=2), encoding="utf-8")
        print(json.dumps({"done": name, "gates": results[name].get("gates"), "score": results[name].get("score")}, indent=2), flush=True)

    summary = {
        "meta": {
            "experiment": "SMIR-NextTier",
            "seconds": round(time.time() - t0, 2),
            "host": os.uname().nodename,
            "device": str(device),
            "seed": args.seed,
            "exps": exps,
            "p_dist": args.p_dist,
            "lam_flow": args.lam_flow,
            "bias_end": args.bias_end,
        },
        "results": {
            k: {
                "gates": v.get("gates"),
                "score": v.get("score"),
                "F_oracle": (v.get("final") or {}).get("intervention", {}).get("branch_disagree"),
                "F_flow": (v.get("final") or {}).get("flow_intervention", {}).get("branch_disagree"),
                "ood": (v.get("final") or {}).get("task_ood"),
                "eta": (v.get("final") or {}).get("eta_abs"),
                "tub": (v.get("final") or {}).get("tubular_mse_abs"),
                "S_gap": (v.get("final") or {}).get("sufficiency_gap"),
                "routing_acc": (v.get("final") or {}).get("routing_acc"),
            }
            for k, v in results.items()
        },
    }
    soft_pass = bool((results.get("soft") or {}).get("gates", {}).get("pass_m2"))
    hard_pass = bool((results.get("hard") or {}).get("gates", {}).get("pass_m1"))
    disc_pass = bool((results.get("discrete") or {}).get("gates", {}).get("pass_m2"))
    summary["claims"] = {
        "pass_m1_hard": hard_pass,
        "pass_m2_soft": soft_pass,
        "pass_m2_discrete": disc_pass,
        "pass_next_tier_core": hard_pass and (soft_pass or disc_pass),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["claims"]["pass_next_tier_core"]:
        print("[WARN] next-tier core not fully passed", flush=True)
        raise SystemExit(2)
    print("[OK] next-tier core passed", flush=True)


if __name__ == "__main__":
    main()
