#!/usr/bin/env python3
"""SMIR main experiment (unified SCF).

Architecture: Semantic Chart Flow (E -> V^T -> Q(z_T) only).
Training:   L = L_task + λ L_flow   (conjugacy energy only; no loss pile-up)
Methods:    scf | cont_ce (λ=0) | outcome (no flow)
Eval:       task, η(conjugacy), tubular, intervention, sufficiency, length curve
Saves:      adapter + heads only
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

from model_smir import OP2ID, SMIRModel


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"{path}:{i}: {e}") from e
    return rows


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ops_tensor(row: dict, T: int, device) -> torch.Tensor:
    ids = [OP2ID.get(a["op"], 3) for a in row["actions"][:T]]
    while len(ids) < T:
        ids.append(OP2ID["null"])
    return torch.tensor([ids[:T]], dtype=torch.long, device=device)


def trainable(model: SMIRModel):
    for p in model.parameters():
        if p.requires_grad:
            yield p


def flow_loss(model: SMIRModel, row: dict) -> torch.Tensor:
    """Conjugacy energy: V(E(s_t), a_t) ≈ E(s_{t+1}) - E(s_t)  (Δt=1)."""
    z_t = model.encode_text(row["prompt_state_t"])
    z_tp1 = model.encode_text(row["prompt_state_tp1"]).detach()
    op_t = torch.tensor([OP2ID.get(row["step_op"], 3)], device=model.device)
    z_next = model.one_step(z_t, op_t)
    return F.mse_loss(z_next, z_tp1)


def train_scf(model, rows, steps, lr, T, lam_flow, seed, tag="SCF", log_every=50):
    rng = random.Random(seed)
    opt = torch.optim.AdamW(trainable(model), lr=lr)
    hist = []
    model.train()
    for step in range(1, steps + 1):
        row = rng.choice(rows)
        y = torch.tensor([row["label"]], dtype=torch.long, device=model.device)
        z0 = model.encode_text(row["prompt"])
        zT = model.flow(z0, ops_tensor(row, T, model.device))
        loss_task = F.cross_entropy(model.logits(zT), y)
        loss_f = flow_loss(model, row)
        loss = loss_task + lam_flow * loss_f
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(trainable(model)), 1.0)
        opt.step()
        if step % log_every == 0 or step == 1:
            acc = float((model.logits(zT).argmax(-1) == y).float().item())
            hist.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "task": float(loss_task.item()),
                    "flow": float(loss_f.item()),
                    "acc": acc,
                }
            )
            print(
                f"[{tag}] step={step} L={loss.item():.4f} task={loss_task.item():.4f} "
                f"flow={loss_f.item():.4f} acc={acc:.0f}",
                flush=True,
            )
    return hist


def train_outcome(model, rows, steps, lr, seed, log_every=50):
    rng = random.Random(seed)
    opt = torch.optim.AdamW(trainable(model), lr=lr)
    hist = []
    model.train()
    for step in range(1, steps + 1):
        row = rng.choice(rows)
        y = torch.tensor([row["label"]], dtype=torch.long, device=model.device)
        z0 = model.encode_text(row["prompt"])
        loss = F.cross_entropy(model.logits(z0), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == 1:
            hist.append({"step": step, "loss": float(loss.item())})
            print(f"[Outcome] step={step} L={loss.item():.4f}", flush=True)
    return hist


@torch.no_grad()
def evaluate(model, rows_id, rows_ood, T, n, use_flow: bool):
    def task_acc(rows, flow, T_use):
        rows = rows[:n]
        ok = 0
        for r in rows:
            z0 = model.encode_text(r["prompt"])
            z = model.flow(z0, ops_tensor(r, T_use, model.device)) if flow else z0
            ok += int(model.logits(z).argmax(-1).item() == r["label"])
        return ok / max(len(rows), 1)

    # conjugacy η
    conj = []
    for r in rows_id[:n]:
        z_t = model.encode_text(r["prompt_state_t"])
        z_tp1 = model.encode_text(r["prompt_state_tp1"])
        op = torch.tensor([OP2ID.get(r["step_op"], 3)], device=model.device)
        conj.append(float(F.mse_loss(model.one_step(z_t, op), z_tp1).item()))
    eta = sum(conj) / max(len(conj), 1)

    # tubular: ||z_T - E(s*)||
    tub = []
    for r in rows_id[:n]:
        z0 = model.encode_text(r["prompt"])
        zT = model.flow(z0, ops_tensor(r, T, model.device)) if use_flow else z0
        z_on = model.encode_text(r["prompt_state_enc"])
        tub.append(float(F.mse_loss(zT, z_on).item()))
    tub_m = sum(tub) / max(len(tub), 1)

    # intervention / off-manifold
    rows_d = [r for r in rows_id if r.get("distinguishing") == 1][:n]
    disagree = follow = on_ok = off_ok = 0
    for r in rows_d:
        z_on = model.encode_text(r["prompt_state_enc"])
        z_w = model.encode_text(r["prompt_state_wrong_enc"])
        p_on = model.logits(z_on).argmax(-1).item()
        p_w = model.logits(z_w).argmax(-1).item()
        disagree += int(p_on != p_w)
        follow += int(p_on == r["label"] and p_w == r["label_wrong"])
        on_ok += int(p_on == r["label"])
        off_ok += int(model.logits(torch.randn_like(z_on)).argmax(-1).item() == r["label"])
    m = max(len(rows_d), 1)

    # sufficiency: true z0 vs noise z0 through same flow
    ok_f = ok_s = 0
    if use_flow:
        for r in rows_id[:n]:
            ops = ops_tensor(r, T, model.device)
            z0 = model.encode_text(r["prompt"])
            ok_f += int(model.logits(model.flow(z0, ops)).argmax(-1).item() == r["label"])
            ok_s += int(model.logits(model.flow(torch.randn_like(z0), ops)).argmax(-1).item() == r["label"])
        s_gap = (ok_f - ok_s) / max(n, 1)
    else:
        s_gap = 0.0

    # length curve T, T+2, T+4
    length_curve = {str(t): task_acc(rows_id, use_flow, t) for t in [T, T + 2, T + 4]}

    out = {
        "task_id": task_acc(rows_id, use_flow, T),
        "task_ood": task_acc(rows_ood, use_flow, T),
        "eta_conjugacy": eta,
        "tubular_mse": tub_m,
        "intervention": {
            "n": len(rows_d),
            "branch_disagree": disagree / m,
            "follow_both": follow / m,
            "on_manifold_acc": on_ok / m,
            "off_manifold_acc": off_ok / m,
            "on_minus_off": (on_ok - off_ok) / m,
        },
        "sufficiency_gap": s_gap,
        "length_curve": length_curve,
    }
    print(json.dumps(out, indent=2), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--z-dim", type=int, default=128)
    ap.add_argument("--lam-flow", type=float, default=1.0)
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    t0 = time.time()

    train = load_jsonl(args.data_dir / "train.jsonl")
    test_id = load_jsonl(args.data_dir / "test_id.jsonl")
    test_ood = load_jsonl(args.data_dir / "test_ood.jsonl")
    print(
        f"[data] train={len(train)} T={args.T} z_dim={args.z_dim} lam_flow={args.lam_flow}",
        flush=True,
    )

    results = {}

    # ---- A1: SCF (unified) ----
    print("==== A1 Train SCF (L_task + λ L_flow) ====", flush=True)
    scf = SMIRModel(args.model, z_dim=args.z_dim, T=args.T)
    h_scf = train_scf(scf, train, args.steps, args.lr, args.T, args.lam_flow, args.seed, tag="SCF")
    print("==== Eval SCF ====", flush=True)
    results["scf"] = evaluate(scf, test_id, test_ood, args.T, args.eval_n, use_flow=True)
    results["scf"]["history"] = h_scf[-5:]
    scf.save_light(args.out_dir / "scf")
    del scf
    torch.cuda.empty_cache()

    # ---- A2 / B1: ContCE = SCF arch with λ=0 ----
    print("==== A2/B1 Train ContCE (λ=0, ablation no L_flow) ====", flush=True)
    cont = SMIRModel(args.model, z_dim=args.z_dim, T=args.T)
    h_cont = train_scf(cont, train, args.steps, args.lr, args.T, 0.0, args.seed + 1, tag="ContCE")
    print("==== Eval ContCE ====", flush=True)
    results["cont_ce"] = evaluate(cont, test_id, test_ood, args.T, args.eval_n, use_flow=True)
    results["cont_ce"]["history"] = h_cont[-5:]
    cont.save_light(args.out_dir / "cont_ce")
    del cont
    torch.cuda.empty_cache()

    # ---- A3: Outcome ----
    print("==== A3 Train Outcome (no flow) ====", flush=True)
    outm = SMIRModel(args.model, z_dim=args.z_dim, T=args.T)
    h_out = train_outcome(outm, train, args.steps, args.lr, args.seed + 2)
    print("==== Eval Outcome ====", flush=True)
    results["outcome"] = evaluate(outm, test_id, test_ood, args.T, args.eval_n, use_flow=False)
    results["outcome"]["history"] = h_out[-5:]
    outm.save_light(args.out_dir / "outcome")
    del outm
    torch.cuda.empty_cache()

    S, C, O = results["scf"], results["cont_ce"], results["outcome"]
    gates = {
        # geometry / conjugacy
        "scf_eta_lt_cont": S["eta_conjugacy"] < C["eta_conjugacy"] * 0.7,
        "scf_tub_lt_cont": S["tubular_mse"] < C["tubular_mse"] * 0.85,
        # faithfulness
        "scf_F_gt_outcome": S["intervention"]["branch_disagree"]
        > O["intervention"]["branch_disagree"] + 0.08,
        "scf_F_gt_cont": S["intervention"]["branch_disagree"]
        > C["intervention"]["branch_disagree"] + 0.03,
        "scf_on_minus_off_gt_0_10": S["intervention"]["on_minus_off"] >= 0.10,
        # task (auxiliary but required for usable mechanism)
        "scf_ood_ge_0_60": S["task_ood"] >= 0.60,
        "scf_ood_ge_outcome": S["task_ood"] >= O["task_ood"] - 0.02,
        # sufficiency
        "scf_S_gap_gt_cont": S["sufficiency_gap"] > C["sufficiency_gap"] + 0.05,
        # length: T+4 should not collapse much vs ContCE relative drop
        "scf_length_T4_ge_0_55": S["length_curve"][str(args.T + 4)] >= 0.55,
    }
    gates["pass_main_geometry"] = bool(gates["scf_eta_lt_cont"] and gates["scf_tub_lt_cont"])
    gates["pass_main_faithfulness"] = bool(
        gates["scf_F_gt_outcome"] and gates["scf_on_minus_off_gt_0_10"]
    )
    gates["pass_main"] = bool(
        gates["pass_main_geometry"]
        and gates["pass_main_faithfulness"]
        and gates["scf_ood_ge_0_60"]
    )
    # ablation claim: removing L_flow hurts F or eta
    gates["ablation_flow_necessary"] = bool(
        gates["scf_eta_lt_cont"] or gates["scf_F_gt_cont"]
    )

    overall = {
        "meta": {
            "experiment": "SMIR-Main-SCF",
            "doc": "SMIR.md + SMIR_PROOFS.md",
            "design": "unified L_task + λ L_flow only",
            "model": args.model,
            "seconds": round(time.time() - t0, 2),
            "host": os.uname().nodename,
            "steps": args.steps,
            "T": args.T,
            "z_dim": args.z_dim,
            "lam_flow": args.lam_flow,
        },
        "results": results,
        "gates": gates,
        "comparisons": {
            "eta_scf_over_cont": S["eta_conjugacy"] / max(C["eta_conjugacy"], 1e-8),
            "tub_scf_over_cont": S["tubular_mse"] / max(C["tubular_mse"], 1e-8),
            "F_scf_minus_cont": S["intervention"]["branch_disagree"]
            - C["intervention"]["branch_disagree"],
            "F_scf_minus_outcome": S["intervention"]["branch_disagree"]
            - O["intervention"]["branch_disagree"],
            "S_gap_scf_minus_cont": S["sufficiency_gap"] - C["sufficiency_gap"],
            "ood_scf_minus_outcome": S["task_ood"] - O["task_ood"],
            "ood_scf_minus_cont": S["task_ood"] - C["task_ood"],
        },
    }
    (args.out_dir / "overall.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"gates": gates, "comparisons": overall["comparisons"], "meta": overall["meta"]},
            indent=2,
        ),
        flush=True,
    )
    if not gates["pass_main"]:
        print("[WARN] SMIR main gates not fully passed", flush=True)
        raise SystemExit(2)
    print("[OK] SMIR main experiment passed", flush=True)


if __name__ == "__main__":
    main()
