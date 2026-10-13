#!/usr/bin/env python3
"""Exp-1 + Exp-2: State-separable warmup then conjugacy flow with endpoint anchor.

Phase-S: train E+Q on shadow states so intervention F becomes possible.
Phase-F: train V (+ light E) with
    L = L_task + λ_flow L_flow + λ_end ||z_T - sg(E(s*))||^2

Controls: phase_s_only | full (S→F).  Does NOT re-run ContCE/Outcome/plain SCF.

Exp-2 eval patches: relative tubular, off-manifold adversarial, F/on-off, η, length.
Optional Exp-3 stub: --metric-align 1 adds tiny tangent response term in Phase-S.
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
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
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


def trainable(model: SMIRModel, heads_only: bool = False):
    if heads_only:
        for p in list(model.to_z.parameters()) + list(model.V.parameters()) + list(model.answer.parameters()):
            if p.requires_grad:
                yield p
        return
    for p in model.parameters():
        if p.requires_grad:
            yield p


def freeze_backbone(model: SMIRModel, freeze: bool) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = not freeze


def flow_loss(model: SMIRModel, row: dict) -> torch.Tensor:
    z_t = model.encode_text(row["prompt_state_t"])
    z_tp1 = model.encode_text(row["prompt_state_tp1"]).detach()
    op_t = torch.tensor([OP2ID.get(row["step_op"], 3)], device=model.device)
    return F.mse_loss(model.one_step(z_t, op_t), z_tp1)


def train_phase_s(model, rows, steps, lr, seed, metric_align: bool, log_every=50):
    """State-separable warmup: CE on on/wrong encodings (+ optional tangent align)."""
    rng = random.Random(seed)
    # Prefer distinguishing rows when available
    dist = [r for r in rows if r.get("distinguishing") == 1] or rows
    opt = torch.optim.AdamW(trainable(model), lr=lr)
    hist = []
    model.train()
    for step in range(1, steps + 1):
        row = rng.choice(dist)
        y = torch.tensor([row["label"]], dtype=torch.long, device=model.device)
        z_on = model.encode_text(row["prompt_state_enc"])
        loss = F.cross_entropy(model.logits(z_on), y)
        if row.get("distinguishing") == 1:
            yw = torch.tensor([row["label_wrong"]], dtype=torch.long, device=model.device)
            z_w = model.encode_text(row["prompt_state_wrong_enc"])
            loss = loss + F.cross_entropy(model.logits(z_w), yw)
            # margin: push embeddings apart in label direction
            loss = loss + 0.1 * F.relu(1.0 - F.mse_loss(z_on, z_w))
        if metric_align and row.get("distinguishing") == 1:
            # Exp-3 stub: encourage large TV between on/wrong predictive distributions
            p_on = F.softmax(model.logits(z_on), dim=-1)
            p_w = F.softmax(model.logits(z_w), dim=-1)
            tv = 0.5 * (p_on - p_w).abs().sum(dim=-1).mean()
            loss = loss + 0.2 * F.relu(0.5 - tv)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(trainable(model)), 1.0)
        opt.step()
        if step % log_every == 0 or step == 1:
            acc = float((model.logits(z_on).argmax(-1) == y).float().item())
            hist.append({"step": step, "loss": float(loss.item()), "acc": acc})
            print(f"[Phase-S] step={step} L={loss.item():.4f} acc={acc:.0f}", flush=True)
    return hist


def train_phase_f(model, rows, steps, lr, T, lam_flow, lam_end, seed, freeze_e: bool, log_every=50):
    """Conjugacy + task + endpoint tubular anchor."""
    rng = random.Random(seed)
    if freeze_e:
        freeze_backbone(model, True)
        # still train to_z lightly with V and answer
        for p in model.to_z.parameters():
            p.requires_grad = True
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
        with torch.no_grad():
            z_star = model.encode_text(row["prompt_state_enc"])
        loss_end = F.mse_loss(zT, z_star)
        loss = loss_task + lam_flow * loss_f + lam_end * loss_end
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
                    "end": float(loss_end.item()),
                    "acc": acc,
                }
            )
            print(
                f"[Phase-F] step={step} L={loss.item():.4f} task={loss_task.item():.4f} "
                f"flow={loss_f.item():.4f} end={loss_end.item():.4f} acc={acc:.0f}",
                flush=True,
            )
    if freeze_e:
        freeze_backbone(model, False)
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

    conj, tub_abs, tub_rel = [], [], []
    for r in rows_id[:n]:
        z_t = model.encode_text(r["prompt_state_t"])
        z_tp1 = model.encode_text(r["prompt_state_tp1"])
        op = torch.tensor([OP2ID.get(r["step_op"], 3)], device=model.device)
        conj.append(float(F.mse_loss(model.one_step(z_t, op), z_tp1).item()))
        z0 = model.encode_text(r["prompt"])
        zT = model.flow(z0, ops_tensor(r, T, model.device)) if use_flow else z0
        z_on = model.encode_text(r["prompt_state_enc"])
        mse = float(F.mse_loss(zT, z_on).item())
        nE2 = float((z_on * z_on).mean().item()) + 1e-8
        tub_abs.append(mse)
        tub_rel.append(mse / nE2)

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
        # off-manifold: large noise
        z_off = z_on + 3.0 * torch.randn_like(z_on)
        off_ok += int(model.logits(z_off).argmax(-1).item() == r["label"])
    m = max(len(rows_d), 1)

    # sufficiency
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

    # Theorem-1 style: off-manifold adversarial accuracy (high = bad shortcut)
    adv_ok = 0
    for r in rows_id[:n]:
        z_far = 10.0 * torch.randn(1, model.z_dim, device=model.device)
        adv_ok += int(model.logits(z_far).argmax(-1).item() == r["label"])
    adv_acc = adv_ok / max(n, 1)

    length_curve = {str(t): task_acc(rows_id, use_flow, t) for t in [T, T + 2, T + 4]}

    return {
        "task_id": task_acc(rows_id, use_flow, T),
        "task_ood": task_acc(rows_ood, use_flow, T),
        "eta_conjugacy": sum(conj) / max(len(conj), 1),
        "tubular_mse_abs": sum(tub_abs) / max(len(tub_abs), 1),
        "tubular_mse_rel": sum(tub_rel) / max(len(tub_rel), 1),
        "intervention": {
            "n": len(rows_d),
            "branch_disagree": disagree / m,
            "follow_both": follow / m,
            "on_manifold_acc": on_ok / m,
            "off_manifold_acc": off_ok / m,
            "on_minus_off": (on_ok - off_ok) / m,
        },
        "sufficiency_gap": s_gap,
        "offmanifold_adv_acc": adv_acc,
        "length_curve": length_curve,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--mode", choices=["full", "phase_s_only"], default="full")
    ap.add_argument("--steps-s", type=int, default=1500)
    ap.add_argument("--steps-f", type=int, default=2500)
    ap.add_argument("--lr-s", type=float, default=1e-4)
    ap.add_argument("--lr-f", type=float, default=5e-5)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--z-dim", type=int, default=128)
    ap.add_argument("--lam-flow", type=float, default=1.0)
    ap.add_argument("--lam-end", type=float, default=0.1)
    ap.add_argument("--freeze-e-in-f", type=int, default=0)
    ap.add_argument("--metric-align", type=int, default=0, help="Exp-3 stub in Phase-S")
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
        f"[SF] mode={args.mode} steps_s={args.steps_s} steps_f={args.steps_f} "
        f"lam_flow={args.lam_flow} lam_end={args.lam_end} metric_align={args.metric_align}",
        flush=True,
    )

    results = {}
    model = SMIRModel(args.model, z_dim=args.z_dim, T=args.T)

    print("==== Phase-S: state-separable warmup ====", flush=True)
    h_s = train_phase_s(
        model, train, args.steps_s, args.lr_s, args.seed, bool(args.metric_align)
    )
    print("==== Eval after Phase-S ====", flush=True)
    results["phase_s"] = evaluate(model, test_id, test_ood, args.T, args.eval_n, use_flow=False)
    results["phase_s"]["history"] = h_s[-5:]
    print(json.dumps(results["phase_s"], indent=2), flush=True)
    model.save_light(args.out_dir / "phase_s")

    if args.mode == "full":
        print("==== Phase-F: conjugacy + endpoint tubular ====", flush=True)
        h_f = train_phase_f(
            model,
            train,
            args.steps_f,
            args.lr_f,
            args.T,
            args.lam_flow,
            args.lam_end,
            args.seed + 1,
            freeze_e=bool(args.freeze_e_in_f),
        )
        print("==== Eval after Phase-F (full) ====", flush=True)
        results["full"] = evaluate(model, test_id, test_ood, args.T, args.eval_n, use_flow=True)
        results["full"]["history"] = h_f[-5:]
        print(json.dumps(results["full"], indent=2), flush=True)
        model.save_light(args.out_dir / "full")

    # Gates focused on D1/D2 fixes — no ContCE/Outcome repeat
    S = results["phase_s"]
    F = results.get("full", S)
    gates = {
        "phase_s_F_ge_0_25": S["intervention"]["branch_disagree"] >= 0.25,
        "phase_s_follow_ge_0_20": S["intervention"]["follow_both"] >= 0.20,
        "phase_s_on_minus_off_ge_0_10": S["intervention"]["on_minus_off"] >= 0.10,
        "full_F_ge_0_25": F["intervention"]["branch_disagree"] >= 0.25,
        "full_on_minus_off_ge_0_10": F["intervention"]["on_minus_off"] >= 0.10,
        "full_eta_lt_0_05": F["eta_conjugacy"] < 0.05,
        "full_tub_rel_lt_2": F["tubular_mse_rel"] < 2.0,
        "full_ood_ge_0_60": F["task_ood"] >= 0.60,
        "full_adv_lt_0_55": F["offmanifold_adv_acc"] < 0.55,
        "full_S_gap_ge_0_08": F["sufficiency_gap"] >= 0.08,
        "full_length_T4_ge_0_55": F["length_curve"][str(args.T + 4)] >= 0.55,
    }
    gates["pass_phase_s"] = bool(
        gates["phase_s_F_ge_0_25"] and gates["phase_s_on_minus_off_ge_0_10"]
    )
    gates["pass_full"] = bool(
        gates["pass_phase_s"]
        and gates["full_F_ge_0_25"]
        and gates["full_on_minus_off_ge_0_10"]
        and gates["full_eta_lt_0_05"]
        and gates["full_tub_rel_lt_2"]
        and gates["full_ood_ge_0_60"]
    )
    # Ablation claim without ContCE: Phase-S alone has F; full keeps F and adds η/tub
    if args.mode == "full":
        gates["flow_preserves_F"] = bool(
            F["intervention"]["branch_disagree"]
            >= S["intervention"]["branch_disagree"] - 0.05
        )
        gates["full_improves_eta_vs_phase_s"] = bool(
            F["eta_conjugacy"] <= S["eta_conjugacy"] * 1.05 + 0.01
        )

    overall = {
        "meta": {
            "experiment": "SMIR-SF-Pipeline",
            "stages": ["Exp-0 diag separate", "Exp-1 Phase-S/F", "Exp-2 eval patches"],
            "mode": args.mode,
            "model": args.model,
            "seconds": round(time.time() - t0, 2),
            "host": os.uname().nodename,
            "steps_s": args.steps_s,
            "steps_f": args.steps_f if args.mode == "full" else 0,
            "T": args.T,
            "z_dim": args.z_dim,
            "lam_flow": args.lam_flow,
            "lam_end": args.lam_end,
            "metric_align": args.metric_align,
        },
        "results": results,
        "gates": gates,
    }
    (args.out_dir / "overall.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print(json.dumps({"gates": gates, "meta": overall["meta"]}, indent=2), flush=True)
    if args.mode == "full" and not gates["pass_full"]:
        print("[WARN] SMIR-SF full gates not passed", flush=True)
        raise SystemExit(2)
    if args.mode == "phase_s_only" and not gates["pass_phase_s"]:
        print("[WARN] Phase-S gates not passed", flush=True)
        raise SystemExit(2)
    print("[OK] SMIR-SF pipeline passed", flush=True)


if __name__ == "__main__":
    main()
