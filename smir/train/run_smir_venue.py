#!/usr/bin/env python3
"""SMIR formal venue experiment on Qwen2.5-7B.

Implements variational objective (SMIR.md eq.14 proxies):
  task + conjugacy + metric alignment + tubular attraction + normal indifference
Baselines: ContCE (task only), Outcome (no flow).
Saves LoRA adapter + heads only (not full 7B).
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
            if not line.strip():
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


def train_smir(model, rows, steps, lr, T, lam, seed, log_every=50):
    rng = random.Random(seed)
    opt = torch.optim.AdamW(trainable(model), lr=lr)
    hist = []
    model.train()
    for step in range(1, steps + 1):
        row = rng.choice(rows)
        d = model.device
        y = torch.tensor([row["label"]], dtype=torch.long, device=d)
        ops = ops_tensor(row, T, d)

        # --- encode ---
        z0 = model.encode_text(row["prompt"])
        zT = model.flow(z0, ops)
        logits = model.logits(zT)
        loss_task = F.cross_entropy(logits, y)

        # --- conjugacy η: V(E(s_t), a_t) ≈ E(s_{t+1}) ---
        z_t = model.encode_text(row["prompt_state_t"])
        z_tp1 = model.encode_text(row["prompt_state_tp1"]).detach()
        op_t = torch.tensor([OP2ID.get(row["step_op"], 3)], device=d)
        z_pred = model.one_step(z_t, op_t)
        loss_conj = F.mse_loss(z_pred, z_tp1)

        # --- tubular attraction: zT close to E(final state); far from noise ---
        z_on = model.encode_text(row["prompt_state_enc"]).detach()
        loss_tub = F.mse_loss(zT, z_on)
        z_noise = torch.randn_like(z_on)
        # push answer from on-manifold encoding; noise should not match y as easily via margin
        loss_tub = loss_tub + F.relu(
            1.0
            - (
                F.cross_entropy(model.logits(z_noise), y, reduction="none")
                - F.cross_entropy(model.logits(z_on), y, reduction="none")
            )
        ).mean()

        # --- metric alignment proxy: local finite-difference response ---
        # perturb toward wrong vs correct state encodings; logits gap should track distinguishing
        z_w = model.encode_text(row["prompt_state_wrong_enc"]).detach()
        eps = 0.25
        z_plus = z_on + eps * (z_on - z_w)
        z_minus = z_on + eps * (z_w - z_on)
        # g*-proxy: distinguishing pairs should induce larger logit change
        delta = (model.logits(z_plus) - model.logits(z_minus)).abs().mean()
        # encourage larger delta when distinguishing else keep mild
        target = 1.0 if row["distinguishing"] == 1 else 0.2
        loss_met = F.mse_loss(delta, torch.tensor(target, device=d))

        # --- normal indifference: additive isotropic noise on z_on should not flip ---
        z_n = z_on + 0.05 * torch.randn_like(z_on)
        loss_norm = F.mse_loss(F.softmax(model.logits(z_n), dim=-1), F.softmax(model.logits(z_on), dim=-1).detach())

        # --- intervention contrast (faithfulness) ---
        loss_iv = F.cross_entropy(model.logits(z_on), y) + F.cross_entropy(
            model.logits(z_w), torch.tensor([row["label_wrong"]], dtype=torch.long, device=d)
        )

        loss = (
            loss_task
            + lam["conj"] * loss_conj
            + lam["tub"] * loss_tub
            + lam["met"] * loss_met
            + lam["norm"] * loss_norm
            + lam["iv"] * loss_iv
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(trainable(model)), 1.0)
        opt.step()

        if step % log_every == 0 or step == 1:
            hist.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "task": float(loss_task.item()),
                    "conj": float(loss_conj.item()),
                    "tub": float(loss_tub.item()),
                    "met": float(loss_met.item()),
                    "iv": float(loss_iv.item()),
                    "acc": float((logits.argmax(-1) == y).float().item()),
                }
            )
            print(
                f"[SMIR] step={step} L={loss.item():.3f} task={loss_task.item():.3f} "
                f"conj={loss_conj.item():.3f} tub={loss_tub.item():.3f} "
                f"met={loss_met.item():.3f} iv={loss_iv.item():.3f}",
                flush=True,
            )
    return hist


def train_cont(model, rows, steps, lr, T, seed, log_every=50):
    rng = random.Random(seed)
    opt = torch.optim.AdamW(trainable(model), lr=lr)
    hist = []
    model.train()
    for step in range(1, steps + 1):
        row = rng.choice(rows)
        y = torch.tensor([row["label"]], dtype=torch.long, device=model.device)
        z0 = model.encode_text(row["prompt"])
        zT = model.flow(z0, ops_tensor(row, T, model.device))
        loss = F.cross_entropy(model.logits(zT), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == 1:
            hist.append({"step": step, "loss": float(loss.item())})
            print(f"[ContCE] step={step} L={loss.item():.4f}", flush=True)
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
def eval_suite(model, rows_id, rows_ood, T, n, use_flow=True):
    def task(rows, flow):
        rows = rows[:n]
        ok = 0
        for r in rows:
            z0 = model.encode_text(r["prompt"])
            z = model.flow(z0, ops_tensor(r, T, model.device)) if flow else z0
            ok += int(model.logits(z).argmax(-1).item() == r["label"])
        return {"n": len(rows), "accuracy": ok / max(len(rows), 1)}

    # conjugacy residual η
    conj = []
    for r in rows_id[:n]:
        z_t = model.encode_text(r["prompt_state_t"])
        z_tp1 = model.encode_text(r["prompt_state_tp1"])
        op_t = torch.tensor([OP2ID.get(r["step_op"], 3)], device=model.device)
        conj.append(float(F.mse_loss(model.one_step(z_t, op_t), z_tp1).item()))

    # tubular: ||zT - E(s*)||
    tub = []
    for r in rows_id[:n]:
        z0 = model.encode_text(r["prompt"])
        zT = model.flow(z0, ops_tensor(r, T, model.device)) if use_flow else z0
        z_on = model.encode_text(r["prompt_state_enc"])
        tub.append(float(F.mse_loss(zT, z_on).item()))

    # intervention / off-manifold falsification
    rows_d = [r for r in rows_id if r["distinguishing"] == 1][:n]
    disagree = follow = off_acc = on_acc = 0
    for r in rows_d:
        z_on = model.encode_text(r["prompt_state_enc"])
        z_w = model.encode_text(r["prompt_state_wrong_enc"])
        p_on = model.logits(z_on).argmax(-1).item()
        p_w = model.logits(z_w).argmax(-1).item()
        disagree += int(p_on != p_w)
        follow += int(p_on == r["label"] and p_w == r["label_wrong"])
        on_acc += int(p_on == r["label"])
        z_off = torch.randn_like(z_on)
        off_acc += int(model.logits(z_off).argmax(-1).item() == r["label"])

    m = max(len(rows_d), 1)
    # composition longer T
    comp = task(rows_id, True)
    # temporarily evaluate with T+2 by padding null ops
    ok = 0
    for r in rows_id[:n]:
        ops = ops_tensor(r, T + 2, model.device)
        z0 = model.encode_text(r["prompt"])
        ok += int(model.logits(model.flow(z0, ops)).argmax(-1).item() == r["label"])
    composition = {"T_test": T + 2, "n": n, "accuracy": ok / max(n, 1)}

    # sufficiency gap: true flow vs noise init
    ok_f = ok_s = 0
    for r in rows_id[:n]:
        ops = ops_tensor(r, T, model.device)
        z0 = model.encode_text(r["prompt"])
        ok_f += int(model.logits(model.flow(z0, ops)).argmax(-1).item() == r["label"])
        ok_s += int(model.logits(model.flow(torch.randn_like(z0), ops)).argmax(-1).item() == r["label"])

    return {
        "task_id": task(rows_id, use_flow),
        "task_ood": task(rows_ood, use_flow),
        "conjugacy_mse_mean": sum(conj) / max(len(conj), 1),
        "tubular_mse_mean": sum(tub) / max(len(tub), 1),
        "intervention": {
            "n": len(rows_d),
            "branch_disagree": disagree / m,
            "follow_both": follow / m,
            "on_manifold_acc": on_acc / m,
            "off_manifold_acc": off_acc / m,
            "on_minus_off": (on_acc - off_acc) / m,
        },
        "sufficiency_gap": (ok_f - ok_s) / max(n, 1),
        "composition": composition,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--z-dim", type=int, default=256)
    ap.add_argument("--eval-n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam-conj", type=float, default=0.5)
    ap.add_argument("--lam-tub", type=float, default=0.3)
    ap.add_argument("--lam-met", type=float, default=0.2)
    ap.add_argument("--lam-norm", type=float, default=0.1)
    ap.add_argument("--lam-iv", type=float, default=0.4)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    t0 = time.time()
    train = load_jsonl(args.data_dir / "train.jsonl")
    test_id = load_jsonl(args.data_dir / "test_id.jsonl")
    test_ood = load_jsonl(args.data_dir / "test_ood.jsonl")
    print(f"[data] train={len(train)} model={args.model}", flush=True)

    lam = {
        "conj": args.lam_conj,
        "tub": args.lam_tub,
        "met": args.lam_met,
        "norm": args.lam_norm,
        "iv": args.lam_iv,
    }
    results = {}

    print("==== Train SMIR ====", flush=True)
    m1 = SMIRModel(args.model, z_dim=args.z_dim, T=args.T)
    h1 = train_smir(m1, train, args.steps, args.lr, args.T, lam, args.seed)
    results["smir"] = eval_suite(m1, test_id, test_ood, args.T, args.eval_n, use_flow=True)
    results["smir"]["history"] = h1[-5:]
    print(json.dumps(results["smir"], indent=2), flush=True)
    m1.save_light(args.out_dir / "smir_light")
    del m1
    torch.cuda.empty_cache()

    print("==== Train ContCE ====", flush=True)
    m2 = SMIRModel(args.model, z_dim=args.z_dim, T=args.T)
    h2 = train_cont(m2, train, args.steps, args.lr, args.T, args.seed + 1)
    results["cont_ce"] = eval_suite(m2, test_id, test_ood, args.T, args.eval_n, use_flow=True)
    results["cont_ce"]["history"] = h2[-5:]
    print(json.dumps(results["cont_ce"], indent=2), flush=True)
    m2.save_light(args.out_dir / "cont_light")
    del m2
    torch.cuda.empty_cache()

    print("==== Train Outcome ====", flush=True)
    m3 = SMIRModel(args.model, z_dim=args.z_dim, T=args.T)
    h3 = train_outcome(m3, train, args.steps, args.lr, args.seed + 2)
    results["outcome"] = eval_suite(m3, test_id, test_ood, args.T, args.eval_n, use_flow=False)
    results["outcome"]["history"] = h3[-5:]
    print(json.dumps(results["outcome"], indent=2), flush=True)
    m3.save_light(args.out_dir / "outcome_light")
    del m3
    torch.cuda.empty_cache()

    S, C, O = results["smir"], results["cont_ce"], results["outcome"]
    gates = {
        "smir_ood_ge_0_65": S["task_ood"]["accuracy"] >= 0.65,
        "smir_conj_lt_cont": S["conjugacy_mse_mean"] < C["conjugacy_mse_mean"] * 0.85,
        "smir_tub_lt_cont": S["tubular_mse_mean"] < C["tubular_mse_mean"] * 0.9,
        "smir_F_gt_outcome": S["intervention"]["branch_disagree"]
        > O["intervention"]["branch_disagree"] + 0.08,
        "smir_F_gt_cont": S["intervention"]["branch_disagree"]
        > C["intervention"]["branch_disagree"] + 0.03,
        "smir_on_minus_off_gt_0_15": S["intervention"]["on_minus_off"] >= 0.15,
        "smir_S_gap_gt_cont": S["sufficiency_gap"] > C["sufficiency_gap"] + 0.05,
        "smir_C_ge_0_60": S["composition"]["accuracy"] >= 0.60,
    }
    gates["pass_geometry"] = bool(
        gates["smir_conj_lt_cont"]
        and gates["smir_tub_lt_cont"]
        and gates["smir_on_minus_off_gt_0_15"]
    )
    gates["pass_faithfulness"] = bool(gates["smir_F_gt_outcome"] and gates["smir_F_gt_cont"])
    gates["pass"] = bool(
        gates["pass_geometry"] and gates["pass_faithfulness"] and gates["smir_ood_ge_0_65"]
    )

    overall = {
        "meta": {
            "framework": "SMIR",
            "doc": "SMIR.md",
            "model": args.model,
            "seconds": round(time.time() - t0, 2),
            "host": os.uname().nodename,
            "steps": args.steps,
            "T": args.T,
            "lam": lam,
            "save": "adapter+heads only",
        },
        "results": results,
        "gates": gates,
        "comparisons": {
            "conj_smir_over_cont": S["conjugacy_mse_mean"] / max(C["conjugacy_mse_mean"], 1e-8),
            "tub_smir_over_cont": S["tubular_mse_mean"] / max(C["tubular_mse_mean"], 1e-8),
            "F_smir_minus_cont": S["intervention"]["branch_disagree"]
            - C["intervention"]["branch_disagree"],
            "F_smir_minus_outcome": S["intervention"]["branch_disagree"]
            - O["intervention"]["branch_disagree"],
            "S_gap_smir_minus_cont": S["sufficiency_gap"] - C["sufficiency_gap"],
            "ood_smir_minus_outcome": S["task_ood"]["accuracy"] - O["task_ood"]["accuracy"],
        },
    }
    (args.out_dir / "overall.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print(json.dumps({"gates": gates, "comparisons": overall["comparisons"], "meta": overall["meta"]}, indent=2), flush=True)
    if not gates["pass"]:
        print("[WARN] SMIR venue gates not fully passed", flush=True)
        raise SystemExit(2)
    print("[OK] SMIR venue passed", flush=True)


if __name__ == "__main__":
    main()
