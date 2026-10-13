#!/usr/bin/env python3
"""SMIR Venue-V: freeze Stage-A (E,Q), train only V for conjugacy+tubular.

Top-venue next step after F unlock:
  - Do NOT retrain 7B / ContCE / joint Stage-B
  - Multi-step teacher-forced conjugacy on frozen chart
  - Endpoint tubular + light task through flow (grads → V only)
  - Report oracle upper bound (true state dynamics → E → Q)

Loads: artifacts/runs/smir_cert/stage_a
Saves: artifacts/runs/smir_venue_v/
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

from model_chart import ChartSMIR
from run_smir_cert import (
    encode_answer,
    evaluate,
    load_jsonl,
    need_struct,
    set_seed,
)
from synlogic_hard import apply_action


def freeze_eq(model: ChartSMIR) -> None:
    for p in model.E.parameters():
        p.requires_grad = False
    for p in model.q_enc.parameters():
        p.requires_grad = False
    for p in model.answer.parameters():
        p.requires_grad = False
    for p in model.V.parameters():
        p.requires_grad = True


def v_params(model: ChartSMIR):
    return [p for p in model.V.parameters() if p.requires_grad]


def chain_conjugacy_loss(model: ChartSMIR, row: dict, device) -> torch.Tensor:
    """Σ_t ||V(E(s_t), a_t) - sg(E(s_{t+1}))||^2  (full chain teacher force)."""
    states = row["states"]
    actions = row["actions"]
    loss = torch.zeros((), device=device)
    n = 0
    for t, a in enumerate(actions):
        if t + 1 >= len(states):
            break
        z_t = model.encode_state(states[t], device)
        with torch.no_grad():
            z_tp1 = model.encode_state(states[t + 1], device)
        loss = loss + F.mse_loss(model.one_step(z_t, a), z_tp1)
        n += 1
    return loss / max(n, 1)


def train_v_only(
    model,
    rows,
    steps,
    lr,
    T,
    lam_flow,
    lam_end,
    lam_task,
    seed,
    device,
    batch_size=32,
    log_every=50,
):
    rng = random.Random(seed)
    opt = torch.optim.Adam(v_params(model), lr=lr, weight_decay=1e-6)
    hist = []
    model.train()
    # E/Q in eval mode for stable BN-free MLP; still fine
    model.E.eval()
    model.q_enc.eval()
    model.answer.eval()
    model.V.train()

    for step in range(1, steps + 1):
        batch = [rng.choice(rows) for _ in range(batch_size)]
        loss = torch.zeros((), device=device)
        acc_n = 0
        flow_v = end_v = task_v = 0.0
        for r in batch:
            q = r["continuation"]["query"]
            y = torch.tensor([r["label"]], dtype=torch.long, device=device)
            loss_f = chain_conjugacy_loss(model, r, device)
            z0 = model.encode_state(r["s0"], device)
            zT = model.flow(z0, r["actions"], T=T)
            with torch.no_grad():
                z_star = model.encode_state(r["state_final"], device)
            loss_end = F.mse_loss(zT, z_star)
            # continue through continuation actions for task
            cont_acts = r["continuation"].get("actions") or []
            z_ans = model.flow(zT, cont_acts, T=len(cont_acts)) if cont_acts else zT
            # also teacher-force cont steps when present
            if cont_acts:
                s_cur = list(r["state_final"])
                z_cur = zT
                for a in cont_acts:

                    class _A:
                        pass

                    aa = _A()
                    aa.op, aa.i, aa.j, aa.k = a["op"], int(a["i"]), int(a["j"]), int(a["k"])
                    s_next = apply_action(s_cur, aa)
                    with torch.no_grad():
                        z_next = model.encode_state(s_next, device)
                    loss_f = loss_f + F.mse_loss(model.one_step(z_cur, a), z_next)
                    z_cur = model.one_step(z_cur, a)
                    s_cur = s_next
                z_ans = z_cur
            logits = model.logits(z_ans, q)
            loss_task = F.cross_entropy(logits, y)
            loss = loss + lam_flow * loss_f + lam_end * loss_end + lam_task * loss_task
            acc_n += int(logits.argmax(-1).item() == r["label"])
            flow_v += float(loss_f.item())
            end_v += float(loss_end.item())
            task_v += float(loss_task.item())
        loss = loss / batch_size
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(v_params(model), 0.5)
        opt.step()
        if step % log_every == 0 or step == 1:
            hist.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "flow": flow_v / batch_size,
                    "end": end_v / batch_size,
                    "task": task_v / batch_size,
                    "acc": acc_n / batch_size,
                }
            )
            print(
                f"[V-only] step={step} L={loss.item():.4f} flow={flow_v/batch_size:.4f} "
                f"end={end_v/batch_size:.4f} task={task_v/batch_size:.4f} acc={acc_n/batch_size:.3f}",
                flush=True,
            )
    return hist


@torch.no_grad()
def oracle_upper_bound(model, rows, n, device):
    """True register dynamics → E → Q (no V). Venue upper bound for chart readout."""
    ok = dis = follow = 0
    rows_d = [r for r in rows if r.get("distinguishing") == 1][:n]
    for r in rows[:n]:
        q = r["continuation"]["query"]
        z = encode_answer(model, r, "final", device)
        ok += int(model.logits(z, q).argmax(-1).item() == r["label"])
    for r in rows_d:
        q = r["continuation"]["query"]
        z = encode_answer(model, r, "final", device)
        zw = encode_answer(model, r, "wrong", device)
        po = model.logits(z, q).argmax(-1).item()
        pw = model.logits(zw, q).argmax(-1).item()
        dis += int(po != pw)
        follow += int(po == r["label"] and pw == r["label_wrong"])
    m = max(len(rows_d), 1)
    return {
        "task_oracle": ok / max(min(n, len(rows)), 1),
        "F_oracle": dis / m,
        "follow_oracle": follow / m,
        "n": len(rows_d),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--init-ckpt", type=Path, required=True, help="Stage-A chart dir")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--lam-flow", type=float, default=1.0)
    ap.add_argument("--lam-end", type=float, default=0.2)
    ap.add_argument("--lam-task", type=float, default=0.2)
    ap.add_argument("--eval-n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    t0 = time.time()

    train = load_jsonl(args.data_dir / "train.jsonl")
    test_id = load_jsonl(args.data_dir / "test_id.jsonl")
    test_ood = load_jsonl(args.data_dir / "test_ood.jsonl")
    need_struct(train)

    print(f"[venue-v] load Stage-A from {args.init_ckpt}", flush=True)
    model = ChartSMIR.load(args.init_ckpt, device=device)
    model.T = args.T
    freeze_eq(model)
    n_v = sum(p.numel() for p in v_params(model))
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[venue-v] trainable V params={n_v} / {n_all}", flush=True)

    print("==== Eval Stage-A init (frozen E,Q) ====", flush=True)
    init_metrics = evaluate(model, test_id, test_ood, args.T, args.eval_n, device)
    oracle = oracle_upper_bound(model, test_id, args.eval_n, device)
    print(json.dumps({"init": init_metrics["intervention"], "oracle": oracle}, indent=2), flush=True)

    print("==== Train V-only (multi-step conjugacy + end) ====", flush=True)
    hist = train_v_only(
        model,
        train,
        args.steps,
        args.lr,
        args.T,
        args.lam_flow,
        args.lam_end,
        args.lam_task,
        args.seed,
        device,
        batch_size=args.batch,
    )

    print("==== Eval after V-only ====", flush=True)
    final = evaluate(model, test_id, test_ood, args.T, args.eval_n, device)
    print(json.dumps(final, indent=2), flush=True)
    model.save(args.out_dir / "v_only")

    I0, I1 = init_metrics["intervention"], final["intervention"]
    gates = {
        "init_F_ge_0_40": I0["branch_disagree"] >= 0.40,
        "final_F_ge_0_50": I1["branch_disagree"] >= 0.50,
        "F_drop_lt_0_10": (I0["branch_disagree"] - I1["branch_disagree"]) <= 0.10,
        "follow_ge_0_40": I1["follow_both"] >= 0.40,
        "on_minus_off_ge_0_15": I1["on_minus_off"] >= 0.15,
        "eta_lt_0_05": final["eta_conjugacy"] < 0.05,
        "tub_rel_lt_0_20": final["tubular_mse_rel"] < 0.20,
        "S_gap_ge_0_10": final["sufficiency_gap"] >= 0.10,
        "adv_lt_0_55": final["offmanifold_adv_acc"] < 0.55,
        "gstar_proxy_gt_0": final["gstar_proxy_tv_gap"] > 0.0,
        "ood_ge_0_70": final["task_ood"] >= 0.70,
        "length_T4_ge_0_65": final["length_curve"][str(args.T + 4)] >= 0.65,
        "eta_improved": final["eta_conjugacy"] < init_metrics["eta_conjugacy"] * 0.5,
        "tub_improved": final["tubular_mse_rel"] < init_metrics["tubular_mse_rel"] * 0.5,
    }
    gates["pass_geometry"] = bool(gates["eta_lt_0_05"] and gates["tub_rel_lt_0_20"])
    gates["pass_faithfulness"] = bool(
        gates["final_F_ge_0_50"]
        and gates["F_drop_lt_0_10"]
        and gates["on_minus_off_ge_0_15"]
    )
    gates["pass_venue_v"] = bool(
        gates["pass_geometry"]
        and gates["pass_faithfulness"]
        and gates["S_gap_ge_0_10"]
        and gates["gstar_proxy_gt_0"]
        and gates["ood_ge_0_70"]
    )

    overall = {
        "meta": {
            "experiment": "SMIR-Venue-V",
            "design": "freeze Stage-A E/Q; train V only; multi-step conjugacy",
            "init_ckpt": str(args.init_ckpt),
            "seconds": round(time.time() - t0, 2),
            "host": os.uname().nodename,
            "device": str(device),
            "steps": args.steps,
            "lr": args.lr,
            "lam_flow": args.lam_flow,
            "lam_end": args.lam_end,
            "lam_task": args.lam_task,
        },
        "oracle_upper_bound": oracle,
        "init": init_metrics,
        "final": final,
        "history_tail": hist[-5:],
        "gates": gates,
        "comparisons": {
            "F_final_minus_init": I1["branch_disagree"] - I0["branch_disagree"],
            "eta_final_over_init": final["eta_conjugacy"]
            / max(init_metrics["eta_conjugacy"], 1e-8),
            "tub_rel_final_over_init": final["tubular_mse_rel"]
            / max(init_metrics["tubular_mse_rel"], 1e-8),
            "ood_final_minus_init": final["task_ood"] - init_metrics["task_ood"],
            "S_gap_final_minus_init": final["sufficiency_gap"]
            - init_metrics["sufficiency_gap"],
            "gap_to_oracle_F": oracle["F_oracle"] - I1["branch_disagree"],
            "gap_to_oracle_task": oracle["task_oracle"] - final["task_state_oracle"],
        },
    }
    (args.out_dir / "overall.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"gates": gates, "comparisons": overall["comparisons"], "oracle": oracle},
            indent=2,
        ),
        flush=True,
    )
    if not gates["pass_venue_v"]:
        print("[WARN] SMIR-Venue-V gates not fully passed", flush=True)
        raise SystemExit(2)
    print("[OK] SMIR-Venue-V passed — geometry+faithfulness certificate", flush=True)


if __name__ == "__main__":
    main()
