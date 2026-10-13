#!/usr/bin/env python3
"""SMIR-Iso-Cert: near-isometric chart + relative conjugacy (fix for Venue-V blow-up).

Fixes: Stage-A CE charts had ||E||~700 and jump MSE explosions.
Here:
  - iso E: (s/40) → Orthogonal Linear → LayerNorm
  - relative conjugacy: ||V(z_t,a)-z_{t+1}||^2 / (||z_t-z_{t+1}||^2 + ε)
  - Stage A: unlock F on iso chart
  - Stage B: freeze Q; train V (+ light E) with relative flow + end + task

Does NOT reuse the bad free-MLP Stage-A / Venue-V run.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model_chart import ChartSMIR, N_REG

sys.path.insert(0, "/project/peilab/why/NEW/ccro/env")
from synlogic_hard import apply_action  # noqa: E402


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


def apply_cont_actions(state: list[float], cont: dict) -> list[float]:
    s = list(state)
    for a in cont.get("actions") or []:

        class _A:
            pass

        aa = _A()
        aa.op, aa.i, aa.j, aa.k = a["op"], int(a["i"]), int(a["j"]), int(a["k"])
        s = apply_action(s, aa)
    return s


def answer_state(row: dict, which: str = "final") -> list[float]:
    base = row["state_final"] if which == "final" else row["state_wrong"]
    return apply_cont_actions(base, row["continuation"])


def encode_answer(model, row, which, device):
    return model.encode_state(answer_state(row, which), device)


def relative_mse(pred, target, base) -> torch.Tensor:
    num = F.mse_loss(pred, target)
    den = F.mse_loss(base.detach(), target) + 1e-3
    return num / den


def chain_rel_conjugacy(model, row, device) -> torch.Tensor:
    states, actions = row["states"], row["actions"]
    loss = torch.zeros((), device=device)
    n = 0
    for t, a in enumerate(actions):
        if t + 1 >= len(states):
            break
        z_t = model.encode_state(states[t], device)
        with torch.no_grad():
            z_tp1 = model.encode_state(states[t + 1], device)
        loss = loss + relative_mse(model.one_step(z_t, a), z_tp1, z_t)
        n += 1
    return loss / max(n, 1)


def forward_task(model, row, T, device):
    q = row["continuation"]["query"]
    z0 = model.encode_state(row["s0"], device)
    zT = model.flow(z0, row["actions"], T=T)
    cont_acts = row["continuation"].get("actions") or []
    z_ans = model.flow(zT, cont_acts, T=len(cont_acts)) if cont_acts else zT
    return zT, z_ans, model.logits(z_ans, q)


def train_stage_a(model, rows, steps, lr, seed, device, batch_size=64, log_every=50):
    rng = random.Random(seed)
    dist = [r for r in rows if r.get("distinguishing") == 1] or rows
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    hist = []
    model.train()
    for step in range(1, steps + 1):
        batch = [rng.choice(dist) for _ in range(batch_size)]
        loss = torch.zeros((), device=device)
        ok = 0
        for r in batch:
            q = r["continuation"]["query"]
            y = torch.tensor([r["label"]], dtype=torch.long, device=device)
            yw = torch.tensor([r["label_wrong"]], dtype=torch.long, device=device)
            z = encode_answer(model, r, "final", device)
            zw = encode_answer(model, r, "wrong", device)
            lo, lw = model.logits(z, q), model.logits(zw, q)
            loss = loss + F.cross_entropy(lo, y) + F.cross_entropy(lw, yw)
            ok += int(lo.argmax(-1).item() == r["label"])
        loss = loss / batch_size
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == 1:
            hist.append({"step": step, "loss": float(loss.item()), "acc": ok / batch_size})
            print(f"[A-Iso] step={step} L={loss.item():.4f} acc={ok/batch_size:.3f}", flush=True)
    return hist


def train_stage_b(model, rows, steps, lr, T, lam_flow, lam_end, lam_task, seed, device, batch_size=32, log_every=50):
    """Freeze Q; train V + light E with relative conjugacy."""
    for p in model.q_enc.parameters():
        p.requires_grad = False
    for p in model.answer.parameters():
        p.requires_grad = False
    params = [p for p in list(model.V.parameters()) + list(model.E.parameters()) if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-6)
    rng = random.Random(seed)
    hist = []
    model.train()
    model.q_enc.eval()
    model.answer.eval()

    for step in range(1, steps + 1):
        batch = [rng.choice(rows) for _ in range(batch_size)]
        loss = torch.zeros((), device=device)
        acc_n = 0
        flow_v = end_v = task_v = 0.0
        for r in batch:
            q = r["continuation"]["query"]
            y = torch.tensor([r["label"]], dtype=torch.long, device=device)
            loss_f = chain_rel_conjugacy(model, r, device)
            z0 = model.encode_state(r["s0"], device)
            zT = model.flow(z0, r["actions"], T=T)
            with torch.no_grad():
                z_star = model.encode_state(r["state_final"], device)
            loss_end = relative_mse(zT, z_star, z0)
            # continuation conjugacy + answer
            cont_acts = r["continuation"].get("actions") or []
            z_cur, s_cur = zT, list(r["state_final"])
            for a in cont_acts:

                class _A:
                    pass

                aa = _A()
                aa.op, aa.i, aa.j, aa.k = a["op"], int(a["i"]), int(a["j"]), int(a["k"])
                s_next = apply_action(s_cur, aa)
                with torch.no_grad():
                    z_next = model.encode_state(s_next, device)
                loss_f = loss_f + relative_mse(model.one_step(z_cur, a), z_next, z_cur)
                z_cur = model.one_step(z_cur, a)
                s_cur = s_next
            z_ans = z_cur
            logits = model.logits(z_ans, q)
            loss_task = F.cross_entropy(logits, y)
            # keep F alive on answer chart
            if r.get("distinguishing") == 1:
                yw = torch.tensor([r["label_wrong"]], dtype=torch.long, device=device)
                zw = encode_answer(model, r, "wrong", device)
                z_on = encode_answer(model, r, "final", device)
                loss_task = loss_task + 0.5 * (
                    F.cross_entropy(model.logits(z_on, q), y)
                    + F.cross_entropy(model.logits(zw, q), yw)
                )
            loss = loss + lam_flow * loss_f + lam_end * loss_end + lam_task * loss_task
            acc_n += int(logits.argmax(-1).item() == r["label"])
            flow_v += float(loss_f.item())
            end_v += float(loss_end.item())
            task_v += float(loss_task.item())
        loss = loss / batch_size
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 0.5)
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
                f"[B-Iso] step={step} L={loss.item():.4f} flow={flow_v/batch_size:.4f} "
                f"end={end_v/batch_size:.4f} task={task_v/batch_size:.4f} acc={acc_n/batch_size:.3f}",
                flush=True,
            )
    for p in model.parameters():
        p.requires_grad = True
    return hist


@torch.no_grad()
def evaluate(model, rows_id, rows_ood, T, n, device):
    def task_acc(rows, T_use, use_flow=True):
        ok = 0
        for r in rows[:n]:
            q = r["continuation"]["query"]
            if use_flow:
                _, _, logits = forward_task(model, r, T_use, device)
                ok += int(logits.argmax(-1).item() == r["label"])
            else:
                z = encode_answer(model, r, "final", device)
                ok += int(model.logits(z, q).argmax(-1).item() == r["label"])
        return ok / max(min(n, len(rows)), 1)

    conj, tub_abs, tub_rel = [], [], []
    norms = []
    for r in rows_id[:n]:
        conj.append(float(chain_rel_conjugacy(model, r, device).item()))
        zT, _, _ = forward_task(model, r, T, device)
        z_star = model.encode_state(r["state_final"], device)
        mse = float(F.mse_loss(zT, z_star).item())
        nE2 = float((z_star * z_star).mean().item()) + 1e-8
        tub_abs.append(mse)
        tub_rel.append(mse / nE2)
        norms.append(float(torch.norm(z_star).item()))

    rows_d = [r for r in rows_id if r.get("distinguishing") == 1][:n]
    disagree = follow = on_ok = off_ok = 0
    for r in rows_d:
        q = r["continuation"]["query"]
        z_on = encode_answer(model, r, "final", device)
        z_w = encode_answer(model, r, "wrong", device)
        p_on = model.logits(z_on, q).argmax(-1).item()
        p_w = model.logits(z_w, q).argmax(-1).item()
        disagree += int(p_on != p_w)
        follow += int(p_on == r["label"] and p_w == r["label_wrong"])
        on_ok += int(p_on == r["label"])
        off_ok += int(model.logits(z_on + 3.0 * torch.randn_like(z_on), q).argmax(-1).item() == r["label"])
    m = max(len(rows_d), 1)

    ok_f = ok_s = 0
    for r in rows_id[:n]:
        q = r["continuation"]["query"]
        _, _, logits = forward_task(model, r, T, device)
        ok_f += int(logits.argmax(-1).item() == r["label"])
        z_noise = torch.randn(1, model.z_dim, device=device)
        zT_n = model.flow(z_noise, r["actions"], T=T)
        cont = r["continuation"].get("actions") or []
        z_ans_n = model.flow(zT_n, cont, T=len(cont)) if cont else zT_n
        ok_s += int(model.logits(z_ans_n, q).argmax(-1).item() == r["label"])
    s_gap = (ok_f - ok_s) / max(n, 1)

    adv_ok = sum(
        int(
            model.logits(8.0 * torch.randn(1, model.z_dim, device=device), r["continuation"]["query"])
            .argmax(-1)
            .item()
            == r["label"]
        )
        for r in rows_id[:n]
    )
    adv_acc = adv_ok / max(n, 1)

    align = []
    for r in rows_d[: min(n, 100)]:
        q = r["continuation"]["query"]
        s = answer_state(r, "final")
        base = F.softmax(model.logits(model.encode_state(s, device), q), dim=-1)
        s2 = list(s)
        s2[random.randrange(N_REG)] += 2.0
        p_sem = F.softmax(model.logits(model.encode_state(s2, device), q), dim=-1)
        z = model.encode_state(s, device)
        z2 = model.encode_state(s2, device)
        delta = torch.norm(z2 - z).item() + 1e-6
        p_junk = F.softmax(
            model.logits(z + (delta / (model.z_dim**0.5)) * torch.randn_like(z), q), dim=-1
        )
        align.append(0.5 * (base - p_sem).abs().sum().item() - 0.5 * (base - p_junk).abs().sum().item())

    l2s = [
        float(torch.norm(encode_answer(model, r, "final", device) - encode_answer(model, r, "wrong", device)).item())
        for r in rows_d[:n]
    ]

    return {
        "task_id": task_acc(rows_id, T),
        "task_ood": task_acc(rows_ood, T),
        "task_state_oracle": task_acc(rows_id, T, use_flow=False),
        "eta_rel_conjugacy": sum(conj) / max(len(conj), 1),
        "tubular_mse_abs": sum(tub_abs) / max(len(tub_abs), 1),
        "tubular_mse_rel": sum(tub_rel) / max(len(tub_rel), 1),
        "chart_norm_mean": sum(norms) / max(len(norms), 1),
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
        "gstar_proxy_tv_gap": sum(align) / max(len(align), 1),
        "chart_l2_on_wrong": sum(l2s) / max(len(l2s), 1),
        "length_curve": {str(t): task_acc(rows_id, t) for t in [T, T + 2, T + 4]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--steps-a", type=int, default=3000)
    ap.add_argument("--steps-b", type=int, default=5000)
    ap.add_argument("--batch-a", type=int, default=64)
    ap.add_argument("--batch-b", type=int, default=32)
    ap.add_argument("--lr-a", type=float, default=3e-3)
    ap.add_argument("--lr-b", type=float, default=1e-4)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--z-dim", type=int, default=64)
    ap.add_argument("--lam-flow", type=float, default=1.0)
    ap.add_argument("--lam-end", type=float, default=0.3)
    ap.add_argument("--lam-task", type=float, default=0.3)
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
    if "continuation" not in train[0]:
        raise RuntimeError("Need SMIR-Shadow-Struct data")

    model = ChartSMIR(z_dim=args.z_dim, T=args.T, iso_chart=True).to(device)
    print(f"[iso-cert] device={device} iso_chart=True train={len(train)}", flush=True)

    results = {}
    print("==== Stage A: iso-chart faithfulness ====", flush=True)
    h_a = train_stage_a(model, train, args.steps_a, args.lr_a, args.seed, device, args.batch_a)
    results["stage_a"] = evaluate(model, test_id, test_ood, args.T, args.eval_n, device)
    results["stage_a"]["history"] = h_a[-5:]
    print(json.dumps(results["stage_a"], indent=2), flush=True)
    model.save(args.out_dir / "stage_a")

    print("==== Stage B: relative conjugacy (freeze Q) ====", flush=True)
    h_b = train_stage_b(
        model,
        train,
        args.steps_b,
        args.lr_b,
        args.T,
        args.lam_flow,
        args.lam_end,
        args.lam_task,
        args.seed + 1,
        device,
        args.batch_b,
    )
    results["stage_b"] = evaluate(model, test_id, test_ood, args.T, args.eval_n, device)
    results["stage_b"]["history"] = h_b[-5:]
    print(json.dumps(results["stage_b"], indent=2), flush=True)
    model.save(args.out_dir / "stage_b")

    A, B = results["stage_a"], results["stage_b"]
    gates = {
        "A_F_ge_0_40": A["intervention"]["branch_disagree"] >= 0.40,
        "B_F_ge_0_50": B["intervention"]["branch_disagree"] >= 0.50,
        "F_preserved": B["intervention"]["branch_disagree"]
        >= A["intervention"]["branch_disagree"] - 0.10,
        "on_minus_off_ge_0_15": B["intervention"]["on_minus_off"] >= 0.15,
        "eta_rel_lt_0_50": B["eta_rel_conjugacy"] < 0.50,
        "tub_rel_lt_0_30": B["tubular_mse_rel"] < 0.30,
        "chart_norm_lt_20": B["chart_norm_mean"] < 20.0,
        "S_gap_ge_0_10": B["sufficiency_gap"] >= 0.10,
        "adv_lt_0_55": B["offmanifold_adv_acc"] < 0.55,
        "gstar_gt_0": B["gstar_proxy_tv_gap"] > 0.0,
        "ood_ge_0_70": B["task_ood"] >= 0.70,
        "eta_improved": B["eta_rel_conjugacy"] < A["eta_rel_conjugacy"] * 0.7,
        "tub_improved": B["tubular_mse_rel"] < A["tubular_mse_rel"] * 0.7,
    }
    gates["pass_iso_cert"] = bool(
        gates["A_F_ge_0_40"]
        and gates["B_F_ge_0_50"]
        and gates["F_preserved"]
        and gates["eta_rel_lt_0_50"]
        and gates["tub_rel_lt_0_30"]
        and gates["chart_norm_lt_20"]
        and gates["ood_ge_0_70"]
    )

    overall = {
        "meta": {
            "experiment": "SMIR-Iso-Cert",
            "design": "near-isometric chart + relative conjugacy; freeze Q in Stage B",
            "seconds": round(time.time() - t0, 2),
            "host": os.uname().nodename,
            "device": str(device),
            "steps_a": args.steps_a,
            "steps_b": args.steps_b,
            "lam_flow": args.lam_flow,
            "lam_end": args.lam_end,
            "lam_task": args.lam_task,
        },
        "results": results,
        "gates": gates,
        "comparisons": {
            "F_B_minus_A": B["intervention"]["branch_disagree"]
            - A["intervention"]["branch_disagree"],
            "eta_B_over_A": B["eta_rel_conjugacy"] / max(A["eta_rel_conjugacy"], 1e-8),
            "tub_B_over_A": B["tubular_mse_rel"] / max(A["tubular_mse_rel"], 1e-8),
            "norm_B": B["chart_norm_mean"],
            "ood_B": B["task_ood"],
        },
    }
    (args.out_dir / "overall.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print(json.dumps({"gates": gates, "comparisons": overall["comparisons"]}, indent=2), flush=True)
    if not gates["pass_iso_cert"]:
        print("[WARN] SMIR-Iso-Cert not fully passed", flush=True)
        raise SystemExit(2)
    print("[OK] SMIR-Iso-Cert passed", flush=True)


if __name__ == "__main__":
    main()
