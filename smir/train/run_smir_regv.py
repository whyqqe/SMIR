#!/usr/bin/env python3
"""SMIR-RegV: highest-success conjugacy experiment.

Detailed rationale:
  Prior residual V on linear iso charts plateaus (η_abs~0.25) because mul
  dynamics are hard in an unstructured ambient MLP. Shadow domain gives the
  true V*=f_a on registers — use that inductive bias.

Design (max success):
  E(s)=s/40 ∈ ℝ^5          identity chart (fixed)
  V = structured add/sub/mul on coords (+ tiny residual)
  Stage A: train Q (and residual lightly) for F
  Stage B: freeze Q; fit residual with absolute teacher force to E(f_a(s))
           (skeleton already near-exact → η should collapse)

Gates target full certificate: F + η≈0 + tubular≈0 + OOD + S-gap.
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

from model_register import N_REG, RegisterSMIR

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


def chain_abs(model, row, device):
    states, actions = row["states"], row["actions"]
    loss = torch.zeros((), device=device)
    n = 0
    for t, a in enumerate(actions):
        if t + 1 >= len(states):
            break
        z_t = model.encode_state(states[t], device)
        z_tp1 = model.encode_state(states[t + 1], device)
        loss = loss + F.mse_loss(model.one_step(z_t, a), z_tp1)
        n += 1
    return loss / max(n, 1)


def train_stage_a(model, rows, steps, lr, seed, device, batch_size=64, log_every=50):
    rng = random.Random(seed)
    dist = [r for r in rows if r.get("distinguishing") == 1] or rows
    # Train Q + residual (V skeleton fixed ops)
    params = list(model.q_enc.parameters()) + list(model.answer.parameters())
    if model.V.res is not None:
        params += list(model.V.res.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
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
            # also keep residual small vs exact one-step
            loss = loss + 0.1 * chain_abs(model, r, device)
            ok += int(lo.argmax(-1).item() == r["label"])
        loss = loss / batch_size
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == 1:
            hist.append({"step": step, "loss": float(loss.item()), "acc": ok / batch_size})
            print(f"[RegV-A] step={step} L={loss.item():.4f} acc={ok/batch_size:.3f}", flush=True)
    return hist


def train_stage_b(model, rows, steps, lr, T, lam_flow, lam_end, lam_task, seed, device, batch_size=64, log_every=50):
    # Freeze Q; only residual (if any)
    for p in model.q_enc.parameters():
        p.requires_grad = False
    for p in model.answer.parameters():
        p.requires_grad = False
    if model.V.res is None:
        print("[RegV-B] no residual — skip fit; skeleton is exact", flush=True)
        return []
    opt = torch.optim.Adam(model.V.res.parameters(), lr=lr, weight_decay=1e-6)
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
            loss_f = chain_abs(model, r, device)
            z0 = model.encode_state(r["s0"], device)
            zT = model.flow(z0, r["actions"], T=T)
            z_star = model.encode_state(r["state_final"], device)
            loss_end = F.mse_loss(zT, z_star)
            cont = r["continuation"].get("actions") or []
            z_ans = model.flow(zT, cont, T=len(cont)) if cont else zT
            # cont conjugacy
            if cont:
                s_cur = list(r["state_final"])
                z_cur = zT
                for a in cont:

                    class _A:
                        pass

                    aa = _A()
                    aa.op, aa.i, aa.j, aa.k = a["op"], int(a["i"]), int(a["j"]), int(a["k"])
                    s_next = apply_action(s_cur, aa)
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
                f"[RegV-B] step={step} L={loss.item():.4f} flow={flow_v/batch_size:.4f} "
                f"end={end_v/batch_size:.4f} task={task_v/batch_size:.4f} acc={acc_n/batch_size:.3f}",
                flush=True,
            )
    for p in model.parameters():
        p.requires_grad = True
    return hist


@torch.no_grad()
def evaluate(model, rows_id, rows_ood, T, n, device):
    def forward_task(row, T_use):
        q = row["continuation"]["query"]
        z0 = model.encode_state(row["s0"], device)
        zT = model.flow(z0, row["actions"], T=T_use)
        cont = row["continuation"].get("actions") or []
        z_ans = model.flow(zT, cont, T=len(cont)) if cont else zT
        return zT, z_ans, model.logits(z_ans, q)

    def task_acc(rows, T_use, use_flow=True):
        ok = 0
        for r in rows[:n]:
            q = r["continuation"]["query"]
            if use_flow:
                _, _, logits = forward_task(r, T_use)
                ok += int(logits.argmax(-1).item() == r["label"])
            else:
                ok += int(model.logits(encode_answer(model, r, "final", device), q).argmax(-1).item() == r["label"])
        return ok / max(min(n, len(rows)), 1)

    conj_abs, tub_abs, tub_rel, norms = [], [], [], []
    for r in rows_id[:n]:
        conj_abs.append(float(chain_abs(model, r, device).item()))
        zT, _, _ = forward_task(r, T)
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
        off_ok += int(model.logits(z_on + 0.5 * torch.randn_like(z_on), q).argmax(-1).item() == r["label"])
    m = max(len(rows_d), 1)

    ok_f = ok_s = 0
    for r in rows_id[:n]:
        q = r["continuation"]["query"]
        _, _, logits = forward_task(r, T)
        ok_f += int(logits.argmax(-1).item() == r["label"])
        z_n = torch.randn(1, model.z_dim, device=device)
        zT_n = model.flow(z_n, r["actions"], T=T)
        cont = r["continuation"].get("actions") or []
        z_ans_n = model.flow(zT_n, cont, T=len(cont)) if cont else zT_n
        ok_s += int(model.logits(z_ans_n, q).argmax(-1).item() == r["label"])
    s_gap = (ok_f - ok_s) / max(n, 1)

    adv = sum(
        int(
            model.logits(3.0 * torch.randn(1, model.z_dim, device=device), r["continuation"]["query"])
            .argmax(-1)
            .item()
            == r["label"]
        )
        for r in rows_id[:n]
    ) / max(n, 1)

    align = []
    for r in rows_d[: min(n, 100)]:
        q = r["continuation"]["query"]
        s = answer_state(r, "final")
        base = F.softmax(model.logits(model.encode_state(s, device), q), dim=-1)
        s2 = list(s)
        s2[random.randrange(N_REG)] += 2.0
        p_sem = F.softmax(model.logits(model.encode_state(s2, device), q), dim=-1)
        z = model.encode_state(s, device)
        delta = torch.norm(model.encode_state(s2, device) - z).item() + 1e-6
        p_junk = F.softmax(
            model.logits(z + (delta / (model.z_dim**0.5)) * torch.randn_like(z), q), dim=-1
        )
        align.append(
            0.5 * (base - p_sem).abs().sum().item() - 0.5 * (base - p_junk).abs().sum().item()
        )

    return {
        "task_id": task_acc(rows_id, T),
        "task_ood": task_acc(rows_ood, T),
        "task_state_oracle": task_acc(rows_id, T, use_flow=False),
        "eta_abs": sum(conj_abs) / max(len(conj_abs), 1),
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
        "offmanifold_adv_acc": adv,
        "gstar_proxy_tv_gap": sum(align) / max(len(align), 1),
        "length_curve": {str(t): task_acc(rows_id, t) for t in [T, T + 2, T + 4]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--steps-a", type=int, default=4000)
    ap.add_argument("--steps-b", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr-a", type=float, default=3e-3)
    ap.add_argument("--lr-b", type=float, default=1e-3)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--lam-flow", type=float, default=1.0)
    ap.add_argument("--lam-end", type=float, default=1.0)
    ap.add_argument("--lam-task", type=float, default=0.2)
    ap.add_argument("--no-residual", action="store_true", help="pure hard V (max conjugacy)")
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

    residual = not args.no_residual
    model = RegisterSMIR(T=args.T, residual=residual).to(device)
    print(f"[RegV] device={device} residual={residual} train={len(train)}", flush=True)

    results = {}
    print("==== Stage A: Q on identity register chart ====", flush=True)
    h_a = train_stage_a(model, train, args.steps_a, args.lr_a, args.seed, device, args.batch)
    results["stage_a"] = evaluate(model, test_id, test_ood, args.T, args.eval_n, device)
    results["stage_a"]["history"] = h_a[-5:]
    print(json.dumps(results["stage_a"], indent=2), flush=True)
    model.save(args.out_dir / "stage_a")

    print("==== Stage B: fit residual / certify structured V ====", flush=True)
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
        args.batch,
    )
    results["stage_b"] = evaluate(model, test_id, test_ood, args.T, args.eval_n, device)
    results["stage_b"]["history"] = h_b[-5:] if h_b else []
    print(json.dumps(results["stage_b"], indent=2), flush=True)
    model.save(args.out_dir / "stage_b")

    B = results["stage_b"]
    A = results["stage_a"]
    gates = {
        "F_ge_0_50": B["intervention"]["branch_disagree"] >= 0.50,
        "follow_ge_0_45": B["intervention"]["follow_both"] >= 0.45,
        "on_minus_off_ge_0_15": B["intervention"]["on_minus_off"] >= 0.15,
        "eta_abs_lt_0_02": B["eta_abs"] < 0.02,
        "tub_abs_lt_0_05": B["tubular_mse_abs"] < 0.05,
        "S_gap_ge_0_15": B["sufficiency_gap"] >= 0.15,
        "adv_lt_0_55": B["offmanifold_adv_acc"] < 0.55,
        "gstar_gt_0": B["gstar_proxy_tv_gap"] > 0.0,
        "ood_ge_0_80": B["task_ood"] >= 0.80,
        "oracle_ge_0_85": B["task_state_oracle"] >= 0.85,
        "F_preserved": B["intervention"]["branch_disagree"]
        >= A["intervention"]["branch_disagree"] - 0.05,
    }
    gates["pass_regv"] = bool(
        gates["F_ge_0_50"]
        and gates["eta_abs_lt_0_02"]
        and gates["tub_abs_lt_0_05"]
        and gates["ood_ge_0_80"]
        and gates["on_minus_off_ge_0_15"]
    )

    overall = {
        "meta": {
            "experiment": "SMIR-RegV",
            "design": "identity R^5 chart + structured register V (+optional residual)",
            "residual": residual,
            "seconds": round(time.time() - t0, 2),
            "host": os.uname().nodename,
            "device": str(device),
            "steps_a": args.steps_a,
            "steps_b": args.steps_b,
        },
        "results": results,
        "gates": gates,
        "comparisons": {
            "eta_B": B["eta_abs"],
            "tub_B": B["tubular_mse_abs"],
            "F_B": B["intervention"]["branch_disagree"],
            "ood_B": B["task_ood"],
            "S_gap_B": B["sufficiency_gap"],
        },
    }
    (args.out_dir / "overall.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print(json.dumps({"gates": gates, "comparisons": overall["comparisons"]}, indent=2), flush=True)
    if not gates["pass_regv"]:
        print("[WARN] SMIR-RegV not fully passed", flush=True)
        raise SystemExit(2)
    print("[OK] SMIR-RegV passed — F + near-exact conjugacy", flush=True)


if __name__ == "__main__":
    main()
