#!/usr/bin/env python3
"""SMIR-V*: fill the biggest gap — learn conjugacy with strong teacher, freeze E/Q.

Design (venue-critical Stage-B fix):
  1. Load near-isometric Stage-A (E,Q already unlock F)
  2. Freeze E and Q completely
  3. Train ONLY V with ABSOLUTE multi-step teacher force:
       Σ_t ||V(E(s_t), a_t) - sg(E(s_{t+1}))||^2
     using known register dynamics s_{t+1}=f_a(s_t)  (shadow-domain advantage)
  4. Endpoint ||z_T - sg(E(s*))||^2 + light task through flow
  5. Same certification suite as Iso-Cert

Does NOT: retrain 7B, relative conjugacy, unfreeze E, joint Stage-B blow-up.
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


def freeze_eq(model: ChartSMIR) -> None:
    for p in model.E.parameters():
        p.requires_grad = False
    for p in model.q_enc.parameters():
        p.requires_grad = False
    for p in model.answer.parameters():
        p.requires_grad = False
    for p in model.V.parameters():
        p.requires_grad = True


def v_params(model):
    return [p for p in model.V.parameters() if p.requires_grad]


def chain_abs_conjugacy(model, row, device) -> torch.Tensor:
    """Absolute teacher-force: known f_a on registers → E targets."""
    states, actions = row["states"], row["actions"]
    loss = torch.zeros((), device=device)
    n = 0
    for t, a in enumerate(actions):
        if t + 1 >= len(states):
            break
        with torch.no_grad():
            z_t = model.encode_state(states[t], device)
            z_tp1 = model.encode_state(states[t + 1], device)
        # z_t must allow grad through V only — detach chart input
        pred = model.one_step(z_t.detach(), a)
        loss = loss + F.mse_loss(pred, z_tp1)
        n += 1
    return loss / max(n, 1)


def cont_abs_conjugacy(model, row, zT, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher-force continuation actions; return (loss, z_ans)."""
    cont_acts = row["continuation"].get("actions") or []
    if not cont_acts:
        return torch.zeros((), device=device), zT
    loss = torch.zeros((), device=device)
    z_cur = zT
    s_cur = list(row["state_final"])
    for a in cont_acts:

        class _A:
            pass

        aa = _A()
        aa.op, aa.i, aa.j, aa.k = a["op"], int(a["i"]), int(a["j"]), int(a["k"])
        s_next = apply_action(s_cur, aa)
        with torch.no_grad():
            z_next = model.encode_state(s_next, device)
        pred = model.one_step(z_cur, a)
        loss = loss + F.mse_loss(pred, z_next)
        z_cur = pred  # differentiable rollout for task
        s_cur = s_next
    return loss / max(len(cont_acts), 1), z_cur


def train_vstar(
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
    batch_size=64,
    log_every=50,
):
    rng = random.Random(seed)
    opt = torch.optim.Adam(v_params(model), lr=lr, weight_decay=1e-6)
    hist = []
    model.train()
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
            loss_f = chain_abs_conjugacy(model, r, device)
            with torch.no_grad():
                z0 = model.encode_state(r["s0"], device)
                z_star = model.encode_state(r["state_final"], device)
            zT = model.flow(z0.detach(), r["actions"], T=T)
            loss_end = F.mse_loss(zT, z_star)
            loss_c, z_ans = cont_abs_conjugacy(model, r, zT, device)
            loss_f = loss_f + loss_c
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
                f"[V*] step={step} L={loss.item():.4f} flow={flow_v/batch_size:.4f} "
                f"end={end_v/batch_size:.4f} task={task_v/batch_size:.4f} acc={acc_n/batch_size:.3f}",
                flush=True,
            )
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
                z = encode_answer(model, r, "final", device)
                ok += int(model.logits(z, q).argmax(-1).item() == r["label"])
        return ok / max(min(n, len(rows)), 1)

    # absolute conjugacy (reportable η)
    conj_abs, conj_rel, tub_abs, tub_rel, norms = [], [], [], [], []
    for r in rows_id[:n]:
        states, actions = r["states"], r["actions"]
        abs_l = rel_l = 0.0
        cnt = 0
        for t, a in enumerate(actions):
            if t + 1 >= len(states):
                break
            z_t = model.encode_state(states[t], device)
            z_tp1 = model.encode_state(states[t + 1], device)
            pred = model.one_step(z_t, a)
            mse = float(F.mse_loss(pred, z_tp1).item())
            den = float(F.mse_loss(z_t, z_tp1).item()) + 1e-3
            abs_l += mse
            rel_l += mse / den
            cnt += 1
        conj_abs.append(abs_l / max(cnt, 1))
        conj_rel.append(rel_l / max(cnt, 1))
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
        off_ok += int(
            model.logits(z_on + 3.0 * torch.randn_like(z_on), q).argmax(-1).item() == r["label"]
        )
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
            model.logits(8.0 * torch.randn(1, model.z_dim, device=device), r["continuation"]["query"])
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
        z2 = model.encode_state(s2, device)
        delta = torch.norm(z2 - z).item() + 1e-6
        p_junk = F.softmax(
            model.logits(z + (delta / (model.z_dim**0.5)) * torch.randn_like(z), q), dim=-1
        )
        align.append(
            0.5 * (base - p_sem).abs().sum().item() - 0.5 * (base - p_junk).abs().sum().item()
        )

    l2s = [
        float(
            torch.norm(
                encode_answer(model, r, "final", device) - encode_answer(model, r, "wrong", device)
            ).item()
        )
        for r in rows_d[:n]
    ]

    return {
        "task_id": task_acc(rows_id, T),
        "task_ood": task_acc(rows_ood, T),
        "task_state_oracle": task_acc(rows_id, T, use_flow=False),
        "eta_abs": sum(conj_abs) / max(len(conj_abs), 1),
        "eta_rel": sum(conj_rel) / max(len(conj_rel), 1),
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
        "chart_l2_on_wrong": sum(l2s) / max(len(l2s), 1),
        "length_curve": {str(t): task_acc(rows_id, t) for t in [T, T + 2, T + 4]},
    }


def maybe_train_stage_a(model, train, steps, lr, device, batch_size, seed):
    """Only if init ckpt missing — quick F unlock on iso chart."""
    rng = random.Random(seed)
    dist = [r for r in train if r.get("distinguishing") == 1] or train
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
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
        opt.step()
        if step % 50 == 0 or step == 1:
            print(f"[A-fallback] step={step} L={loss.item():.4f} acc={ok/batch_size:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--init-ckpt", type=Path, default=None, help="Iso Stage-A dir (preferred)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--z-dim", type=int, default=64)
    ap.add_argument("--lam-flow", type=float, default=1.0)
    ap.add_argument("--lam-end", type=float, default=0.5)
    ap.add_argument("--lam-task", type=float, default=0.15)
    ap.add_argument("--eval-n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fallback-steps-a", type=int, default=2500)
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
        raise RuntimeError("Need SMIR-Shadow-Struct")

    if args.init_ckpt and (args.init_ckpt / "chart.pt").exists():
        print(f"[V*] load frozen chart from {args.init_ckpt}", flush=True)
        model = ChartSMIR.load(args.init_ckpt, device=device)
        if not model.iso_chart:
            print("[WARN] init ckpt is not iso_chart; continuing anyway", flush=True)
    else:
        print("[V*] no Stage-A ckpt — fallback iso Stage-A then V*", flush=True)
        model = ChartSMIR(z_dim=args.z_dim, T=args.T, iso_chart=True).to(device)
        maybe_train_stage_a(
            model, train, args.fallback_steps_a, 3e-3, device, 64, args.seed
        )
        model.save(args.out_dir / "stage_a_fallback")

    model.T = args.T
    freeze_eq(model)
    print(
        f"[V*] trainable V={sum(p.numel() for p in v_params(model))} "
        f"iso={model.iso_chart}",
        flush=True,
    )

    print("==== Eval init (frozen E,Q) ====", flush=True)
    init = evaluate(model, test_id, test_ood, args.T, args.eval_n, device)
    print(json.dumps({"intervention": init["intervention"], "eta_abs": init["eta_abs"], "tub_rel": init["tubular_mse_rel"]}, indent=2), flush=True)

    print("==== Train V* absolute teacher-force ====", flush=True)
    hist = train_vstar(
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

    print("==== Eval final ====", flush=True)
    final = evaluate(model, test_id, test_ood, args.T, args.eval_n, device)
    print(json.dumps(final, indent=2), flush=True)
    model.save(args.out_dir / "vstar")

    I0, I1 = init["intervention"], final["intervention"]
    gates = {
        "init_F_ge_0_40": I0["branch_disagree"] >= 0.40,
        "final_F_ge_0_50": I1["branch_disagree"] >= 0.50,
        "F_drop_lt_0_10": (I0["branch_disagree"] - I1["branch_disagree"]) <= 0.10,
        "on_minus_off_ge_0_15": I1["on_minus_off"] >= 0.15,
        "eta_abs_lt_0_05": final["eta_abs"] < 0.05,
        "eta_rel_lt_0_50": final["eta_rel"] < 0.50,
        "tub_rel_lt_0_20": final["tubular_mse_rel"] < 0.20,
        "tub_abs_lt_0_50": final["tubular_mse_abs"] < 0.50,
        "S_gap_ge_0_10": final["sufficiency_gap"] >= 0.10,
        "adv_lt_0_55": final["offmanifold_adv_acc"] < 0.55,
        "gstar_gt_0": final["gstar_proxy_tv_gap"] > 0.0,
        "ood_ge_0_70": final["task_ood"] >= 0.70,
        "eta_improved": final["eta_abs"] < init["eta_abs"] * 0.3,
        "tub_improved": final["tubular_mse_abs"] < init["tubular_mse_abs"] * 0.3,
        "chart_norm_lt_20": final["chart_norm_mean"] < 20.0,
    }
    gates["pass_geometry"] = bool(
        gates["eta_abs_lt_0_05"] and gates["tub_rel_lt_0_20"] and gates["tub_abs_lt_0_50"]
    )
    gates["pass_faithfulness"] = bool(
        gates["final_F_ge_0_50"] and gates["F_drop_lt_0_10"] and gates["on_minus_off_ge_0_15"]
    )
    gates["pass_vstar"] = bool(
        gates["pass_geometry"]
        and gates["pass_faithfulness"]
        and gates["S_gap_ge_0_10"]
        and gates["ood_ge_0_70"]
        and gates["chart_norm_lt_20"]
    )

    overall = {
        "meta": {
            "experiment": "SMIR-V*",
            "design": "freeze iso E/Q; absolute teacher-force V on known f_a",
            "init_ckpt": str(args.init_ckpt) if args.init_ckpt else None,
            "seconds": round(time.time() - t0, 2),
            "host": os.uname().nodename,
            "device": str(device),
            "steps": args.steps,
            "lr": args.lr,
            "lam_flow": args.lam_flow,
            "lam_end": args.lam_end,
            "lam_task": args.lam_task,
        },
        "init": init,
        "final": final,
        "history_tail": hist[-5:],
        "gates": gates,
        "comparisons": {
            "F_final_minus_init": I1["branch_disagree"] - I0["branch_disagree"],
            "eta_abs_final_over_init": final["eta_abs"] / max(init["eta_abs"], 1e-8),
            "tub_abs_final_over_init": final["tubular_mse_abs"] / max(init["tubular_mse_abs"], 1e-8),
            "ood_final": final["task_ood"],
            "S_gap_final": final["sufficiency_gap"],
        },
    }
    (args.out_dir / "overall.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print(json.dumps({"gates": gates, "comparisons": overall["comparisons"]}, indent=2), flush=True)
    if not gates["pass_vstar"]:
        print("[WARN] SMIR-V* gates not fully passed", flush=True)
        raise SystemExit(2)
    print("[OK] SMIR-V* passed — conjugacy gap closed with F preserved", flush=True)


if __name__ == "__main__":
    main()
