#!/usr/bin/env python3
"""SMIR-Cert: remaining-goals certification pipeline (structured chart).

Fixes the ill-posed F protocol: Q(z, query) — labels depend on continuation.
Does NOT re-run 7B ContCE/Outcome/SCF/SF text-encoder experiments.

Stages:
  A  Faithfulness warmup: CE(Q(E(s),q),y) + CE(Q(E(s_bad),q),y_bad)
  B  Conjugacy+tubular: L_task + λ_flow L_flow + λ_end ||z_T - sg(E(s*))||^2
  C  Certification suite: F, on-off, η, tubular_rel, S-gap, off-manifold adv,
     g*-proxy metric align, length curve, OOD
  D  (optional) Harder length stress on OOD chains
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

from model_chart import ChartSMIR, N_REG

import sys

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


def need_struct(rows: list[dict]) -> None:
    if not rows or "continuation" not in rows[0] or "state_final" not in rows[0]:
        raise RuntimeError(
            "Dataset missing structured fields (continuation/state_final). "
            "Rebuild with smir/data/build_shadow.py into SMIR-Shadow-Struct."
        )


def apply_cont_actions(state: list[float], cont: dict) -> list[float]:
    """y = query(apply_actions(s, c.actions)) — must apply continuation before readout."""
    s = list(state)
    for a in cont.get("actions") or []:
        # Action is a plain dict in jsonl
        class _A:
            pass

        aa = _A()
        aa.op, aa.i, aa.j, aa.k = a["op"], int(a["i"]), int(a["j"]), int(a["k"])
        s = apply_action(s, aa)
    return s


def answer_state(row: dict, which: str = "final") -> list[float]:
    base = row["state_final"] if which == "final" else row["state_wrong"]
    return apply_cont_actions(base, row["continuation"])


def encode_answer(model, row: dict, which: str, device):
    return model.encode_state(answer_state(row, which), device)


def forward_task(model, row, T, device):
    """s0 -chain-> z_T -cont.actions-> z_ans, then Q(z_ans, q)."""
    q = row["continuation"]["query"]
    z0 = model.encode_state(row["s0"], device)
    zT = model.flow(z0, row["actions"], T=T)
    cont_acts = row["continuation"].get("actions") or []
    z_ans = model.flow(zT, cont_acts, T=len(cont_acts)) if cont_acts else zT
    return zT, z_ans, model.logits(z_ans, q)


def train_stage_a(model, rows, steps, lr, seed, device, batch_size=64, log_every=50):
    """Mini-batch Stage A — single-sample SGD was stuck at ~2*log(2)."""
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
            logits_on = model.logits(z, q)
            logits_w = model.logits(zw, q)
            loss = loss + F.cross_entropy(logits_on, y) + F.cross_entropy(logits_w, yw)
            ok += int(logits_on.argmax(-1).item() == r["label"])
        loss = loss / batch_size
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == 1:
            acc = ok / batch_size
            hist.append({"step": step, "loss": float(loss.item()), "acc": acc})
            print(f"[A-Faithful] step={step} L={loss.item():.4f} acc={acc:.3f}", flush=True)
    return hist


def flow_loss(model, r, device):
    z_t = model.encode_state(r["state_t"], device)
    z_tp1 = model.encode_state(r["state_tp1"], device).detach()
    return F.mse_loss(model.one_step(z_t, r["step_action"]), z_tp1)


def train_stage_b(model, rows, steps, lr, T, lam_flow, lam_end, seed, device, batch_size=32, log_every=50):
    rng = random.Random(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    hist = []
    model.train()
    for step in range(1, steps + 1):
        batch = [rng.choice(rows) for _ in range(batch_size)]
        loss = torch.zeros((), device=device)
        acc_n = 0
        task_v = flow_v = end_v = 0.0
        for r in batch:
            q = r["continuation"]["query"]
            y = torch.tensor([r["label"]], dtype=torch.long, device=device)
            zT, z_ans, logits = forward_task(model, r, T, device)
            loss_task = F.cross_entropy(logits, y)
            loss_f = flow_loss(model, r, device)
            with torch.no_grad():
                z_star = model.encode_state(r["state_final"], device)
                z_ans_star = encode_answer(model, r, "final", device)
            loss_end = F.mse_loss(zT, z_star) + F.mse_loss(z_ans, z_ans_star)
            if r.get("distinguishing") == 1:
                yw = torch.tensor([r["label_wrong"]], dtype=torch.long, device=device)
                zw = encode_answer(model, r, "wrong", device)
                loss_fth = F.cross_entropy(model.logits(z_ans_star, q), y) + F.cross_entropy(
                    model.logits(zw, q), yw
                )
            else:
                loss_fth = torch.zeros((), device=device)
            loss = loss + loss_task + lam_flow * loss_f + lam_end * loss_end + 0.5 * loss_fth
            acc_n += int(logits.argmax(-1).item() == r["label"])
            task_v += float(loss_task.item())
            flow_v += float(loss_f.item())
            end_v += float(loss_end.item())
        loss = loss / batch_size
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == 1:
            hist.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "task": task_v / batch_size,
                    "flow": flow_v / batch_size,
                    "end": end_v / batch_size,
                    "acc": acc_n / batch_size,
                }
            )
            print(
                f"[B-Flow] step={step} L={loss.item():.4f} task={task_v/batch_size:.4f} "
                f"flow={flow_v/batch_size:.4f} end={end_v/batch_size:.4f} acc={acc_n/batch_size:.3f}",
                flush=True,
            )
    return hist


@torch.no_grad()
def evaluate(model, rows_id, rows_ood, T, n, device):
    def task_acc(rows, T_use, use_flow=True):
        ok = 0
        rows = rows[:n]
        for r in rows:
            q = r["continuation"]["query"]
            if use_flow:
                _, _, logits = forward_task(model, r, T_use, device)
                ok += int(logits.argmax(-1).item() == r["label"])
            else:
                z = encode_answer(model, r, "final", device)
                ok += int(model.logits(z, q).argmax(-1).item() == r["label"])
        return ok / max(len(rows), 1)

    conj, tub_abs, tub_rel = [], [], []
    for r in rows_id[:n]:
        conj.append(float(flow_loss(model, r, device).item()))
        zT, _, _ = forward_task(model, r, T, device)
        z_star = model.encode_state(r["state_final"], device)
        mse = float(F.mse_loss(zT, z_star).item())
        nE2 = float((z_star * z_star).mean().item()) + 1e-8
        tub_abs.append(mse)
        tub_rel.append(mse / nE2)

    # F: same continuation, on vs wrong answer-states
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
        z_off = z_on + 3.0 * torch.randn_like(z_on)
        off_ok += int(model.logits(z_off, q).argmax(-1).item() == r["label"])
    m = max(len(rows_d), 1)

    ok_f = ok_s = 0
    for r in rows_id[:n]:
        q = r["continuation"]["query"]
        z0 = model.encode_state(r["s0"], device)
        _, z_ans, logits = forward_task(model, r, T, device)
        ok_f += int(logits.argmax(-1).item() == r["label"])
        z_noise = torch.randn_like(z0)
        zT_n = model.flow(z_noise, r["actions"], T=T)
        cont_acts = r["continuation"].get("actions") or []
        z_ans_n = model.flow(zT_n, cont_acts, T=len(cont_acts)) if cont_acts else zT_n
        ok_s += int(model.logits(z_ans_n, q).argmax(-1).item() == r["label"])
    s_gap = (ok_f - ok_s) / max(n, 1)

    adv_ok = 0
    for r in rows_id[:n]:
        q = r["continuation"]["query"]
        z_far = 8.0 * torch.randn(1, model.z_dim, device=device)
        adv_ok += int(model.logits(z_far, q).argmax(-1).item() == r["label"])
    adv_acc = adv_ok / max(n, 1)

    align_scores = []
    for r in rows_d[: min(n, 100)]:
        q = r["continuation"]["query"]
        s = answer_state(r, "final")
        base = F.softmax(model.logits(model.encode_state(s, device), q), dim=-1)
        s2 = list(s)
        idx = random.randrange(N_REG)
        s2[idx] += 2.0
        p_sem = F.softmax(model.logits(model.encode_state(s2, device), q), dim=-1)
        tv_sem = 0.5 * (base - p_sem).abs().sum().item()
        z = model.encode_state(s, device)
        z2 = model.encode_state(s2, device)
        delta = torch.norm(z2 - z).item() + 1e-6
        z_junk = z + (delta / (model.z_dim**0.5)) * torch.randn_like(z)
        p_junk = F.softmax(model.logits(z_junk, q), dim=-1)
        tv_junk = 0.5 * (base - p_junk).abs().sum().item()
        align_scores.append(tv_sem - tv_junk)
    g_proxy = sum(align_scores) / max(len(align_scores), 1)

    length_curve = {str(t): task_acc(rows_id, t) for t in [T, T + 2, T + 4]}

    l2s = []
    for r in rows_d[:n]:
        z_on = encode_answer(model, r, "final", device)
        z_w = encode_answer(model, r, "wrong", device)
        l2s.append(float(torch.norm(z_on - z_w).item()))

    return {
        "task_id": task_acc(rows_id, T),
        "task_ood": task_acc(rows_ood, T),
        "task_state_oracle": task_acc(rows_id, T, use_flow=False),
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
        "gstar_proxy_tv_gap": g_proxy,
        "chart_l2_on_wrong": sum(l2s) / max(len(l2s), 1),
        "length_curve": length_curve,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--steps-a", type=int, default=3000)
    ap.add_argument("--steps-b", type=int, default=4000)
    ap.add_argument("--batch-a", type=int, default=64)
    ap.add_argument("--batch-b", type=int, default=32)
    ap.add_argument("--lr-a", type=float, default=3e-3)
    ap.add_argument("--lr-b", type=float, default=1e-3)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--z-dim", type=int, default=64)
    ap.add_argument("--lam-flow", type=float, default=1.0)
    ap.add_argument("--lam-end", type=float, default=0.5)
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
    print(
        f"[cert] device={device} train={len(train)} T={args.T} z_dim={args.z_dim} "
        f"lam_end={args.lam_end}",
        flush=True,
    )

    model = ChartSMIR(z_dim=args.z_dim, T=args.T).to(device)
    results = {}

    # ---- Stage A ----
    print("==== Stage A: faithfulness warmup (Q(E(s),q)) ====", flush=True)
    h_a = train_stage_a(
        model, train, args.steps_a, args.lr_a, args.seed, device, batch_size=args.batch_a
    )
    print("==== Eval after A ====", flush=True)
    results["stage_a"] = evaluate(model, test_id, test_ood, args.T, args.eval_n, device)
    results["stage_a"]["history"] = h_a[-5:]
    print(json.dumps(results["stage_a"], indent=2), flush=True)
    model.save(args.out_dir / "stage_a")
    f_a = results["stage_a"]["intervention"]["branch_disagree"]
    if f_a < 0.25:
        print(
            f"[WARN] Stage A F={f_a:.3f}<0.25; continue Stage B anyway (soft gate)",
            flush=True,
        )

    # ---- Stage B ----
    print("==== Stage B: conjugacy + endpoint tubular ====", flush=True)
    h_b = train_stage_b(
        model,
        train,
        args.steps_b,
        args.lr_b,
        args.T,
        args.lam_flow,
        args.lam_end,
        args.seed + 1,
        device,
        batch_size=args.batch_b,
    )
    print("==== Eval after B (cert) ====", flush=True)
    results["stage_b"] = evaluate(model, test_id, test_ood, args.T, args.eval_n, device)
    results["stage_b"]["history"] = h_b[-5:]
    print(json.dumps(results["stage_b"], indent=2), flush=True)
    model.save(args.out_dir / "stage_b")

    A, B = results["stage_a"], results["stage_b"]
    gates = {
        # remaining-goal pack
        "F_ge_0_40": B["intervention"]["branch_disagree"] >= 0.40,
        "follow_ge_0_30": B["intervention"]["follow_both"] >= 0.30,
        "on_minus_off_ge_0_15": B["intervention"]["on_minus_off"] >= 0.15,
        "eta_lt_0_05": B["eta_conjugacy"] < 0.05,
        "tub_rel_lt_1": B["tubular_mse_rel"] < 1.0,
        "S_gap_ge_0_10": B["sufficiency_gap"] >= 0.10,
        "adv_lt_0_55": B["offmanifold_adv_acc"] < 0.55,
        "gstar_proxy_gt_0": B["gstar_proxy_tv_gap"] > 0.0,
        "ood_ge_0_75": B["task_ood"] >= 0.75,
        "length_T4_ge_0_70": B["length_curve"][str(args.T + 4)] >= 0.70,
        "chart_separated": B["chart_l2_on_wrong"] >= 0.3,
        "stage_a_unlocked_F": A["intervention"]["branch_disagree"] >= 0.25,
        "flow_preserves_F": B["intervention"]["branch_disagree"]
        >= A["intervention"]["branch_disagree"] - 0.05,
    }
    gates["pass_faithfulness"] = bool(
        gates["F_ge_0_40"] and gates["follow_ge_0_30"] and gates["on_minus_off_ge_0_15"]
    )
    gates["pass_geometry"] = bool(gates["eta_lt_0_05"] and gates["tub_rel_lt_1"])
    gates["pass_certificate"] = bool(
        gates["pass_faithfulness"]
        and gates["pass_geometry"]
        and gates["S_gap_ge_0_10"]
        and gates["adv_lt_0_55"]
        and gates["gstar_proxy_gt_0"]
        and gates["ood_ge_0_75"]
    )
    gates["pass_cert"] = bool(gates["pass_certificate"] and gates["stage_a_unlocked_F"])

    overall = {
        "meta": {
            "experiment": "SMIR-Cert",
            "doc": "SMIR.md remaining goals: F, tubular, S, g*-proxy, off-manifold, OOD",
            "design": "structured chart E(s)+Q(z,q); no 7B text encoder",
            "seconds": round(time.time() - t0, 2),
            "host": os.uname().nodename,
            "device": str(device),
            "steps_a": args.steps_a,
            "steps_b": args.steps_b,
            "T": args.T,
            "z_dim": args.z_dim,
            "lam_flow": args.lam_flow,
            "lam_end": args.lam_end,
        },
        "results": results,
        "gates": gates,
        "comparisons": {
            "F_B_minus_A": B["intervention"]["branch_disagree"]
            - A["intervention"]["branch_disagree"],
            "eta_B": B["eta_conjugacy"],
            "tub_rel_B": B["tubular_mse_rel"],
            "ood_B": B["task_ood"],
            "gstar_proxy_B": B["gstar_proxy_tv_gap"],
        },
    }
    (args.out_dir / "overall.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print(json.dumps({"gates": gates, "comparisons": overall["comparisons"]}, indent=2), flush=True)
    if not gates["pass_cert"]:
        print("[WARN] SMIR-Cert gates not fully passed", flush=True)
        raise SystemExit(2)
    print("[OK] SMIR-Cert remaining goals passed", flush=True)


if __name__ == "__main__":
    main()
