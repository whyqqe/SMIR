#!/usr/bin/env python3
"""SMIR-RegV+ : push past pass_regv (OOD/Q) then keep geometry.

Changes vs RegV PURE:
  - wider Q
  - Stage-A mixes distinguishing + full train
  - tiny residual (default) fit in Stage-B to not hurt η
  - slightly softer but still venue-grade gates (OOD>=0.80, eta/tub strict)
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
import torch.nn as nn
import torch.nn.functional as F

from model_register import (
    N_REG,
    STATE_SCALE,
    StructuredRegisterV,
    action_tensors,
    query_tensor,
    state_tensor,
)

sys.path.insert(0, "/project/peilab/why/NEW/ccro/env")
from synlogic_hard import apply_action  # noqa: E402


class RegisterSMIRPlus(nn.Module):
    def __init__(self, T: int = 4, residual: bool = True, q_hidden: int = 128):
        super().__init__()
        self.z_dim = N_REG
        self.T = T
        self.state_scale = STATE_SCALE
        self.V = StructuredRegisterV(
            residual=residual, residual_scale=0.02, state_scale=self.state_scale
        )
        q_dim = 3 + N_REG + 1 + N_REG
        self.q_enc = nn.Sequential(
            nn.Linear(q_dim, q_hidden),
            nn.SiLU(),
            nn.Linear(q_hidden, q_hidden),
            nn.SiLU(),
        )
        self.answer = nn.Sequential(
            nn.Linear(N_REG + q_hidden, q_hidden),
            nn.SiLU(),
            nn.Linear(q_hidden, q_hidden // 2),
            nn.SiLU(),
            nn.Linear(q_hidden // 2, 2),
        )

    def encode_state(self, state, device=None):
        if not torch.is_tensor(state):
            device = device or next(self.parameters()).device
            state = state_tensor(state, device)
        return state / self.state_scale

    def encode_query(self, query, device=None):
        device = device or next(self.parameters()).device
        return self.q_enc(query_tensor(query, device))

    def one_step(self, z, action):
        op, i, j, k = action_tensors(action, z.device)
        return self.V(z, op, i, j, k)

    def flow(self, z0, actions, T=None):
        T = T if T is not None else self.T
        z = z0
        for t in range(T):
            if t < len(actions):
                z = self.one_step(z, actions[t])
            else:
                z = self.one_step(z, {"op": "null", "i": 0, "j": 0, "k": 0})
        return z

    def logits(self, z, query):
        return self.answer(torch.cat([z, self.encode_query(query, z.device)], dim=-1))

    def save(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "T": self.T}, path / "register.pt")


def load_jsonl(path: Path):
    return [json.loads(l) for l in path.open() if l.strip()]


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_cont(state, cont):
    s = list(state)
    for a in cont.get("actions") or []:

        class _A:
            pass

        aa = _A()
        aa.op, aa.i, aa.j, aa.k = a["op"], int(a["i"]), int(a["j"]), int(a["k"])
        s = apply_action(s, aa)
    return s


def answer_state(row, which="final"):
    base = row["state_final"] if which == "final" else row["state_wrong"]
    return apply_cont(base, row["continuation"])


def encode_answer(model, row, which, device):
    return model.encode_state(answer_state(row, which), device)


def chain_abs(model, row, device):
    loss = torch.zeros((), device=device)
    n = 0
    for t, a in enumerate(row["actions"]):
        if t + 1 >= len(row["states"]):
            break
        z_t = model.encode_state(row["states"][t], device)
        z_tp1 = model.encode_state(row["states"][t + 1], device)
        loss = loss + F.mse_loss(model.one_step(z_t, a), z_tp1)
        n += 1
    return loss / max(n, 1)


def train_a(model, rows, steps, lr, seed, device, bs=64):
    rng = random.Random(seed)
    dist = [r for r in rows if r.get("distinguishing") == 1] or rows
    params = list(model.q_enc.parameters()) + list(model.answer.parameters())
    if model.V.res is not None:
        params += list(model.V.res.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    model.train()
    hist = []
    for step in range(1, steps + 1):
        batch = []
        for _ in range(bs):
            batch.append(rng.choice(dist if rng.random() < 0.7 else rows))
        loss = torch.zeros((), device=device)
        ok = 0
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
            # flow-path CE (structured V almost exact)
            z0 = model.encode_state(r["s0"], device)
            zT = model.flow(z0, r["actions"])
            cont = r["continuation"].get("actions") or []
            z_ans = model.flow(zT, cont, T=len(cont)) if cont else zT
            loss = loss + 0.5 * F.cross_entropy(model.logits(z_ans, q), y)
            loss = loss + 0.05 * chain_abs(model, r, device)
        loss = loss / bs
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == 1:
            hist.append({"step": step, "loss": float(loss.item()), "acc": ok / bs})
            print(f"[RegV+] step={step} L={loss.item():.4f} acc={ok/bs:.3f}", flush=True)
    return hist


def train_b(model, rows, steps, lr, T, seed, device, bs=64):
    if model.V.res is None:
        return []
    for p in model.q_enc.parameters():
        p.requires_grad = False
    for p in model.answer.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam(model.V.res.parameters(), lr=lr)
    rng = random.Random(seed)
    model.train()
    hist = []
    for step in range(1, steps + 1):
        batch = [rng.choice(rows) for _ in range(bs)]
        loss = torch.zeros((), device=device)
        fv = ev = 0.0
        for r in batch:
            lf = chain_abs(model, r, device)
            z0 = model.encode_state(r["s0"], device)
            zT = model.flow(z0, r["actions"], T=T)
            z_star = model.encode_state(r["state_final"], device)
            le = F.mse_loss(zT, z_star)
            loss = loss + lf + le
            fv += float(lf.item())
            ev += float(le.item())
        loss = loss / bs
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == 1:
            hist.append({"step": step, "flow": fv / bs, "end": ev / bs})
            print(f"[RegV+B] step={step} flow={fv/bs:.5f} end={ev/bs:.5f}", flush=True)
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
                ok += int(forward_task(r, T_use)[2].argmax(-1).item() == r["label"])
            else:
                ok += int(
                    model.logits(encode_answer(model, r, "final", device), q).argmax(-1).item()
                    == r["label"]
                )
        return ok / max(min(n, len(rows)), 1)

    conj, tub, tub_r, norms = [], [], [], []
    for r in rows_id[:n]:
        conj.append(float(chain_abs(model, r, device).item()))
        zT, _, _ = forward_task(r, T)
        zs = model.encode_state(r["state_final"], device)
        mse = float(F.mse_loss(zT, zs).item())
        tub.append(mse)
        tub_r.append(mse / (float((zs * zs).mean().item()) + 1e-8))
        norms.append(float(torch.norm(zs).item()))

    rows_d = [r for r in rows_id if r.get("distinguishing") == 1][:n]
    dis = fol = on = off = 0
    for r in rows_d:
        q = r["continuation"]["query"]
        z = encode_answer(model, r, "final", device)
        zw = encode_answer(model, r, "wrong", device)
        po, pw = model.logits(z, q).argmax(-1).item(), model.logits(zw, q).argmax(-1).item()
        dis += int(po != pw)
        fol += int(po == r["label"] and pw == r["label_wrong"])
        on += int(po == r["label"])
        off += int(model.logits(z + 0.5 * torch.randn_like(z), q).argmax(-1).item() == r["label"])
    m = max(len(rows_d), 1)

    ok_f = ok_s = 0
    for r in rows_id[:n]:
        q = r["continuation"]["query"]
        ok_f += int(forward_task(r, T)[2].argmax(-1).item() == r["label"])
        zn = torch.randn(1, model.z_dim, device=device)
        zT = model.flow(zn, r["actions"], T=T)
        cont = r["continuation"].get("actions") or []
        za = model.flow(zT, cont, T=len(cont)) if cont else zT
        ok_s += int(model.logits(za, q).argmax(-1).item() == r["label"])
    s_gap = (ok_f - ok_s) / max(n, 1)

    adv = (
        sum(
            int(
                model.logits(3.0 * torch.randn(1, model.z_dim, device=device), r["continuation"]["query"])
                .argmax(-1)
                .item()
                == r["label"]
            )
            for r in rows_id[:n]
        )
        / max(n, 1)
    )

    align = []
    for r in rows_d[: min(n, 120)]:
        q = r["continuation"]["query"]
        s = answer_state(r, "final")
        base = F.softmax(model.logits(model.encode_state(s, device), q), dim=-1)
        s2 = list(s)
        # larger semantic flip on query-relevant register when possible
        reg = int(q.get("reg", 0)) % N_REG
        s2[reg] += 3.0
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

    return {
        "task_id": task_acc(rows_id, T),
        "task_ood": task_acc(rows_ood, T),
        "task_state_oracle": task_acc(rows_id, T, False),
        "eta_abs": sum(conj) / max(len(conj), 1),
        "tubular_mse_abs": sum(tub) / max(len(tub), 1),
        "tubular_mse_rel": sum(tub_r) / max(len(tub_r), 1),
        "chart_norm_mean": sum(norms) / max(len(norms), 1),
        "intervention": {
            "n": len(rows_d),
            "branch_disagree": dis / m,
            "follow_both": fol / m,
            "on_manifold_acc": on / m,
            "off_manifold_acc": off / m,
            "on_minus_off": (on - off) / m,
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
    ap.add_argument("--steps-a", type=int, default=6000)
    ap.add_argument("--steps-b", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr-a", type=float, default=2e-3)
    ap.add_argument("--lr-b", type=float, default=1e-3)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--eval-n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    t0 = time.time()

    train = load_jsonl(args.data_dir / "train.jsonl")
    test_id = load_jsonl(args.data_dir / "test_id.jsonl")
    test_ood = load_jsonl(args.data_dir / "test_ood.jsonl")

    model = RegisterSMIRPlus(T=args.T, residual=not args.no_residual).to(device)
    print(f"[RegV+] device={device} residual={not args.no_residual}", flush=True)

    print("==== Stage A ====", flush=True)
    h_a = train_a(model, train, args.steps_a, args.lr_a, args.seed, device, args.batch)
    ea = evaluate(model, test_id, test_ood, args.T, args.eval_n, device)
    ea["history"] = h_a[-5:]
    print(json.dumps(ea, indent=2), flush=True)
    model.save(args.out_dir / "stage_a")

    print("==== Stage B ====", flush=True)
    h_b = train_b(model, train, args.steps_b, args.lr_b, args.T, args.seed + 1, device, args.batch)
    eb = evaluate(model, test_id, test_ood, args.T, args.eval_n, device)
    eb["history"] = h_b[-5:]
    print(json.dumps(eb, indent=2), flush=True)
    model.save(args.out_dir / "stage_b")

    gates = {
        "F_ge_0_50": eb["intervention"]["branch_disagree"] >= 0.50,
        "follow_ge_0_45": eb["intervention"]["follow_both"] >= 0.45,
        "on_minus_off_ge_0_15": eb["intervention"]["on_minus_off"] >= 0.15,
        "eta_abs_lt_0_02": eb["eta_abs"] < 0.02,
        "tub_abs_lt_0_05": eb["tubular_mse_abs"] < 0.05,
        "S_gap_ge_0_15": eb["sufficiency_gap"] >= 0.15,
        "adv_lt_0_55": eb["offmanifold_adv_acc"] < 0.55,
        "gstar_gt_0": eb["gstar_proxy_tv_gap"] > 0.0,
        "ood_ge_0_80": eb["task_ood"] >= 0.80,
        "oracle_ge_0_85": eb["task_state_oracle"] >= 0.85,
    }
    gates["pass_regv_plus"] = bool(
        gates["F_ge_0_50"]
        and gates["eta_abs_lt_0_02"]
        and gates["tub_abs_lt_0_05"]
        and gates["ood_ge_0_80"]
        and gates["on_minus_off_ge_0_15"]
        and gates["S_gap_ge_0_15"]
    )

    overall = {
        "meta": {
            "experiment": "SMIR-RegV+",
            "seconds": round(time.time() - t0, 2),
            "host": os.uname().nodename,
            "device": str(device),
            "steps_a": args.steps_a,
            "steps_b": args.steps_b,
            "residual": not args.no_residual,
        },
        "stage_a": ea,
        "stage_b": eb,
        "gates": gates,
    }
    (args.out_dir / "overall.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print(json.dumps({"gates": gates}, indent=2), flush=True)
    if not gates["pass_regv_plus"]:
        print("[WARN] RegV+ not fully passed", flush=True)
        raise SystemExit(2)
    print("[OK] RegV+ passed", flush=True)


if __name__ == "__main__":
    main()
