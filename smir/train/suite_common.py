#!/usr/bin/env python3
"""Shared helpers for venue suite experiments."""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

import sys

sys.path.insert(0, "/project/peilab/why/NEW/ccro/env")
from synlogic_hard import apply_action  # noqa: E402


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


@torch.no_grad()
def evaluate_bundle(model, rows_id, rows_ood, T, n, device):
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

    conj, tub = [], []
    for r in rows_id[:n]:
        conj.append(float(chain_abs(model, r, device).item()))
        zT, _, _ = forward_task(r, T)
        zs = model.encode_state(r["state_final"], device)
        tub.append(float(F.mse_loss(zT, zs).item()))

    rows_d = [r for r in rows_id if r.get("distinguishing") == 1][:n]
    dis = fol = on = off = 0
    for r in rows_d:
        q = r["continuation"]["query"]
        z = encode_answer(model, r, "final", device)
        zw = encode_answer(model, r, "wrong", device)
        po = model.logits(z, q).argmax(-1).item()
        pw = model.logits(zw, q).argmax(-1).item()
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

    tmax = max((len(r["actions"]) for r in rows_id[:n]), default=T)
    length = {str(t): task_acc(rows_id, t) for t in sorted(set([T, T + 2, T + 4, tmax, tmax + 2]))}

    return {
        "task_id": task_acc(rows_id, T),
        "task_ood": task_acc(rows_ood, T),
        "task_state_oracle": task_acc(rows_id, T, False),
        "eta_abs": sum(conj) / max(len(conj), 1),
        "tubular_mse_abs": sum(tub) / max(len(tub), 1),
        "intervention": {
            "n": len(rows_d),
            "branch_disagree": dis / m,
            "follow_both": fol / m,
            "on_manifold_acc": on / m,
            "off_manifold_acc": off / m,
            "on_minus_off": (on - off) / m,
        },
        "sufficiency_gap": (ok_f - ok_s) / max(n, 1),
        "length_curve": length,
    }
