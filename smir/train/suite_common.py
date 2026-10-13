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

from model_register import N_REG  # noqa: E402


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


def _op_id(op: str) -> int:
    return {"add": 0, "sub": 1, "mul": 2, "null": 3}.get(op, 3)


def _is_graph_model(model) -> bool:
    if bool(getattr(model, "is_graph", False)):
        return True
    kind = str(getattr(model, "kind", "") or "")
    return kind.startswith("g")


@torch.no_grad()
def routing_accuracy(model, rows, n, device):
    """Fraction of steps where soft/discrete router argmax matches true op."""
    V = getattr(model, "V", None)
    if V is None or not hasattr(V, "route"):
        return None
    is_graph = _is_graph_model(model)
    if is_graph:
        from model_suite import GOP2ID, graph_action_tensors

        action_fn = graph_action_tensors
        op_id_fn = lambda op: GOP2ID.get(op, 2)
        max_true = 1  # relax/set; skip null
    else:
        from model_register import action_tensors

        action_fn = action_tensors
        op_id_fn = _op_id
        max_true = 2  # add/sub/mul; skip null
    ok = tot = 0
    for r in rows[:n]:
        for t, a in enumerate(r["actions"]):
            if t + 1 >= len(r["states"]):
                break
            z = model.encode_state(r["states"][t], device)
            op, i, j, k = action_fn(a, device)
            if hasattr(V, "route_logits"):
                logits = V.route_logits(z, op, i, j, k)
            elif is_graph:
                op_oh = F.one_hot(op.clamp(0, 2), 3).float()
                i_oh = F.one_hot(i.clamp(0, N_REG - 1), N_REG).float()
                j_oh = F.one_hot(j.clamp(0, N_REG - 1), N_REG).float()
                logits = V.route(torch.cat([z, op_oh, i_oh, j_oh], dim=-1))
            else:
                op_oh = F.one_hot(op.clamp(0, 3), 4).float()
                i_oh = F.one_hot(i.clamp(0, N_REG - 1), N_REG).float()
                j_oh = F.one_hot(j.clamp(0, N_REG - 1), N_REG).float()
                k_oh = F.one_hot(k.clamp(0, N_REG - 1), N_REG).float()
                logits = V.route(torch.cat([z, op_oh, i_oh, j_oh, k_oh], dim=-1))
            pred = int(logits.argmax(-1).item())
            true = op_id_fn(a["op"])
            if true <= max_true:
                ok += int(pred == true)
                tot += 1
    return ok / max(tot, 1)


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
    # Flow-F: true branch via learned rollout; wrong branch via wrong state + cont flow
    f_dis = f_fol = f_on = 0
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

        z0 = model.encode_state(r["s0"], device)
        Tc = max(len(r["actions"]), 1)
        zT = model.flow(z0, r["actions"], T=Tc)
        cont = r["continuation"].get("actions") or []
        z_true = model.flow(zT, cont, T=len(cont)) if cont else zT
        z_wrong0 = model.encode_state(r["state_wrong"], device)
        z_wrong = model.flow(z_wrong0, cont, T=len(cont)) if cont else z_wrong0
        fo = model.logits(z_true, q).argmax(-1).item()
        fw = model.logits(z_wrong, q).argmax(-1).item()
        f_dis += int(fo != fw)
        f_fol += int(fo == r["label"] and fw == r["label_wrong"])
        f_on += int(fo == r["label"])
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
    route_acc = routing_accuracy(model, rows_id, n, device)

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
        "flow_intervention": {
            "n": len(rows_d),
            "branch_disagree": f_dis / m,
            "follow_both": f_fol / m,
            "on_manifold_acc": f_on / m,
        },
        "routing_acc": route_acc,
        "sufficiency_gap": (ok_f - ok_s) / max(n, 1),
        "length_curve": length,
    }
