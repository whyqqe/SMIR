#!/usr/bin/env python3
"""Shared helpers for NeurIPS/ICML-oriented continuation-theory suite."""
from __future__ import annotations

import copy
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from suite_common import encode_answer, evaluate_bundle, load_jsonl


def split_by_len(rows, maxlen):
    return [r for r in rows if int(r.get("chain_len", 0)) <= maxlen]


def clone_row_with_query(row, query):
    r = copy.deepcopy(row)
    r["continuation"] = dict(r["continuation"])
    r["continuation"]["query"] = query
    return r


def make_nonseparating_query():
    """Query that is almost always YES for register gt with very low threshold — weak separator."""
    return {"reg": 0, "thr": -1e6, "kind": "gt"}


def apply_nonseparating_queries(rows: List[dict]) -> List[dict]:
    q = make_nonseparating_query()
    out = []
    for r in rows:
        rr = clone_row_with_query(r, q)
        # relabel under weak query: treat as always class 1 if we can't recompute — use label 1
        rr["label"] = 1
        rr["label_wrong"] = 1  # destroys distinguishing signal
        rr["distinguishing"] = 0
        out.append(rr)
    return out


def continuation_ce(model, row, mode, device, rng):
    q = row["continuation"]["query"]
    cont = row["continuation"].get("actions") or []
    y = torch.tensor([row["label"]], device=device)
    actions = list(row["actions"])

    if mode == "empty":
        z = encode_answer(model, row, "final", device)
        return F.cross_entropy(model.logits(z, q), y)

    if mode == "cfact" and int(row.get("distinguishing", 0)) == 1:
        yw = torch.tensor([row["label_wrong"]], device=device)
        z_w0 = model.encode_state(row["state_wrong"], device)
        z_w = model.flow(z_w0, cont, T=len(cont)) if cont else z_w0
        loss = F.cross_entropy(model.logits(z_w, q), yw)
        z_t = encode_answer(model, row, "final", device)
        loss = loss + F.cross_entropy(model.logits(z_t, q), y)
        return loss

    if mode == "prefix" and len(actions) >= 2 and len(row.get("states", [])) >= 3:
        t = rng.randint(1, len(actions) - 1)
        z = model.encode_state(row["states"][t], device)
        rest = actions[t:]
        z = model.flow(z, rest, T=len(rest)) if rest else z
    else:
        z0 = model.encode_state(row["s0"], device)
        z = model.flow(z0, actions, T=max(len(actions), 1))
    z_ans = model.flow(z, cont, T=len(cont)) if cont else z
    return F.cross_entropy(model.logits(z_ans, q), y)


def train_l_continuation(
    model,
    rows_all,
    rows_short,
    steps,
    lr,
    device,
    bs,
    seed,
    p_dist,
    prefer_short,
    mode_probs: List[Tuple[str, float]],
    tag,
    freeze_v=False,
    leak_row=False,
):
    """Single-objective continuation training. Optional leak_row: feed true final state into logits path via wrapper."""
    rng = random.Random(seed)
    dist_all = [r for r in rows_all if r.get("distinguishing") == 1] or rows_all
    dist_short = [r for r in rows_short if r.get("distinguishing") == 1] or rows_short
    pool = rows_short if prefer_short else rows_all
    dist_pool = dist_short if prefer_short else dist_all
    for p in model.V.parameters():
        p.requires_grad = not freeze_v
    q_enc = getattr(model, "q_enc", None)
    answer = getattr(model, "answer", None) or getattr(model, "answer_leak", None)
    for mod in (q_enc, answer):
        if mod is None:
            continue
        for p in mod.parameters():
            p.requires_grad = True
    # leak encoder if present
    if hasattr(model, "leak_enc"):
        for p in model.leak_enc.parameters():
            p.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError(f"train_l_continuation: no trainable params for {type(model).__name__}")
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    modes, probs = zip(*mode_probs)
    model.train()
    for step in range(1, steps + 1):
        batch = [rng.choice(dist_pool if rng.random() < p_dist else pool) for _ in range(bs)]
        loss = torch.zeros((), device=device)
        for r in batch:
            mode = rng.choices(modes, weights=probs, k=1)[0]
            if mode == "cfact" and int(r.get("distinguishing", 0)) != 1:
                mode = "full"
            if leak_row and hasattr(model, "set_leak_row"):
                model.set_leak_row(r)
            loss = loss + continuation_ce(model, r, mode, device, rng)
            # Always attach a counterfactual continuation term on distinguishing rows
            # (still L_continuation; prevents non-causal readout collapse).
            if mode != "cfact" and int(r.get("distinguishing", 0)) == 1:
                loss = loss + continuation_ce(model, r, "cfact", device, rng)
        if hasattr(model, "set_leak_row"):
            model.set_leak_row(None)
        loss = loss / bs
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 50 == 0 or step == 1:
            print(f"[{tag}] step={step} L={loss.item():.4f}", flush=True)


@torch.no_grad()
def continuation_error_by_length(model, rows, n, device, lengths=None):
    rows = rows[:n]
    if lengths is None:
        lengths = sorted(set(int(r["chain_len"]) for r in rows) | {2, 3, 4, 5, 6, 7})
    out = {}
    for L in lengths:
        sub = [r for r in rows if int(r["chain_len"]) == L][:n]
        if not sub:
            # truncate/pad evaluation: flow with min(L, len(actions))
            sub = rows[: min(n, len(rows))]
            ok = 0
            for r in sub:
                q = r["continuation"]["query"]
                cont = r["continuation"].get("actions") or []
                acts = r["actions"][:L]
                z0 = model.encode_state(r["s0"], device)
                z = model.flow(z0, acts, T=max(len(acts), 1))
                z = model.flow(z, cont, T=len(cont)) if cont else z
                # label may be wrong if truncated — skip if L != chain_len
                if L != int(r["chain_len"]):
                    continue
                ok += int(model.logits(z, q).argmax(-1).item() == r["label"])
            m = sum(1 for r in sub if L == int(r["chain_len"]))
            out[str(L)] = ok / max(m, 1) if m else None
            continue
        ok = 0
        for r in sub:
            q = r["continuation"]["query"]
            cont = r["continuation"].get("actions") or []
            z0 = model.encode_state(r["s0"], device)
            z = model.flow(z0, r["actions"], T=max(len(r["actions"]), 1))
            z = model.flow(z, cont, T=len(cont)) if cont else z
            ok += int(model.logits(z, q).argmax(-1).item() == r["label"])
        out[str(L)] = ok / max(len(sub), 1)
    return out


@torch.no_grad()
def eval_replace_battery(model, rows, n, device):
    rows = rows[:n]
    full = rep = pref = cfact = 0
    n_cf = 0
    for r in rows:
        q = r["continuation"]["query"]
        cont = r["continuation"].get("actions") or []
        actions = r["actions"]
        z0 = model.encode_state(r["s0"], device)
        zT = model.flow(z0, actions, T=max(len(actions), 1))
        z_full = model.flow(zT, cont, T=len(cont)) if cont else zT
        full += int(model.logits(z_full, q).argmax(-1).item() == r["label"])

        if len(actions) >= 2 and len(r.get("states", [])) >= 3:
            t = max(1, len(actions) // 2)
            z_mid = model.encode_state(r["states"][t], device)
            z_r = model.flow(z_mid, actions[t:], T=len(actions[t:])) if actions[t:] else z_mid
            z_r = model.flow(z_r, cont, T=len(cont)) if cont else z_r
            rep += int(model.logits(z_r, q).argmax(-1).item() == r["label"])
            z_p = model.flow(z0, actions[:t], T=t)
            z_p = model.flow(z_p, actions[t:], T=len(actions[t:])) if actions[t:] else z_p
            z_p = model.flow(z_p, cont, T=len(cont)) if cont else z_p
            pref += int(model.logits(z_p, q).argmax(-1).item() == r["label"])
        else:
            rep += int(model.logits(z_full, q).argmax(-1).item() == r["label"])
            pref += int(model.logits(z_full, q).argmax(-1).item() == r["label"])

        if int(r.get("distinguishing", 0)) == 1:
            n_cf += 1
            z_w0 = model.encode_state(r["state_wrong"], device)
            z_w = model.flow(z_w0, cont, T=len(cont)) if cont else z_w0
            po = model.logits(z_full, q).argmax(-1).item()
            pw = model.logits(z_w, q).argmax(-1).item()
            cfact += int(po != pw and po == r["label"] and pw == r["label_wrong"])
    m = max(len(rows), 1)
    return {
        "full_rollout_acc": full / m,
        "replace_midstate_acc": rep / m,
        "prefix_latent_acc": pref / m,
        "cfact_follow_both": cfact / max(n_cf, 1),
        "n_cfact": n_cf,
    }


def theory_core_gates(bundle, replace, gate_f=0.45, gate_ood=0.75, gate_rep=0.70):
    F = bundle["flow_intervention"]["branch_disagree"]
    return {
        "F_flow": F,
        "ood": bundle["task_ood"],
        "task_id": bundle["task_id"],
        "eta": bundle["eta_abs"],
        "tub": bundle["tubular_mse_abs"],
        "S_gap": bundle["sufficiency_gap"],
        "replace_acc": replace["replace_midstate_acc"],
        "cfact_follow": replace["cfact_follow_both"],
        "pass_core": (
            F >= gate_f
            and bundle["task_ood"] >= gate_ood
            and replace["replace_midstate_acc"] >= gate_rep
            and bundle["sufficiency_gap"] >= 0.08
        ),
    }


class LeakSuite(nn.Module):
    """Bypass bottleneck: answer head can see encoded true final state (C2)."""

    def __init__(self, base: nn.Module, leak_strength: float = 1.0):
        super().__init__()
        self.base = base
        self.kind = base.kind
        self.is_graph = getattr(base, "is_graph", False)
        self.z_dim = base.z_dim
        self.T = base.T
        self.leak_strength = float(leak_strength)
        self.leak_enc = nn.Sequential(
            nn.Linear(base.z_dim, base.z_dim),
            nn.SiLU(),
            nn.Linear(base.z_dim, base.z_dim),
        )
        self._leak_row = None
        # widen answer to accept z + leak
        hid = 128
        self.answer_leak = nn.Sequential(
            nn.Linear(base.z_dim * 2 + 128, hid),
            nn.SiLU(),
            nn.Linear(hid, 2),
        )
        # Alias for train_l_continuation / SuiteSMIR-compatible attribute access.
        self.answer = self.answer_leak
        # reuse q_enc from base; answer takes [z, leak, q]
        self.q_enc = base.q_enc
        self.V = base.V

    def set_leak_row(self, row):
        self._leak_row = row

    def encode_state(self, state, device=None):
        return self.base.encode_state(state, device)

    def encode_query(self, query, device=None):
        return self.base.encode_query(query, device)

    def one_step(self, z, action):
        return self.base.one_step(z, action)

    def flow(self, z0, actions, T=None):
        return self.base.flow(z0, actions, T=T)

    def logits(self, z, query):
        qe = self.encode_query(query, z.device)
        if self._leak_row is not None and self.leak_strength > 0:
            z_true = self.encode_state(self._leak_row["state_final"], z.device)
            leak = self.leak_enc(z_true) * self.leak_strength
        else:
            leak = torch.zeros_like(z)
        return self.answer_leak(torch.cat([z, leak, qe], dim=-1))

    def save(self, path):
        self.base.save(path)


class ChartWrap(nn.Module):
    def __init__(self, base, mode, seed=0):
        super().__init__()
        self.base = base
        self.mode = mode
        self.kind = base.kind
        self.is_graph = bool(getattr(base, "is_graph", str(base.kind).startswith("g")))
        self.z_dim = base.z_dim
        self.T = base.T
        g = torch.Generator().manual_seed(seed + 7)
        if mode == "proj":
            P = torch.randn(5, 5, generator=g)
            P[:, -1] = 0
            self.register_buffer("P", P / (P.norm() + 1e-6) * (5 ** 0.5))
        self.scale = {"legal": 1.0, "scale2": 2.0}.get(mode, 1.0)

    def encode_state(self, state, device=None):
        z = self.base.encode_state(state, device)
        if self.mode in {"legal", "scale2"}:
            return z * self.scale
        if self.mode == "proj":
            return z @ self.P.to(z.device)
        if self.mode == "zero":
            return torch.zeros_like(z)
        return z

    def encode_query(self, query, device=None):
        return self.base.encode_query(query, device)

    def one_step(self, z, action):
        return self.base.one_step(z, action)

    def flow(self, z0, actions, T=None):
        return self.base.flow(z0, actions, T=T)

    def logits(self, z, query):
        return self.base.logits(z, query)

    @property
    def V(self):
        return self.base.V


# Heavier cfact: without it, Q collapses to majority and Flow-F≈0 even with Hard V*.
FULL_MODES = [("empty", 0.15), ("full", 0.30), ("prefix", 0.20), ("cfact", 0.35)]
SHORT_MODES = [("empty", 0.50), ("full", 0.50), ("prefix", 0.0), ("cfact", 0.0)]
