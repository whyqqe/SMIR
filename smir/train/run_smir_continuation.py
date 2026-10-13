#!/usr/bin/env python3
"""Preliminary validation of new_theory.md: counterfactual continuation consistency.

Single training objective L_continuation (no conjugacy / route auxiliary losses).
Batch mixes: empty readout, full rollouts, prefix continuations, counterfactual branches.
Evaluation battery follows §11: length OOD, intervention F, mid-state replace, leak/sufficiency.
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

from model_suite import SuiteSMIR
from suite_common import encode_answer, evaluate_bundle, load_jsonl, set_seed


CONT_MODES = ("empty", "full", "prefix", "cfact")


def split_by_len(rows, maxlen):
    return [r for r in rows if int(r.get("chain_len", 0)) <= maxlen]


def continuation_loss(model, row, mode, device, rng):
    """One-sample CE under a continuation mode. Returns (loss, aux_dict)."""
    q = row["continuation"]["query"]
    cont = row["continuation"].get("actions") or []
    y = torch.tensor([row["label"]], device=device)

    if mode == "empty":
        # empty α: readout from encoded terminal state (chart E)
        z = encode_answer(model, row, "final", device)
        loss = F.cross_entropy(model.logits(z, q), y)
        return loss, {"mode": mode}

    if mode == "cfact" and row.get("distinguishing") == 1:
        yw = torch.tensor([row["label_wrong"]], device=device)
        # counterfactual: start from wrong intermediate state, same query continuation
        z_w0 = model.encode_state(row["state_wrong"], device)
        z_w = model.flow(z_w0, cont, T=len(cont)) if cont else z_w0
        loss = F.cross_entropy(model.logits(z_w, q), yw)
        # also keep true branch readable
        z_t = encode_answer(model, row, "final", device)
        loss = loss + F.cross_entropy(model.logits(z_t, q), y)
        return loss, {"mode": mode}

    # full or prefix execution through latent T
    actions = list(row["actions"])
    if mode == "prefix" and len(actions) >= 2:
        t = rng.randint(1, len(actions) - 1)
        # replace mid-state with encoded true state, then continue (tests T from reachable z)
        z = model.encode_state(row["states"][t], device)
        rest = actions[t:]
        z = model.flow(z, rest, T=len(rest)) if rest else z
    else:
        z0 = model.encode_state(row["s0"], device)
        Tc = max(len(actions), 1)
        z = model.flow(z0, actions, T=Tc)
    z_ans = model.flow(z, cont, T=len(cont)) if cont else z
    loss = F.cross_entropy(model.logits(z_ans, q), y)
    return loss, {"mode": mode}


def train_continuation(
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
    mode_probs,
    tag,
    freeze_v=False,
):
    rng = random.Random(seed)
    dist_all = [r for r in rows_all if r.get("distinguishing") == 1] or rows_all
    dist_short = [r for r in rows_short if r.get("distinguishing") == 1] or rows_short
    pool = rows_short if prefer_short else rows_all
    dist_pool = dist_short if prefer_short else dist_all

    for p in model.V.parameters():
        p.requires_grad = not freeze_v
    for p in list(model.q_enc.parameters()) + list(model.answer.parameters()):
        p.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    modes, probs = zip(*mode_probs)
    model.train()

    for step in range(1, steps + 1):
        batch = [rng.choice(dist_pool if rng.random() < p_dist else pool) for _ in range(bs)]
        loss = torch.zeros((), device=device)
        ok = n_cf = 0
        for r in batch:
            mode = rng.choices(modes, weights=probs, k=1)[0]
            if mode == "cfact" and r.get("distinguishing") != 1:
                mode = "full"
            li, _ = continuation_loss(model, r, mode, device, rng)
            loss = loss + li
            with torch.no_grad():
                q = r["continuation"]["query"]
                z = encode_answer(model, r, "final", device)
                ok += int(model.logits(z, q).argmax(-1).item() == r["label"])
                n_cf += int(mode == "cfact")
        loss = loss / bs
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 50 == 0 or step == 1:
            print(
                f"[{tag}] step={step} L={loss.item():.4f} readout_acc={ok/bs:.3f} cfact_in_batch={n_cf}",
                flush=True,
            )


@torch.no_grad()
def eval_replace_and_prefix(model, rows, n, device):
    """Mid-state replace + prefix continuation accuracy (theory §11)."""
    rows = rows[:n]
    rep_ok = pref_ok = full_ok = 0
    for r in rows:
        q = r["continuation"]["query"]
        cont = r["continuation"].get("actions") or []
        actions = r["actions"]
        # full latent rollout
        z0 = model.encode_state(r["s0"], device)
        zT = model.flow(z0, actions, T=max(len(actions), 1))
        z_full = model.flow(zT, cont, T=len(cont)) if cont else zT
        full_ok += int(model.logits(z_full, q).argmax(-1).item() == r["label"])

        if len(actions) >= 2 and len(r["states"]) >= 3:
            t = max(1, len(actions) // 2)
            # replace: inject true mid chart state, continue
            z_mid = model.encode_state(r["states"][t], device)
            z_r = model.flow(z_mid, actions[t:], T=len(actions[t:])) if actions[t:] else z_mid
            z_r = model.flow(z_r, cont, T=len(cont)) if cont else z_r
            rep_ok += int(model.logits(z_r, q).argmax(-1).item() == r["label"])
            # prefix-only latent (no replace): flow first t then rest
            z_p = model.flow(z0, actions[:t], T=t)
            z_p = model.flow(z_p, actions[t:], T=len(actions[t:])) if actions[t:] else z_p
            z_p = model.flow(z_p, cont, T=len(cont)) if cont else z_p
            pref_ok += int(model.logits(z_p, q).argmax(-1).item() == r["label"])
        else:
            rep_ok += int(model.logits(z_full, q).argmax(-1).item() == r["label"])
            pref_ok += int(model.logits(z_full, q).argmax(-1).item() == r["label"])
    m = max(len(rows), 1)
    return {
        "full_rollout_acc": full_ok / m,
        "replace_midstate_acc": rep_ok / m,
        "prefix_latent_acc": pref_ok / m,
    }


def theory_gates(bundle, replace, gate_f=0.45, gate_ood=0.75):
    F = bundle["flow_intervention"]["branch_disagree"]
    ood = bundle["task_ood"]
    leak = bundle["sufficiency_gap"]
    rep = replace["replace_midstate_acc"]
    return {
        "F_flow": F,
        "ood": ood,
        "eta": bundle["eta_abs"],
        "tub": bundle["tubular_mse_abs"],
        "S_gap": leak,
        "replace_acc": rep,
        "prefix_acc": replace["prefix_latent_acc"],
        "pass_continuation_core": (
            F >= gate_f
            and ood >= gate_ood
            and leak >= 0.08
            and rep >= 0.70
        ),
        "pass_replace": rep >= 0.70,
        "pass_geometry_diag": bundle["eta_abs"] < 0.1 and bundle["tubular_mse_abs"] < 0.35,
    }


def run_one(kind, data_dir, out_dir, args, device, is_graph=False):
    train = load_jsonl(data_dir / "train.jsonl")
    test_id = load_jsonl(data_dir / "test_id.jsonl")
    test_ood = load_jsonl(data_dir / "test_ood.jsonl")
    short = split_by_len(train, 4 if not is_graph else 4)
    T = args.T_graph if is_graph else args.T

    mode_probs = [
        ("empty", 0.20),
        ("full", 0.35),
        ("prefix", 0.25),
        ("cfact", 0.20),
    ]

    model = SuiteSMIR(kind=kind, T=T).to(device)
    freeze_v = kind in {"hard", "ghard"}
    tag = f"{kind}"
    print(f"[CONT] kind={kind} graph={is_graph} freeze_V={freeze_v} train={len(train)}", flush=True)

    # Phase A: short / high distinguishing
    train_continuation(
        model, train, short, args.steps_1, 2e-3, device, args.batch, args.seed,
        p_dist=0.85, prefer_short=True, mode_probs=mode_probs, tag=f"{tag}-A",
        freeze_v=freeze_v,
    )
    # Phase B: longer mix
    train_continuation(
        model, train, short, args.steps_2, 1e-3, device, args.batch, args.seed + 1,
        p_dist=0.75, prefer_short=False, mode_probs=mode_probs, tag=f"{tag}-B",
        freeze_v=freeze_v,
    )
    # Phase C: if learnable V, light joint polish still under L_cont only
    if not freeze_v:
        train_continuation(
            model, train, short, args.steps_v, 5e-4, device, args.batch, args.seed + 2,
            p_dist=0.80, prefer_short=False, mode_probs=mode_probs, tag=f"{tag}-C",
            freeze_v=False,
        )

    bundle = evaluate_bundle(model, test_id, test_ood, T, args.eval_n, device)
    replace = eval_replace_and_prefix(model, test_id, args.eval_n, device)
    gates = theory_gates(bundle, replace, gate_f=args.gate_f, gate_ood=args.gate_ood)

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "final")
    summary = {
        "meta": {
            "experiment": "SMIR-ContinuationTheory",
            "theory": "new_theory.md counterfactual continuation consistency",
            "kind": kind,
            "domain": "graph" if is_graph else "register",
            "seed": args.seed,
            "single_objective": "L_continuation",
            "no_aux_conj_route": True,
            "host": os.uname().nodename,
        },
        "final": bundle,
        "replace_battery": replace,
        "gates": gates,
    }
    (out_dir / "overall.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"kind": kind, "gates": gates}, indent=2), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reg-data", type=Path, required=True)
    ap.add_argument("--graph-data", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kinds", type=str, default="hard,discrete,mlp")
    ap.add_argument("--graph-kinds", type=str, default="ghard,gdisc")
    ap.add_argument("--steps-1", type=int, default=2500)
    ap.add_argument("--steps-2", type=int, default=3500)
    ap.add_argument("--steps-v", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--eval-n", type=int, default=300)
    ap.add_argument("--T", type=int, default=5)
    ap.add_argument("--T-graph", type=int, default=8)
    ap.add_argument("--gate-f", type=float, default=0.45)
    ap.add_argument("--gate-ood", type=float, default=0.75)
    ap.add_argument("--skip-graph", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    results = {}
    for kind in [k.strip() for k in args.kinds.split(",") if k.strip()]:
        results[kind] = run_one(kind, args.reg_data, args.out_dir / f"reg_{kind}", args, device, False)

    if not args.skip_graph and args.graph_data and (args.graph_data / "train.jsonl").exists():
        for kind in [k.strip() for k in args.graph_kinds.split(",") if k.strip()]:
            results[kind] = run_one(kind, args.graph_data, args.out_dir / f"graph_{kind}", args, device, True)

    # Feasibility claims for this seed
    reg_hard = (results.get("hard") or {}).get("gates") or {}
    reg_disc = (results.get("discrete") or {}).get("gates") or {}
    reg_mlp = (results.get("mlp") or {}).get("gates") or {}
    g_hard = (results.get("ghard") or {}).get("gates") or {}
    g_disc = (results.get("gdisc") or {}).get("gates") or {}

    claims = {
        "pass_reg_hard_cont": bool(reg_hard.get("pass_continuation_core")),
        "pass_reg_discrete_cont": bool(reg_disc.get("pass_continuation_core")),
        "pass_reg_mlp_cont": bool(reg_mlp.get("pass_continuation_core")),
        "structured_beats_mlp": (
            float(reg_hard.get("F_flow") or 0) > float(reg_mlp.get("F_flow") or 0) + 0.05
            if reg_hard and reg_mlp
            else False
        ),
        "pass_graph_hard_cont": bool(g_hard.get("pass_continuation_core")),
        "pass_graph_discrete_cont": bool(g_disc.get("pass_continuation_core")),
        "pass_theory_feasibility": bool(
            reg_hard.get("pass_continuation_core")
            and (reg_disc.get("pass_continuation_core") or g_hard.get("pass_continuation_core"))
        ),
    }
    summary = {
        "meta": {
            "seconds": round(time.time() - t0, 2),
            "seed": args.seed,
            "host": os.uname().nodename,
            "objective": "L_continuation_only",
        },
        "claims": claims,
        "results": {k: v.get("gates") for k, v in results.items()},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not claims["pass_theory_feasibility"]:
        raise SystemExit(2)
    print("[OK] continuation theory feasibility passed", flush=True)


if __name__ == "__main__":
    main()
