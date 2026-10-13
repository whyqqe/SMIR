#!/usr/bin/env python3
"""Exp-0: Diagnose E-collapse vs Q-ignore on a frozen SCF (or SF) checkpoint.

Does NOT retrain ContCE/Outcome/SCF. Probes:
  1) ||E(on)-E(wrong)|| / cosine  -> E collapse?
  2) linear probe E(state)->label  vs Q(E(state)) -> Q ignore?
  3) ||z_T|| vs ||E(s*)|| / stepwise drift -> tubular pathology?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from model_smir import OP2ID, SMIRModel


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ops_tensor(row: dict, T: int, device) -> torch.Tensor:
    ids = [OP2ID.get(a["op"], 3) for a in row["actions"][:T]]
    while len(ids) < T:
        ids.append(OP2ID["null"])
    return torch.tensor([ids[:T]], dtype=torch.long, device=device)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--ckpt", type=Path, required=True, help="light save dir (adapter+heads)")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--z-dim", type=int, default=None)
    ap.add_argument("--T", type=int, default=None)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    test_id = load_jsonl(args.data_dir / "test_id.jsonl")
    rows_d = [r for r in test_id if r.get("distinguishing") == 1][: args.n]
    rows = test_id[: args.n]
    print(f"[diag] load {args.ckpt} n_dist={len(rows_d)} n={len(rows)}", flush=True)

    model = SMIRModel.load_light(args.model, args.ckpt, z_dim=args.z_dim, T=args.T)
    T = model.T
    device = model.device

    # --- Probe 1: E separation ---
    l2s, cosines = [], []
    for r in rows_d:
        z_on = model.encode_text(r["prompt_state_enc"])
        z_w = model.encode_text(r["prompt_state_wrong_enc"])
        l2s.append(float(torch.norm(z_on - z_w).item()))
        cosines.append(float(F.cosine_similarity(z_on, z_w).item()))
    e_sep = {
        "n": len(rows_d),
        "l2_mean": sum(l2s) / max(len(l2s), 1),
        "l2_median": float(sorted(l2s)[len(l2s) // 2]) if l2s else 0.0,
        "cosine_mean": sum(cosines) / max(len(cosines), 1),
    }

    # --- Probe 2: linear probe vs Q ---
    # Collect E(state) for on-manifold states with labels
    zs, ys = [], []
    for r in rows_d:
        zs.append(model.encode_text(r["prompt_state_enc"]).squeeze(0).cpu())
        ys.append(r["label"])
        zs.append(model.encode_text(r["prompt_state_wrong_enc"]).squeeze(0).cpu())
        ys.append(r["label_wrong"])
    Z = torch.stack(zs)  # [2n, d]
    Y = torch.tensor(ys, dtype=torch.long)
    # closed-form ridge logistic via least squares on one-hot (fast proxy)
    Yoh = F.one_hot(Y, 2).float()
    # Z @ W ≈ Yoh ; W = (Z^T Z + λI)^{-1} Z^T Yoh
    d = Z.size(1)
    XtX = Z.T @ Z + 1e-2 * torch.eye(d)
    W = torch.linalg.solve(XtX, Z.T @ Yoh)
    pred_lin = (Z @ W).argmax(-1)
    lin_acc = float((pred_lin == Y).float().mean().item())

    q_ok = 0
    for r in rows_d:
        p_on = model.logits(model.encode_text(r["prompt_state_enc"])).argmax(-1).item()
        p_w = model.logits(model.encode_text(r["prompt_state_wrong_enc"])).argmax(-1).item()
        q_ok += int(p_on == r["label"])
        q_ok += int(p_w == r["label_wrong"])
    q_acc = q_ok / max(2 * len(rows_d), 1)
    disagree = 0
    for r in rows_d:
        p_on = model.logits(model.encode_text(r["prompt_state_enc"])).argmax(-1).item()
        p_w = model.logits(model.encode_text(r["prompt_state_wrong_enc"])).argmax(-1).item()
        disagree += int(p_on != p_w)
    q_branch = disagree / max(len(rows_d), 1)

    # --- Probe 3: scale / tubular drift ---
    norms_zT, norms_E, abs_mse, rel_mse, step_drift = [], [], [], [], []
    for r in rows:
        z0 = model.encode_text(r["prompt"])
        ops = ops_tensor(r, T, device)
        z = z0
        drifts = []
        # stepwise vs successive state encodings if available
        states = r.get("states") or []
        for t in range(ops.size(1)):
            z = model.V(z, ops[:, t])
            if t < len(states) and isinstance(states[t], dict) and "text" in states[t]:
                # optional; shadow may store differently
                pass
        zT = z
        z_star = model.encode_text(r["prompt_state_enc"])
        nT = float(torch.norm(zT).item())
        nE = float(torch.norm(z_star).item())
        mse = float(F.mse_loss(zT, z_star).item())
        norms_zT.append(nT)
        norms_E.append(nE)
        abs_mse.append(mse)
        rel_mse.append(mse / max(nE * nE / zT.numel(), 1e-8))
        # one-step conjugacy residual at t=0 using step_op
        z_t = model.encode_text(r["prompt_state_t"])
        z_tp1 = model.encode_text(r["prompt_state_tp1"])
        op = torch.tensor([OP2ID.get(r["step_op"], 3)], device=device)
        step_drift.append(float(F.mse_loss(model.one_step(z_t, op), z_tp1).item()))

    scale = {
        "norm_zT_mean": sum(norms_zT) / max(len(norms_zT), 1),
        "norm_E_mean": sum(norms_E) / max(len(norms_E), 1),
        "tubular_mse_abs": sum(abs_mse) / max(len(abs_mse), 1),
        "tubular_mse_rel": sum(rel_mse) / max(len(rel_mse), 1),
        "eta_one_step": sum(step_drift) / max(len(step_drift), 1),
        "norm_ratio_zT_over_E": (sum(norms_zT) / max(sum(norms_E), 1e-8)),
    }

    # Verdict
    e_collapsed = e_sep["cosine_mean"] > 0.95 or e_sep["l2_mean"] < 0.5
    q_ignores = (lin_acc >= 0.70 and q_acc < 0.60) or (q_branch < 0.05 and lin_acc >= 0.65)
    both_bad = lin_acc < 0.60 and q_acc < 0.60
    tubular_pathology = scale["norm_ratio_zT_over_E"] > 5.0 or scale["tubular_mse_abs"] > 100.0

    verdict = {
        "E_collapsed": bool(e_collapsed),
        "Q_ignores_separable_E": bool(q_ignores),
        "E_and_Q_both_weak": bool(both_bad),
        "tubular_pathology": bool(tubular_pathology),
        "recommend": (
            "fix_E_first"
            if e_collapsed or both_bad
            else ("fix_Q_readout" if q_ignores else "proceed_PhaseF_with_L_end")
        ),
    }

    out = {
        "meta": {"ckpt": str(args.ckpt), "n": args.n, "T": T, "z_dim": model.z_dim},
        "e_separation": e_sep,
        "readout": {
            "linear_probe_acc": lin_acc,
            "Q_state_acc": q_acc,
            "Q_branch_disagree": q_branch,
        },
        "scale_tubular": scale,
        "verdict": verdict,
    }
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)
    print(f"[diag] recommend={verdict['recommend']}", flush=True)


if __name__ == "__main__":
    main()
