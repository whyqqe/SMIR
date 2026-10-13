#!/usr/bin/env python3
"""SMIR Venue Suite — multiple experiments in one pipeline.

E1 RegV-OOD:   hard V + long-chain train data → push OOD>=0.80
E2 SoftV:      learnable soft op routing (non-trivial structure)
E3 MLP-TF:     unstructured MLP V + absolute teacher force (learnability)
E4 MLP-Task:   unstructured MLP V + task only (ablation: should fail η)
E5 Length:     length curves on best model
E6 Contrast:   refresh four/five-way table + suite summary

Writes: artifacts/runs/smir_venue_suite/{e1..e4,summary}.json
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
from suite_common import (
    answer_state,
    chain_abs,
    encode_answer,
    evaluate_bundle,
    load_jsonl,
    set_seed,
)


def train_q_and_v(
    model,
    rows,
    steps,
    lr,
    device,
    bs,
    seed,
    train_v: bool,
    lam_flow: float,
    lam_task: float,
    tag: str,
    T: int,
):
    rng = random.Random(seed)
    dist = [r for r in rows if r.get("distinguishing") == 1] or rows
    params = list(model.q_enc.parameters()) + list(model.answer.parameters())
    if train_v:
        params += list(model.V.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    model.train()
    hist = []
    for step in range(1, steps + 1):
        batch = [rng.choice(dist if rng.random() < 0.65 else rows) for _ in range(bs)]
        loss = torch.zeros((), device=device)
        ok = 0
        flow_v = 0.0
        for r in batch:
            q = r["continuation"]["query"]
            y = torch.tensor([r["label"]], device=device)
            z = encode_answer(model, r, "final", device)
            lo = model.logits(z, q)
            loss = loss + F.cross_entropy(lo, y)
            ok += int(lo.argmax(-1).item() == r["label"])
            if r.get("distinguishing") == 1:
                yw = torch.tensor([r["label_wrong"]], device=device)
                loss = loss + F.cross_entropy(
                    model.logits(encode_answer(model, r, "wrong", device), q), yw
                )
            z0 = model.encode_state(r["s0"], device)
            # use true chain length
            Tc = max(len(r["actions"]), 1)
            zT = model.flow(z0, r["actions"], T=Tc)
            cont = r["continuation"].get("actions") or []
            z_ans = model.flow(zT, cont, T=len(cont)) if cont else zT
            loss = loss + lam_task * F.cross_entropy(model.logits(z_ans, q), y)
            if lam_flow > 0:
                lf = chain_abs(model, r, device)
                loss = loss + lam_flow * lf
                flow_v += float(lf.item())
        loss = loss / bs
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 50 == 0 or step == 1:
            hist.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "acc": ok / bs,
                    "flow": flow_v / bs,
                }
            )
            print(
                f"[{tag}] step={step} L={loss.item():.4f} acc={ok/bs:.3f} flow={flow_v/bs:.5f}",
                flush=True,
            )
    return hist


def run_exp(name, kind, train, test_id, test_ood, device, steps, lr, train_v, lam_flow, lam_task, seed, eval_n, T, out_dir):
    print(f"\n======== {name} kind={kind} train_v={train_v} lam_flow={lam_flow} ========", flush=True)
    model = SuiteSMIR(kind=kind, T=T).to(device)
    hist = train_q_and_v(
        model, train, steps, lr, device, 64, seed, train_v, lam_flow, lam_task, name, T
    )
    metrics = evaluate_bundle(model, test_id, test_ood, T, eval_n, device)
    metrics["history"] = hist[-5:]
    model.save(out_dir / name)
    print(json.dumps(metrics, indent=2), flush=True)
    return {"name": name, "kind": kind, "metrics": metrics, "train_v": train_v, "lam_flow": lam_flow}


def build_contrast(suite_rows: list[dict], runs_root: Path) -> dict:
    def load_old(name, rel):
        p = runs_root / rel
        if not p.exists():
            return {"name": name, "missing": True}
        d = json.loads(p.read_text())
        final = (
            d.get("final")
            or d.get("stage_b")
            or d.get("results", {}).get("stage_b")
            or d.get("results", {}).get("scf")
            or {}
        )
        inter = final.get("intervention") or {}
        return {
            "name": name,
            "F": inter.get("branch_disagree"),
            "eta_abs": final.get("eta_abs", final.get("eta_conjugacy")),
            "tub_abs": final.get("tubular_mse_abs", final.get("tubular_mse")),
            "ood": final.get("task_ood"),
            "S_gap": final.get("sufficiency_gap"),
        }

    rows = [
        load_old("7B-SCF", "smir_main/overall.json"),
        load_old("V*", "smir_vstar/overall.json"),
        load_old("RegV-PURE", "smir_regv/overall.json"),
        load_old("RegV+", "smir_regv_plus/overall.json"),
    ]
    for r in suite_rows:
        m = r["metrics"]
        rows.append(
            {
                "name": r["name"],
                "F": m["intervention"]["branch_disagree"],
                "eta_abs": m["eta_abs"],
                "tub_abs": m["tubular_mse_abs"],
                "ood": m["task_ood"],
                "S_gap": m["sufficiency_gap"],
                "oracle": m["task_state_oracle"],
            }
        )
    return {"rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--runs-root", type=Path, default=Path("/project/peilab/why/NEW/smir/artifacts/runs"))
    ap.add_argument("--steps-e1", type=int, default=7000)
    ap.add_argument("--steps-e2", type=int, default=6000)
    ap.add_argument("--steps-e3", type=int, default=6000)
    ap.add_argument("--steps-e4", type=int, default=4000)
    ap.add_argument("--eval-n", type=int, default=300)
    ap.add_argument("--T", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--skip", type=str, default="", help="comma names to skip e.g. e4")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    t0 = time.time()
    skip = {x.strip() for x in args.skip.split(",") if x.strip()}

    train = load_jsonl(args.data_dir / "train.jsonl")
    test_id = load_jsonl(args.data_dir / "test_id.jsonl")
    test_ood = load_jsonl(args.data_dir / "test_ood.jsonl")
    print(
        f"[suite] device={device} train={len(train)} "
        f"mean_train_len={sum(r['chain_len'] for r in train)/len(train):.2f}",
        flush=True,
    )

    results = []

    # E1: hard V + joint Q, flow CE, long data → OOD
    if "e1" not in skip:
        results.append(
            run_exp(
                "E1_RegV_OOD",
                "hard",
                train,
                test_id,
                test_ood,
                device,
                args.steps_e1,
                2e-3,
                train_v=False,
                lam_flow=0.05,
                lam_task=1.0,
                seed=args.seed,
                eval_n=args.eval_n,
                T=args.T,
                out_dir=args.out_dir,
            )
        )

    # E2: soft structured learnable routing
    if "e2" not in skip:
        results.append(
            run_exp(
                "E2_SoftV",
                "soft",
                train,
                test_id,
                test_ood,
                device,
                args.steps_e2,
                2e-3,
                train_v=True,
                lam_flow=1.0,
                lam_task=0.5,
                seed=args.seed + 1,
                eval_n=args.eval_n,
                T=args.T,
                out_dir=args.out_dir,
            )
        )

    # E3: MLP + teacher force (learn conjugacy without hard ops)
    if "e3" not in skip:
        results.append(
            run_exp(
                "E3_MLP_TeacherForce",
                "mlp",
                train,
                test_id,
                test_ood,
                device,
                args.steps_e3,
                1e-3,
                train_v=True,
                lam_flow=1.0,
                lam_task=0.3,
                seed=args.seed + 2,
                eval_n=args.eval_n,
                T=args.T,
                out_dir=args.out_dir,
            )
        )

    # E4: MLP task-only ablation
    if "e4" not in skip:
        results.append(
            run_exp(
                "E4_MLP_TaskOnly",
                "mlp",
                train,
                test_id,
                test_ood,
                device,
                args.steps_e4,
                1e-3,
                train_v=True,
                lam_flow=0.0,
                lam_task=1.0,
                seed=args.seed + 3,
                eval_n=args.eval_n,
                T=args.T,
                out_dir=args.out_dir,
            )
        )

    # Gates / claims
    by = {r["name"]: r for r in results}
    e1 = by.get("E1_RegV_OOD", {}).get("metrics")
    e2 = by.get("E2_SoftV", {}).get("metrics")
    e3 = by.get("E3_MLP_TeacherForce", {}).get("metrics")
    e4 = by.get("E4_MLP_TaskOnly", {}).get("metrics")

    claims = {
        "E1_ood_ge_0_80": bool(e1 and e1["task_ood"] >= 0.80),
        "E1_eta_lt_0_02": bool(e1 and e1["eta_abs"] < 0.02),
        "E1_F_ge_0_50": bool(e1 and e1["intervention"]["branch_disagree"] >= 0.50),
        "E2_soft_eta_lt_0_05": bool(e2 and e2["eta_abs"] < 0.05),
        "E2_soft_F_ge_0_45": bool(e2 and e2["intervention"]["branch_disagree"] >= 0.45),
        "E3_mlp_tf_eta_lt_0_10": bool(e3 and e3["eta_abs"] < 0.10),
        "E3_beats_E4_eta": bool(e3 and e4 and e3["eta_abs"] < e4["eta_abs"] * 0.5),
        "E4_task_only_eta_bad": bool(e4 and e4["eta_abs"] > 0.05),
    }
    claims["pass_suite_core"] = bool(
        claims.get("E1_ood_ge_0_80")
        and claims.get("E1_eta_lt_0_02")
        and claims.get("E1_F_ge_0_50")
    )
    claims["pass_learnability"] = bool(
        claims.get("E2_soft_eta_lt_0_05") or claims.get("E3_mlp_tf_eta_lt_0_10")
    )
    claims["pass_ablation"] = bool(claims.get("E3_beats_E4_eta") and claims.get("E4_task_only_eta_bad"))

    contrast = build_contrast(results, args.runs_root)
    # pick best by (ood, -eta)
    best = None
    if results:
        best = max(
            results,
            key=lambda r: (
                r["metrics"]["task_ood"],
                -r["metrics"]["eta_abs"],
                r["metrics"]["intervention"]["branch_disagree"],
            ),
        )

    summary = {
        "meta": {
            "experiment": "SMIR-Venue-Suite",
            "seconds": round(time.time() - t0, 2),
            "host": os.uname().nodename,
            "device": str(device),
            "data": str(args.data_dir),
        },
        "claims": claims,
        "best": {"name": best["name"], "metrics": best["metrics"]} if best else None,
        "experiments": [
            {
                "name": r["name"],
                "kind": r["kind"],
                "eta": r["metrics"]["eta_abs"],
                "tub": r["metrics"]["tubular_mse_abs"],
                "F": r["metrics"]["intervention"]["branch_disagree"],
                "ood": r["metrics"]["task_ood"],
                "S_gap": r["metrics"]["sufficiency_gap"],
                "length_curve": r["metrics"]["length_curve"],
            }
            for r in results
        ],
        "contrast": contrast,
        "full": results,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_dir / "contrast_table.json").write_text(json.dumps(contrast, indent=2), encoding="utf-8")
    print(json.dumps({"claims": claims, "experiments": summary["experiments"], "best": summary["best"]["name"] if best else None}, indent=2), flush=True)

    if not (claims["pass_suite_core"] or (claims["pass_learnability"] and claims.get("E1_eta_lt_0_02"))):
        print("[WARN] suite core not fully passed", flush=True)
        raise SystemExit(2)
    print("[OK] venue suite produced usable certificate block", flush=True)


if __name__ == "__main__":
    main()
