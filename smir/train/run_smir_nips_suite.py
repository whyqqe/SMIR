#!/usr/bin/env python3
"""NeurIPS/ICML-oriented validation of new_theory.md continuation consistency.

Phases:
  A  Feasibility: A1 oracle-T, A2 learnable-T, A3 sampling ablation
  B  Identifiability: B1 length curves, B2 nonseparating queries, B3 error vs length
  C  Bottleneck necessity: C1 replace, C2 leak/bypass, C3 cfact follow
  D  Scope: D1 structure spectrum, D2 graph domain, D3 wrong charts
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from model_suite import SuiteSMIR
from nips_common import (
    FULL_MODES,
    SHORT_MODES,
    ChartWrap,
    LeakSuite,
    apply_nonseparating_queries,
    continuation_error_by_length,
    eval_replace_battery,
    split_by_len,
    theory_core_gates,
    train_l_continuation,
)
from suite_common import evaluate_bundle, load_jsonl, set_seed


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def train_curriculum(model, train, short, args, device, tag, freeze_v, modes, steps_scale=1.0):
    s1 = max(int(args.steps_1 * steps_scale), 50)
    s2 = max(int(args.steps_2 * steps_scale), 50)
    train_l_continuation(
        model, train, short, s1, 2e-3, device, args.batch, args.seed,
        0.85, True, modes, f"{tag}-A", freeze_v=freeze_v,
    )
    train_l_continuation(
        model, train, short, s2, 1e-3, device, args.batch, args.seed + 1,
        0.75, False, modes, f"{tag}-B", freeze_v=freeze_v,
    )
    if not freeze_v:
        sv = max(int(args.steps_v * steps_scale), 50)
        train_l_continuation(
            model, train, short, sv, 5e-4, device, args.batch, args.seed + 2,
            0.80, False, modes, f"{tag}-C", freeze_v=False,
        )


def full_eval(model, test_id, test_ood, T, n, device, gate_f, gate_ood):
    bundle = evaluate_bundle(model, test_id, test_ood, T, n, device)
    replace = eval_replace_battery(model, test_id, n, device)
    length = continuation_error_by_length(model, test_id + test_ood, n, device)
    gates = theory_core_gates(bundle, replace, gate_f=gate_f, gate_ood=gate_ood)
    return {"bundle": bundle, "replace": replace, "length_acc": length, "gates": gates}


def run_phase_a(args, train, short, test_id, test_ood, device, out):
    """A1 hard oracle-T; A2 discrete learnable; A3 short-only vs full sampling."""
    results = {}

    # A1
    print("==== A1 oracle-T (hard, freeze V) ====", flush=True)
    m = SuiteSMIR(kind="hard", T=args.T).to(device)
    train_curriculum(m, train, short, args, device, "A1", True, FULL_MODES)
    ev = full_eval(m, test_id, test_ood, args.T, args.eval_n, device, args.gate_f, args.gate_ood)
    m.save(out / "A1_hard" / "final")
    results["A1_hard"] = ev
    dump(out / "A1_hard" / "overall.json", {"gates": ev["gates"], "replace": ev["replace"], "length_acc": ev["length_acc"]})

    # A2
    print("==== A2 learnable-T (discrete, L_cont only) ====", flush=True)
    m2 = SuiteSMIR(kind="discrete", T=args.T).to(device)
    # warm-start Q from A1
    m2.q_enc.load_state_dict(m.q_enc.state_dict())
    m2.answer.load_state_dict(m.answer.state_dict())
    train_curriculum(m2, train, short, args, device, "A2", False, FULL_MODES)
    ev2 = full_eval(m2, test_id, test_ood, args.T, args.eval_n, device, args.gate_f, args.gate_ood)
    m2.save(out / "A2_discrete" / "final")
    results["A2_discrete"] = ev2
    dump(out / "A2_discrete" / "overall.json", {"gates": ev2["gates"], "replace": ev2["replace"], "length_acc": ev2["length_acc"]})

    # A3a short-only sampling
    print("==== A3a short-only sampling (discrete) ====", flush=True)
    m3a = SuiteSMIR(kind="discrete", T=args.T).to(device)
    m3a.q_enc.load_state_dict(m.q_enc.state_dict())
    m3a.answer.load_state_dict(m.answer.state_dict())
    train_curriculum(m3a, train, short, args, device, "A3a", False, SHORT_MODES)
    ev3a = full_eval(m3a, test_id, test_ood, args.T, args.eval_n, device, args.gate_f, args.gate_ood)
    results["A3a_short_sample"] = ev3a
    dump(out / "A3a_short_sample" / "overall.json", {"gates": ev3a["gates"], "replace": ev3a["replace"], "length_acc": ev3a["length_acc"]})

    # A3b full sampling — reuse A2 numbers as full coverage
    results["A3b_full_sample"] = ev2
    a3_claim = {
        "full_beats_short_on_F": ev2["gates"]["F_flow"] > ev3a["gates"]["F_flow"] + 0.05,
        "full_beats_short_on_cfact": ev2["replace"]["cfact_follow_both"] > ev3a["replace"]["cfact_follow_both"] + 0.05,
        "full_beats_short_on_ood": ev2["gates"]["ood"] >= ev3a["gates"]["ood"] - 0.02,
        "pass_A3_sampling_matters": (
            ev2["gates"]["F_flow"] > ev3a["gates"]["F_flow"] + 0.05
            or ev2["replace"]["cfact_follow_both"] > ev3a["replace"]["cfact_follow_both"] + 0.05
        ),
    }
    results["A3_claims"] = a3_claim
    dump(out / "A3_claims.json", a3_claim)

    claims = {
        "pass_A1": bool(ev["gates"]["pass_core"]),
        "pass_A2": bool(ev2["gates"]["pass_core"]),
        "pass_A3_sampling_matters": bool(a3_claim["pass_A3_sampling_matters"]),
        "pass_phase_A": bool(ev["gates"]["pass_core"]) and bool(a3_claim["pass_A3_sampling_matters"]),
    }
    # keep references for later phases
    results["_models"] = {"A1": m, "A2": m2}
    results["claims"] = claims
    print(json.dumps({"phase_A": claims}, indent=2), flush=True)
    return results


def run_phase_b(args, train, short, test_id, test_ood, device, out, phase_a):
    """B1 length curves from A1/A2; B2 nonseparating train; B3 error-vs-length table."""
    results = {}
    m1 = phase_a["_models"]["A1"]
    m2 = phase_a["_models"]["A2"]

    results["B1_length_A1"] = phase_a["A1_hard"]["length_acc"]
    results["B1_length_A2"] = phase_a["A2_discrete"]["length_acc"]
    dump(out / "B1_length_curves.json", {
        "A1_hard": results["B1_length_A1"],
        "A2_discrete": results["B1_length_A2"],
        "A3a_short": phase_a["A3a_short_sample"]["length_acc"],
    })

    # B2: train discrete on nonseparating queries, eval on real test
    print("==== B2 nonseparating-query training ====", flush=True)
    weak_train = apply_nonseparating_queries(train)
    weak_short = split_by_len(weak_train, 4)
    mb = SuiteSMIR(kind="discrete", T=args.T).to(device)
    train_curriculum(mb, weak_train, weak_short, args, device, "B2", False, FULL_MODES, steps_scale=0.8)
    evb = full_eval(mb, test_id, test_ood, args.T, args.eval_n, device, args.gate_f, args.gate_ood)
    results["B2_nonsep"] = evb
    b2_claim = {
        "nonsep_fails_core": not bool(evb["gates"]["pass_core"]),
        "nonsep_F_below_A2": evb["gates"]["F_flow"] < phase_a["A2_discrete"]["gates"]["F_flow"] - 0.08,
        "pass_B2_identifiability": (
            (not bool(evb["gates"]["pass_core"]))
            and evb["gates"]["F_flow"] < phase_a["A2_discrete"]["gates"]["F_flow"] - 0.05
        ),
    }
    results["B2_claims"] = b2_claim
    dump(out / "B2_nonsep" / "overall.json", {"gates": evb["gates"], "claims": b2_claim})

    # B3: single-step vs long — report length table + decay score
    def decay_score(length_acc):
        vals = [(int(k), v) for k, v in length_acc.items() if v is not None]
        if len(vals) < 2:
            return 0.0
        vals.sort()
        return float(vals[0][1] - vals[-1][1])

    results["B3"] = {
        "A1_decay": decay_score(results["B1_length_A1"]),
        "A2_decay": decay_score(results["B1_length_A2"]),
        "A3a_decay": decay_score(phase_a["A3a_short_sample"]["length_acc"]),
        "short_decays_more": decay_score(phase_a["A3a_short_sample"]["length_acc"])
        > decay_score(results["B1_length_A2"]) + 0.03,
    }
    dump(out / "B3_error_propagation.json", results["B3"])

    claims = {
        "pass_B1_reported": True,
        "pass_B2_identifiability": bool(b2_claim["pass_B2_identifiability"]),
        "pass_B3_short_decays_more": bool(results["B3"]["short_decays_more"]),
        "pass_phase_B": bool(b2_claim["pass_B2_identifiability"]),
    }
    results["claims"] = claims
    print(json.dumps({"phase_B": claims}, indent=2), flush=True)
    return results


def run_phase_c(args, train, short, test_id, test_ood, device, out, phase_a):
    """C1 replace already in gates; C2 leak bypass; C3 cfact follow comparison."""
    results = {}
    m2 = phase_a["_models"]["A2"]

    # C1 from A2 replace battery
    c1 = {
        "replace_acc": phase_a["A2_discrete"]["replace"]["replace_midstate_acc"],
        "full_acc": phase_a["A2_discrete"]["replace"]["full_rollout_acc"],
        "pass_C1_replace": phase_a["A2_discrete"]["replace"]["replace_midstate_acc"]
        >= phase_a["A2_discrete"]["replace"]["full_rollout_acc"] - 0.05
        and phase_a["A2_discrete"]["replace"]["replace_midstate_acc"] >= 0.70,
    }
    results["C1"] = c1
    dump(out / "C1_replace.json", c1)

    # C2: train with leak, eval with and without leak
    print("==== C2 leak / bypass bottleneck ====", flush=True)
    base = SuiteSMIR(kind="discrete", T=args.T).to(device)
    leak = LeakSuite(base, leak_strength=1.0).to(device)
    # train with leak enabled
    s1, s2, sv = args.steps_1, args.steps_2, args.steps_v
    train_l_continuation(
        leak, train, short, s1, 2e-3, device, args.batch, args.seed + 20,
        0.85, True, FULL_MODES, "C2-A", freeze_v=False, leak_row=True,
    )
    train_l_continuation(
        leak, train, short, s2, 1e-3, device, args.batch, args.seed + 21,
        0.75, False, FULL_MODES, "C2-B", freeze_v=False, leak_row=True,
    )
    train_l_continuation(
        leak, train, short, sv, 5e-4, device, args.batch, args.seed + 22,
        0.80, False, FULL_MODES, "C2-C", freeze_v=False, leak_row=True,
    )

    # eval WITH leak on test (fake success path): manually score using leak rows
    leak.eval()
    ok_leak = 0
    for r in test_id[: args.eval_n]:
        leak.set_leak_row(r)
        z = encode_for_eval(leak, r, device)
        ok_leak += int(leak.logits(z, r["continuation"]["query"]).argmax(-1).item() == r["label"])
    leak.set_leak_row(None)
    acc_with_leak = ok_leak / max(min(args.eval_n, len(test_id)), 1)

    # eval WITHOUT leak (true bottleneck)
    ev_noleak = full_eval(leak, test_id, test_ood, args.T, args.eval_n, device, args.gate_f, args.gate_ood)
    c2 = {
        "acc_with_leak": acc_with_leak,
        "gates_no_leak": ev_noleak["gates"],
        "replace_no_leak": ev_noleak["replace"],
        "pass_C2_leak_hurts_bottleneck": (
            acc_with_leak > ev_noleak["gates"]["task_id"] + 0.05
            or (not bool(ev_noleak["gates"]["pass_core"]))
        )
        and acc_with_leak >= 0.75,
    }
    # stronger: with-leak high but F/replace without leak clearly worse than A2
    c2["pass_C2_strong"] = (
        acc_with_leak >= 0.80
        and ev_noleak["gates"]["F_flow"] < phase_a["A2_discrete"]["gates"]["F_flow"] - 0.05
    )
    results["C2"] = c2
    dump(out / "C2_leak" / "overall.json", c2)

    # C3: cfact follow A2 vs A3a
    c3 = {
        "A2_cfact": phase_a["A2_discrete"]["replace"]["cfact_follow_both"],
        "A3a_cfact": phase_a["A3a_short_sample"]["replace"]["cfact_follow_both"],
        "A1_cfact": phase_a["A1_hard"]["replace"]["cfact_follow_both"],
        "pass_C3_full_covers_cfact": phase_a["A2_discrete"]["replace"]["cfact_follow_both"]
        > phase_a["A3a_short_sample"]["replace"]["cfact_follow_both"] + 0.05
        or phase_a["A2_discrete"]["replace"]["cfact_follow_both"] >= 0.35,
    }
    results["C3"] = c3
    dump(out / "C3_cfact.json", c3)

    claims = {
        "pass_C1_replace": bool(c1["pass_C1_replace"]),
        "pass_C2_leak": bool(c2["pass_C2_leak_hurts_bottleneck"] or c2["pass_C2_strong"]),
        "pass_C3_cfact": bool(c3["pass_C3_full_covers_cfact"]),
        "pass_phase_C": bool(c1["pass_C1_replace"]) and bool(c2.get("pass_C2_leak_hurts_bottleneck") or c2.get("pass_C2_strong")),
    }
    results["claims"] = claims
    # keep A2 model alive
    results["_models"] = {"A2": m2}
    print(json.dumps({"phase_C": claims}, indent=2), flush=True)
    return results


def encode_for_eval(model, row, device):
    # latent rollout for fair comparison when leak disabled
    z0 = model.encode_state(row["s0"], device)
    z = model.flow(z0, row["actions"], T=max(len(row["actions"]), 1))
    cont = row["continuation"].get("actions") or []
    return model.flow(z, cont, T=len(cont)) if cont else z


def run_phase_d(args, train, short, test_id, test_ood, device, out, phase_a, graph_data: Path | None):
    results = {}

    # D1 mlp spectrum
    print("==== D1 structure spectrum (mlp) ====", flush=True)
    mlp = SuiteSMIR(kind="mlp", T=args.T).to(device)
    train_curriculum(mlp, train, short, args, device, "D1mlp", False, FULL_MODES, steps_scale=1.0)
    evm = full_eval(mlp, test_id, test_ood, args.T, args.eval_n, device, args.gate_f, args.gate_ood)
    results["D1_mlp"] = evm
    d1 = {
        "hard_F": phase_a["A1_hard"]["gates"]["F_flow"],
        "disc_F": phase_a["A2_discrete"]["gates"]["F_flow"],
        "mlp_F": evm["gates"]["F_flow"],
        "structured_beats_mlp": phase_a["A1_hard"]["gates"]["F_flow"] > evm["gates"]["F_flow"] + 0.08,
        "pass_D1": phase_a["A1_hard"]["gates"]["F_flow"] > evm["gates"]["F_flow"] + 0.08,
    }
    dump(out / "D1_spectrum.json", {"gates_mlp": evm["gates"], "claims": d1})

    # D3 wrong charts on A1
    print("==== D3 wrong charts ====", flush=True)
    m1 = phase_a["_models"]["A1"]
    chart_rows = {}
    for mode in ["legal", "scale2", "proj", "zero"]:
        wrap = ChartWrap(m1, mode, seed=args.seed)
        mm = evaluate_bundle(wrap, test_id, test_ood, args.T, args.eval_n, device)
        chart_rows[mode] = {
            "F": mm["intervention"]["branch_disagree"],
            "ood": mm["task_ood"],
            "task_id": mm["task_id"],
        }
    d3 = {
        "rows": chart_rows,
        "proj_hurts": chart_rows["proj"]["F"] < chart_rows["legal"]["F"] - 0.08,
        "zero_hurts": chart_rows["zero"]["F"] < 0.25,
        "pass_D3": chart_rows["proj"]["F"] < chart_rows["legal"]["F"] - 0.08,
    }
    results["D3"] = d3
    dump(out / "D3_charts.json", d3)

    # D2 graph domain
    d2 = {"skipped": True}
    if graph_data is not None and (graph_data / "train.jsonl").exists() and not args.skip_graph:
        print("==== D2 graph domain (ghard + gdisc) ====", flush=True)
        gtrain = load_jsonl(graph_data / "train.jsonl")
        gid = load_jsonl(graph_data / "test_id.jsonl")
        good = load_jsonl(graph_data / "test_ood.jsonl")
        gshort = split_by_len(gtrain, 4)
        gh = SuiteSMIR(kind="ghard", T=args.T_graph).to(device)
        train_curriculum(gh, gtrain, gshort, args, device, "D2ghard", True, FULL_MODES, steps_scale=0.9)
        evh = full_eval(gh, gid, good, args.T_graph, args.eval_n, device, args.gate_f, max(args.gate_ood - 0.05, 0.70))
        gd = SuiteSMIR(kind="gdisc", T=args.T_graph).to(device)
        gd.q_enc.load_state_dict(gh.q_enc.state_dict())
        gd.answer.load_state_dict(gh.answer.state_dict())
        train_curriculum(gd, gtrain, gshort, args, device, "D2gdisc", False, FULL_MODES, steps_scale=0.9)
        evd = full_eval(gd, gid, good, args.T_graph, args.eval_n, device, args.gate_f, max(args.gate_ood - 0.05, 0.70))
        d2 = {
            "skipped": False,
            "ghard": evh["gates"],
            "gdisc": evd["gates"],
            "pass_D2": bool(evh["gates"]["pass_core"]) or bool(evd["gates"]["pass_core"]),
        }
        gh.save(out / "D2_ghard" / "final")
        dump(out / "D2_graph.json", d2)
    results["D2"] = d2
    results["D1"] = d1

    claims = {
        "pass_D1_spectrum": bool(d1["pass_D1"]),
        "pass_D2_graph": bool(d2.get("pass_D2", False)) if not d2.get("skipped") else None,
        "pass_D3_charts": bool(d3["pass_D3"]),
        "pass_phase_D": bool(d1["pass_D1"]) and bool(d3["pass_D3"]),
    }
    results["claims"] = claims
    print(json.dumps({"phase_D": claims}, indent=2), flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reg-data", type=Path, required=True)
    ap.add_argument("--graph-data", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phases", type=str, default="A,B,C,D")
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

    train = load_jsonl(args.reg_data / "train.jsonl")
    test_id = load_jsonl(args.reg_data / "test_id.jsonl")
    test_ood = load_jsonl(args.reg_data / "test_ood.jsonl")
    short = split_by_len(train, 4)
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    print(f"[NIPS] device={device} seed={args.seed} phases={phases}", flush=True)

    all_claims = {}
    phase_a = phase_b = phase_c = phase_d = None

    if "A" in phases:
        phase_a = run_phase_a(args, train, short, test_id, test_ood, device, args.out_dir / "A")
        all_claims.update({f"A/{k}": v for k, v in phase_a["claims"].items()})
    if "B" in phases:
        if phase_a is None:
            raise RuntimeError("Phase B requires A in the same run")
        phase_b = run_phase_b(args, train, short, test_id, test_ood, device, args.out_dir / "B", phase_a)
        all_claims.update({f"B/{k}": v for k, v in phase_b["claims"].items()})
    if "C" in phases:
        if phase_a is None:
            raise RuntimeError("Phase C requires A in the same run")
        phase_c = run_phase_c(args, train, short, test_id, test_ood, device, args.out_dir / "C", phase_a)
        all_claims.update({f"C/{k}": v for k, v in phase_c["claims"].items()})
    if "D" in phases:
        if phase_a is None:
            raise RuntimeError("Phase D requires A in the same run")
        phase_d = run_phase_d(
            args, train, short, test_id, test_ood, device, args.out_dir / "D", phase_a, args.graph_data
        )
        all_claims.update({f"D/{k}": v for k, v in phase_d["claims"].items()})

    # Top-level NeurIPS-oriented feasibility pack for this seed
    pass_nips_core = bool(all_claims.get("A/pass_phase_A")) and bool(all_claims.get("B/pass_phase_B")) and bool(
        all_claims.get("C/pass_phase_C")
    )
    pass_nips_strong = pass_nips_core and bool(all_claims.get("D/pass_phase_D"))

    summary = {
        "meta": {
            "experiment": "SMIR-NIPS-ContinuationTheory",
            "theory": "new_theory.md",
            "seed": args.seed,
            "seconds": round(time.time() - t0, 2),
            "host": os.uname().nodename,
            "phases": phases,
            "objective": "L_continuation_only",
        },
        "claims": {
            **all_claims,
            "pass_nips_core": pass_nips_core,  # A+B+C
            "pass_nips_strong": pass_nips_strong,  # A+B+C+D
        },
        "A": None if phase_a is None else {
            "claims": phase_a["claims"],
            "A1": phase_a["A1_hard"]["gates"],
            "A2": phase_a["A2_discrete"]["gates"],
            "A3": phase_a["A3_claims"],
        },
        "B": None if phase_b is None else {"claims": phase_b["claims"], "B2": phase_b["B2_claims"], "B3": phase_b["B3"]},
        "C": None if phase_c is None else {"claims": phase_c["claims"], "C1": phase_c["C1"], "C2": {k: v for k, v in phase_c["C2"].items() if k != "gates_no_leak"}, "C3": phase_c["C3"]},
        "D": None if phase_d is None else {"claims": phase_d["claims"], "D1": phase_d["D1"], "D2": phase_d["D2"], "D3_pass": phase_d["D3"].get("pass_D3")},
    }
    # strip non-serializable
    dump(args.out_dir / "summary.json", summary)
    print(json.dumps(summary["claims"], indent=2), flush=True)
    if not pass_nips_core:
        raise SystemExit(2)
    print("[OK] NIPS-core (A+B+C) passed", flush=True)


if __name__ == "__main__":
    main()
