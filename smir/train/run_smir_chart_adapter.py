#!/usr/bin/env python3
"""Weak-positive chart recovery (addresses defect #4 partially, without full LLM CoT).

Learn a small adapter E_phi: bag-of-features(state_text) -> R^5 to match legal chart s/c,
freeze Hard V, train Q. Compare to a frozen random text embedding (wrong chart).

Claim: recovering a legal chart restores Oracle-F; random text features do not.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_suite import SuiteSMIR
from suite_common import encode_answer, evaluate_bundle, load_jsonl, set_seed


def tok_feat(text: str, dim: int = 64) -> torch.Tensor:
    v = torch.zeros(dim)
    for i, ch in enumerate(text[:200]):
        v[(ord(ch) + 31 * i) % dim] += 1.0
    n = v.norm()
    return v / n if n > 0 else v


class TextChart(nn.Module):
    def __init__(self, base: SuiteSMIR, adapter: nn.Module | None, fixed: torch.Tensor | None):
        super().__init__()
        self.base = base
        self.adapter = adapter
        self.register_buffer("fixed", fixed if fixed is not None else torch.zeros(1))
        self.use_fixed = fixed is not None
        self.kind = base.kind
        self.z_dim = base.z_dim
        self.T = base.T

    def encode_state(self, state, device=None, text=None):
        # evaluate_bundle calls encode_state(state) without text — fall back to legal chart
        # For training we pass text explicitly via forward_train.
        return self.base.encode_state(state, device)

    def encode_from_text(self, text, device):
        feat = tok_feat(text).to(device).unsqueeze(0)
        if self.use_fixed:
            # wrong chart: random projection of text features into R^5
            return (feat @ self.fixed.to(device)) 
        return self.adapter(feat)

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--hard-ckpt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eval-n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    train = load_jsonl(args.data_dir / "train.jsonl")
    test_id = load_jsonl(args.data_dir / "test_id.jsonl")
    test_ood = load_jsonl(args.data_dir / "test_ood.jsonl")

    ckpt = torch.load(args.hard_ckpt / "suite.pt", map_location=device)
    hard = SuiteSMIR(kind="hard", T=ckpt.get("T", 5)).to(device)
    hard.load_state_dict(ckpt["state_dict"])
    for p in hard.V.parameters():
        p.requires_grad = False

    # Positive: learn adapter text->chart
    adapter = nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 5)).to(device)
    # freeze Q initially? train adapter to match legal chart + train Q
    q_params = list(hard.q_enc.parameters()) + list(hard.answer.parameters())
    opt = torch.optim.Adam(list(adapter.parameters()) + q_params, lr=1e-3)
    rng = random.Random(args.seed)
    dist = [r for r in train if r.get("distinguishing") == 1] or train

    for step in range(1, args.steps + 1):
        batch = [rng.choice(dist) for _ in range(32)]
        loss = torch.zeros((), device=device)
        for r in batch:
            text = r.get("state_final_text") or str(r["state_final"])
            z_pred = adapter(tok_feat(text).to(device).unsqueeze(0))
            z_true = hard.encode_state(r["state_final"], device)
            loss = loss + F.mse_loss(z_pred, z_true)
            q = r["continuation"]["query"]
            y = torch.tensor([r["label"]], device=device)
            loss = loss + F.cross_entropy(hard.logits(z_pred, q), y)
            if r.get("distinguishing") == 1:
                tw = r.get("state_wrong_text") or str(r["state_wrong"])
                zw = adapter(tok_feat(tw).to(device).unsqueeze(0))
                loss = loss + F.cross_entropy(
                    hard.logits(zw, q), torch.tensor([r["label_wrong"]], device=device)
                )
        loss = loss / 32
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == 1:
            print(f"[adapter] step={step} L={loss.item():.4f}", flush=True)

    # Eval positive: use adapter encodings for answer states via custom loop
    def eval_text_chart(use_adapter: bool, fixed=None):
        ok_f = ok_fol = n = 0
        rows = [r for r in test_id if r.get("distinguishing") == 1][: args.eval_n]
        for r in rows:
            q = r["continuation"]["query"]
            if use_adapter:
                z = adapter(tok_feat(r.get("state_final_text") or str(r["state_final"])).to(device).unsqueeze(0))
                zw = adapter(tok_feat(r.get("state_wrong_text") or str(r["state_wrong"])).to(device).unsqueeze(0))
            else:
                feat = tok_feat(r.get("state_final_text") or str(r["state_final"])).to(device).unsqueeze(0)
                featw = tok_feat(r.get("state_wrong_text") or str(r["state_wrong"])).to(device).unsqueeze(0)
                z = feat @ fixed
                zw = featw @ fixed
            po = hard.logits(z, q).argmax(-1).item()
            pw = hard.logits(zw, q).argmax(-1).item()
            ok_f += int(po != pw)
            ok_fol += int(po == r["label"] and pw == r["label_wrong"])
            n += 1
        # legal chart baseline oracle
        return {"Oracle-F": ok_f / max(n, 1), "follow": ok_fol / max(n, 1), "n": n}

    g = torch.Generator().manual_seed(args.seed + 99)
    fixed = torch.randn(64, 5, generator=g)
    fixed = fixed / (fixed.norm() + 1e-6)
    fixed = fixed.to(device)

    pos = eval_text_chart(True)
    neg = eval_text_chart(False, fixed=fixed)
    # legal reference
    m_legal = evaluate_bundle(hard, test_id, test_ood, 5, args.eval_n, device)
    out = {
        "meta": {"purpose": "weak-positive chart adapter vs random text chart"},
        "legal_chart_OracleF": m_legal["intervention"]["branch_disagree"],
        "adapter_recovered": pos,
        "random_text_chart": neg,
        "claims": {
            "adapter_beats_random": pos["Oracle-F"] > neg["Oracle-F"] + 0.15,
            "adapter_near_legal": pos["Oracle-F"] >= 0.40,
            "pass_weak_positive": pos["Oracle-F"] > neg["Oracle-F"] + 0.15 and pos["Oracle-F"] >= 0.40,
        },
    }
    (args.out_dir / "chart_adapter.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)
    if not out["claims"]["pass_weak_positive"]:
        raise SystemExit(2)
    print("[OK] weak-positive chart adapter passed", flush=True)


if __name__ == "__main__":
    main()
