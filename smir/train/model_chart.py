#!/usr/bin/env python3
"""Structured semantic-chart SMIR (no 7B last-token bottleneck).

Chart: registers ℝ^5 → z ∈ ℝ^d
  - default: free MLP (can be ill-conditioned for dynamics)
  - iso_chart=True: near-isometric Linear+LayerNorm on scaled registers
Flow:  V(z, action) with full (op,i,j,k)
Readout: Q(z, query) — label depends on continuation query (critical for F).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

OP2ID = {"add": 0, "sub": 1, "mul": 2, "null": 3}
KIND2ID = {"gt": 0, "eq0": 1, "cmp": 2}
N_REG = 5
STATE_SCALE = 40.0


def state_tensor(state, device) -> torch.Tensor:
    return torch.tensor([state], dtype=torch.float32, device=device)


def action_tensors(action: dict, device):
    op = torch.tensor([OP2ID.get(action["op"], 3)], dtype=torch.long, device=device)
    i = torch.tensor([int(action["i"])], dtype=torch.long, device=device)
    j = torch.tensor([int(action["j"])], dtype=torch.long, device=device)
    k = torch.tensor([int(action["k"])], dtype=torch.long, device=device)
    return op, i, j, k


def query_tensor(query: dict, device) -> torch.Tensor:
    kind = KIND2ID.get(query["kind"], 0)
    reg = int(query["reg"])
    thr = float(query["thr"])
    feat = torch.zeros(1, 3 + N_REG + 1 + N_REG, device=device)
    feat[0, kind] = 1.0
    feat[0, 3 + reg] = 1.0
    feat[0, 3 + N_REG] = thr / 40.0
    if query["kind"] == "cmp":
        other = int(thr)
        other = max(0, min(N_REG - 1, other))
        feat[0, 3 + N_REG + 1 + other] = 1.0
    return feat


class SemanticFlow(nn.Module):
    def __init__(self, z_dim: int, n_ops: int = 4):
        super().__init__()
        self.op_emb = nn.Embedding(n_ops, 32)
        self.idx_emb = nn.Embedding(N_REG, 16)
        self.net = nn.Sequential(
            nn.Linear(z_dim + 32 + 16 * 3, z_dim * 2),
            nn.SiLU(),
            nn.Linear(z_dim * 2, z_dim),
        )

    def forward(self, z, op, i, j, k):
        u = torch.cat(
            [self.op_emb(op), self.idx_emb(i), self.idx_emb(j), self.idx_emb(k)], dim=-1
        )
        return z + self.net(torch.cat([z, u], dim=-1))


class ChartSMIR(nn.Module):
    def __init__(self, z_dim: int = 64, T: int = 4, iso_chart: bool = False):
        super().__init__()
        self.z_dim = z_dim
        self.T = T
        self.iso_chart = iso_chart
        self.state_scale = STATE_SCALE
        q_dim = 3 + N_REG + 1 + N_REG
        if iso_chart:
            # Near-isometric chart: scaled registers → orthogonal linear → LayerNorm
            lin = nn.Linear(N_REG, z_dim, bias=False)
            nn.init.orthogonal_(lin.weight)
            self.E = nn.Sequential(lin, nn.LayerNorm(z_dim))
        else:
            self.E = nn.Sequential(
                nn.Linear(N_REG, z_dim * 2),
                nn.SiLU(),
                nn.Linear(z_dim * 2, z_dim),
            )
        self.V = SemanticFlow(z_dim)
        self.q_enc = nn.Sequential(nn.Linear(q_dim, z_dim), nn.SiLU())
        self.answer = nn.Sequential(
            nn.Linear(z_dim * 2, z_dim),
            nn.SiLU(),
            nn.Linear(z_dim, 2),
        )

    def encode_state(self, state, device=None) -> torch.Tensor:
        if not torch.is_tensor(state):
            device = device or next(self.parameters()).device
            state = state_tensor(state, device)
        if self.iso_chart:
            state = state / self.state_scale
        return self.E(state)

    def encode_query(self, query: dict, device=None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        return self.q_enc(query_tensor(query, device))

    def one_step(self, z, action: dict) -> torch.Tensor:
        device = z.device
        op, i, j, k = action_tensors(action, device)
        return self.V(z, op, i, j, k)

    def flow(self, z0, actions: list[dict], T: int | None = None) -> torch.Tensor:
        T = T if T is not None else self.T
        z = z0
        for t in range(T):
            if t < len(actions):
                z = self.one_step(z, actions[t])
            else:
                z = self.one_step(z, {"op": "null", "i": 0, "j": 0, "k": 0})
        return z

    def logits(self, z, query: dict) -> torch.Tensor:
        q = self.encode_query(query, device=z.device)
        return self.answer(torch.cat([z, q], dim=-1))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "z_dim": self.z_dim,
                "T": self.T,
                "iso_chart": self.iso_chart,
            },
            path / "chart.pt",
        )

    @classmethod
    def load(cls, path: str | Path, device=None):
        path = Path(path)
        ckpt = torch.load(path / "chart.pt", map_location=device or "cpu", weights_only=True)
        model = cls(
            z_dim=int(ckpt["z_dim"]),
            T=int(ckpt["T"]),
            iso_chart=bool(ckpt.get("iso_chart", False)),
        )
        model.load_state_dict(ckpt["state_dict"])
        if device is not None:
            model.to(device)
        return model
