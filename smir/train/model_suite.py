#!/usr/bin/env python3
"""Model variants for venue suite: hard struct / soft struct / MLP on R^5."""
from __future__ import annotations

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


class MLPFlow(nn.Module):
    """Unstructured residual V on R^5 (ablation / learnable)."""

    def __init__(self, z_dim=N_REG):
        super().__init__()
        self.op_emb = nn.Embedding(4, 16)
        self.idx_emb = nn.Embedding(N_REG, 8)
        self.net = nn.Sequential(
            nn.Linear(z_dim + 16 + 24, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, z_dim),
        )

    def forward(self, z, op, i, j, k):
        u = torch.cat(
            [self.op_emb(op), self.idx_emb(i), self.idx_emb(j), self.idx_emb(k)], dim=-1
        )
        return z + self.net(torch.cat([z, u], dim=-1))


class SoftRegisterV(nn.Module):
    """Soft mixture over {add,sub,mul}; learns op routing (non-trivial)."""

    def __init__(self, state_scale=STATE_SCALE):
        super().__init__()
        self.state_scale = state_scale
        self.z_clip = 200.0 / state_scale
        self.route = nn.Sequential(
            nn.Linear(N_REG + 4 + 3 * N_REG, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )

    def forward(self, z, op, i, j, k):
        b = z.size(0)
        ar = torch.arange(b, device=z.device)
        xi, xj = z[ar, i], z[ar, j]
        add, sub, mul = xi + xj, xi - xj, self.state_scale * xi * xj
        op_oh = F.one_hot(op.clamp(0, 3), 4).float()
        i_oh = F.one_hot(i.clamp(0, N_REG - 1), N_REG).float()
        j_oh = F.one_hot(j.clamp(0, N_REG - 1), N_REG).float()
        k_oh = F.one_hot(k.clamp(0, N_REG - 1), N_REG).float()
        logits = self.route(torch.cat([z, op_oh, i_oh, j_oh, k_oh], dim=-1))
        # bias toward true op for stable start, but fully learnable
        logits = logits + 2.0 * F.one_hot(op.clamp(0, 2), 3).float()
        w = F.softmax(logits, dim=-1)
        val = w[:, 0] * add + w[:, 1] * sub + w[:, 2] * mul
        val = val.clamp(-self.z_clip, self.z_clip)
        z2 = z.clone()
        z2[ar, k] = val
        return z2


class SuiteSMIR(nn.Module):
    def __init__(self, kind: str = "hard", T: int = 5, q_hidden: int = 128):
        super().__init__()
        assert kind in {"hard", "soft", "mlp"}
        self.kind = kind
        self.z_dim = N_REG
        self.T = T
        self.state_scale = STATE_SCALE
        if kind == "hard":
            self.V = StructuredRegisterV(residual=False, state_scale=STATE_SCALE)
        elif kind == "soft":
            self.V = SoftRegisterV(state_scale=STATE_SCALE)
        else:
            self.V = MLPFlow(N_REG)
        self.q_enc = nn.Sequential(
            nn.Linear(3 + N_REG + 1 + N_REG, q_hidden),
            nn.SiLU(),
            nn.Linear(q_hidden, q_hidden),
            nn.SiLU(),
        )
        self.answer = nn.Sequential(
            nn.Linear(N_REG + q_hidden, q_hidden),
            nn.SiLU(),
            nn.Linear(q_hidden, 2),
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

    def save(self, path):
        from pathlib import Path

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "kind": self.kind, "T": self.T}, path / "suite.pt")
