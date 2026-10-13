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
    """Soft mixture over {add,sub,mul}; annealable bias toward true op."""

    def __init__(self, state_scale=STATE_SCALE, op_bias: float = 2.0):
        super().__init__()
        self.state_scale = state_scale
        self.z_clip = 200.0 / state_scale
        self.op_bias = float(op_bias)
        self.route = nn.Sequential(
            nn.Linear(N_REG + 4 + 3 * N_REG, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )

    def set_op_bias(self, bias: float):
        self.op_bias = float(bias)

    def route_logits(self, z, op, i, j, k):
        op_oh = F.one_hot(op.clamp(0, 3), 4).float()
        i_oh = F.one_hot(i.clamp(0, N_REG - 1), N_REG).float()
        j_oh = F.one_hot(j.clamp(0, N_REG - 1), N_REG).float()
        k_oh = F.one_hot(k.clamp(0, N_REG - 1), N_REG).float()
        logits = self.route(torch.cat([z, op_oh, i_oh, j_oh, k_oh], dim=-1))
        logits = logits + self.op_bias * F.one_hot(op.clamp(0, 2), 3).float()
        return logits

    def forward(self, z, op, i, j, k):
        b = z.size(0)
        ar = torch.arange(b, device=z.device)
        xi, xj = z[ar, i], z[ar, j]
        add, sub, mul = xi + xj, xi - xj, self.state_scale * xi * xj
        w = F.softmax(self.route_logits(z, op, i, j, k), dim=-1)
        val = w[:, 0] * add + w[:, 1] * sub + w[:, 2] * mul
        val = val.clamp(-self.z_clip, self.z_clip)
        z2 = z.clone()
        z2[ar, k] = val
        return z2


class DiscreteRegisterV(nn.Module):
    """Learn discrete op routing; execute with hard arithmetic (STE)."""

    def __init__(self, state_scale=STATE_SCALE, tau: float = 1.0):
        super().__init__()
        self.state_scale = state_scale
        self.z_clip = 200.0 / state_scale
        self.tau = float(tau)
        self.route = nn.Sequential(
            nn.Linear(N_REG + 4 + 3 * N_REG, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )

    def set_tau(self, tau: float):
        self.tau = float(tau)

    def route_logits(self, z, op, i, j, k):
        op_oh = F.one_hot(op.clamp(0, 3), 4).float()
        i_oh = F.one_hot(i.clamp(0, N_REG - 1), N_REG).float()
        j_oh = F.one_hot(j.clamp(0, N_REG - 1), N_REG).float()
        k_oh = F.one_hot(k.clamp(0, N_REG - 1), N_REG).float()
        return self.route(torch.cat([z, op_oh, i_oh, j_oh, k_oh], dim=-1))

    def forward(self, z, op, i, j, k):
        b = z.size(0)
        ar = torch.arange(b, device=z.device)
        xi, xj = z[ar, i], z[ar, j]
        cands = torch.stack(
            [xi + xj, xi - xj, self.state_scale * xi * xj], dim=-1
        )  # [B,3]
        logits = self.route_logits(z, op, i, j, k)
        # hard=True already applies STE in PyTorch
        w = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)
        val = (cands * w).sum(-1).clamp(-self.z_clip, self.z_clip)
        z2 = z.clone()
        z2[ar, k] = val
        return z2


GOP2ID = {"relax": 0, "set": 1, "null": 2}


def graph_action_tensors(action: dict, device):
    op = torch.tensor([GOP2ID.get(action["op"], 2)], dtype=torch.long, device=device)
    i = torch.tensor([int(action["i"])], dtype=torch.long, device=device)
    j = torch.tensor([int(action["j"])], dtype=torch.long, device=device)
    k = torch.tensor([int(action.get("k", 0))], dtype=torch.long, device=device)
    return op, i, j, k


class GraphHardV(nn.Module):
    """Exact graph distance dynamics in chart coords z=s/scale."""

    def __init__(self, state_scale=STATE_SCALE):
        super().__init__()
        self.state_scale = state_scale
        self.step = 1.0 / state_scale

    def forward(self, z, op, i, j, k):
        b = z.size(0)
        ar = torch.arange(b, device=z.device)
        z2 = z.clone()
        # op: 0=relax, 1=set, 2=null
        is_set = (op == 1).float().view(-1)
        is_rel = (op == 0).float().view(-1)
        # set i -> 0
        z2[ar, i] = (1.0 - is_set) * z2[ar, i]
        # relax j <- min(j, i+step)
        cand = z[ar, i] + self.step
        z2[ar, j] = is_rel * torch.minimum(z[ar, j], cand) + (1.0 - is_rel) * z2[ar, j]
        return z2


class GraphSoftV(nn.Module):
    """Soft mix over {relax, set, identity}-like candidates."""

    def __init__(self, state_scale=STATE_SCALE, op_bias: float = 2.0):
        super().__init__()
        self.state_scale = state_scale
        self.step = 1.0 / state_scale
        self.op_bias = float(op_bias)
        self.route = nn.Sequential(
            nn.Linear(N_REG + 3 + 2 * N_REG, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )

    def set_op_bias(self, b):
        self.op_bias = float(b)

    def route_logits(self, z, op, i, j, k):
        op_oh = F.one_hot(op.clamp(0, 2), 3).float()
        i_oh = F.one_hot(i.clamp(0, N_REG - 1), N_REG).float()
        j_oh = F.one_hot(j.clamp(0, N_REG - 1), N_REG).float()
        logits = self.route(torch.cat([z, op_oh, i_oh, j_oh], dim=-1))
        return logits + self.op_bias * F.one_hot(op.clamp(0, 2), 3).float()

    def forward(self, z, op, i, j, k):
        b = z.size(0)
        ar = torch.arange(b, device=z.device)
        w = F.softmax(self.route_logits(z, op, i, j, k), dim=-1)
        z_id = z
        z_set = z.clone()
        z_set[ar, i] = 0.0
        z_rel = z.clone()
        z_rel[ar, j] = torch.minimum(z[ar, j], z[ar, i] + self.step)
        return w[:, 0:1] * z_rel + w[:, 1:2] * z_set + w[:, 2:3] * z_id


class GraphDiscreteV(nn.Module):
    def __init__(self, state_scale=STATE_SCALE, tau: float = 1.0):
        super().__init__()
        self.state_scale = state_scale
        self.step = 1.0 / state_scale
        self.tau = float(tau)
        self.route = nn.Sequential(
            nn.Linear(N_REG + 3 + 2 * N_REG, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )

    def set_tau(self, tau):
        self.tau = float(tau)

    def route_logits(self, z, op, i, j, k):
        op_oh = F.one_hot(op.clamp(0, 2), 3).float()
        i_oh = F.one_hot(i.clamp(0, N_REG - 1), N_REG).float()
        j_oh = F.one_hot(j.clamp(0, N_REG - 1), N_REG).float()
        return self.route(torch.cat([z, op_oh, i_oh, j_oh], dim=-1))

    def forward(self, z, op, i, j, k):
        b = z.size(0)
        ar = torch.arange(b, device=z.device)
        logits = self.route_logits(z, op, i, j, k)
        w = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)
        z_id = z
        z_set = z.clone()
        z_set[ar, i] = 0.0
        z_rel = z.clone()
        z_rel[ar, j] = torch.minimum(z[ar, j], z[ar, i] + self.step)
        return w[:, 0:1] * z_rel + w[:, 1:2] * z_set + w[:, 2:3] * z_id


class SuiteSMIR(nn.Module):
    def __init__(self, kind: str = "hard", T: int = 5, q_hidden: int = 128, soft_bias: float = 2.0):
        super().__init__()
        assert kind in {"hard", "soft", "mlp", "discrete", "ghard", "gsoft", "gdisc"}
        self.kind = kind
        self.z_dim = N_REG
        self.T = T
        self.state_scale = STATE_SCALE
        self.is_graph = kind.startswith("g")
        if kind == "hard":
            self.V = StructuredRegisterV(residual=False, state_scale=STATE_SCALE)
        elif kind == "soft":
            self.V = SoftRegisterV(state_scale=STATE_SCALE, op_bias=soft_bias)
        elif kind == "discrete":
            self.V = DiscreteRegisterV(state_scale=STATE_SCALE)
        elif kind == "ghard":
            self.V = GraphHardV(state_scale=STATE_SCALE)
        elif kind == "gsoft":
            self.V = GraphSoftV(state_scale=STATE_SCALE, op_bias=soft_bias)
        elif kind == "gdisc":
            self.V = GraphDiscreteV(state_scale=STATE_SCALE)
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
        if self.is_graph:
            op, i, j, k = graph_action_tensors(action, z.device)
        else:
            op, i, j, k = action_tensors(action, z.device)
        return self.V(z, op, i, j, k)

    def flow(self, z0, actions, T=None):
        T = T if T is not None else self.T
        z = z0
        null = {"op": "null", "i": 0, "j": 0, "k": 0}
        for t in range(T):
            if t < len(actions):
                z = self.one_step(z, actions[t])
            else:
                z = self.one_step(z, null)
        return z

    def logits(self, z, query):
        return self.answer(torch.cat([z, self.encode_query(query, z.device)], dim=-1))

    def save(self, path):
        from pathlib import Path

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "kind": self.kind, "T": self.T}, path / "suite.pt")
