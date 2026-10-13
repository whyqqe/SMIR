#!/usr/bin/env python3
"""Register-space SMIR chart: highest-success conjugacy architecture.

E:  z = s / scale ∈ ℝ^5   (identity chart — semantically exact)
V:  structured register ops (add/sub/mul) on coordinates (+ optional residual)
Q:  Q(z, query) as before

This uses the shadow-domain advantage: V* is known. Conjugacy becomes
near-exact when V implements f_a in the same coordinates as E.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        other = max(0, min(N_REG - 1, int(thr)))
        feat[0, 3 + N_REG + 1 + other] = 1.0
    return feat


class StructuredRegisterV(nn.Module):
    """Apply add/sub/mul on register coordinates (same space as E).

    Chart is z=s/scale. Then:
      add/sub: z'_k = z_i ± z_j
      mul:     z'_k = scale * z_i * z_j   (= (s_i*s_j)/scale)
    Matches synlogic_hard clip |s|<=200 → |z|<=200/scale.
    """

    def __init__(self, residual: bool = True, residual_scale: float = 0.05, state_scale: float = STATE_SCALE):
        super().__init__()
        self.residual_scale = residual_scale
        self.use_residual = residual
        self.state_scale = state_scale
        self.z_clip = 200.0 / state_scale
        if residual:
            self.res = nn.Sequential(
                nn.Linear(N_REG + 4 + 3 * N_REG, 64),
                nn.SiLU(),
                nn.Linear(64, N_REG),
            )
        else:
            self.res = None

    def forward(self, z: torch.Tensor, op, i, j, k) -> torch.Tensor:
        b = z.size(0)
        ar = torch.arange(b, device=z.device)
        xi = z[ar, i]
        xj = z[ar, j]
        add = xi + xj
        sub = xi - xj
        mul = self.state_scale * xi * xj
        zero = torch.zeros_like(add)
        stacked = torch.stack([add, sub, mul, zero], dim=-1)
        val = stacked[ar, op].clamp(-self.z_clip, self.z_clip)
        z2 = z.clone()
        z2[ar, k] = val
        if self.res is not None and self.use_residual:
            op_oh = F.one_hot(op.clamp(0, 3), 4).float()
            i_oh = F.one_hot(i.clamp(0, N_REG - 1), N_REG).float()
            j_oh = F.one_hot(j.clamp(0, N_REG - 1), N_REG).float()
            k_oh = F.one_hot(k.clamp(0, N_REG - 1), N_REG).float()
            inp = torch.cat([z, op_oh, i_oh, j_oh, k_oh], dim=-1)
            z2 = z2 + self.residual_scale * self.res(inp)
        return z2


class RegisterSMIR(nn.Module):
    def __init__(self, T: int = 4, residual: bool = True):
        super().__init__()
        self.z_dim = N_REG
        self.T = T
        self.state_scale = STATE_SCALE
        self.V = StructuredRegisterV(residual=residual, state_scale=self.state_scale)
        q_dim = 3 + N_REG + 1 + N_REG
        self.q_enc = nn.Sequential(nn.Linear(q_dim, 64), nn.SiLU(), nn.Linear(64, 64))
        self.answer = nn.Sequential(nn.Linear(N_REG + 64, 64), nn.SiLU(), nn.Linear(64, 2))

    def encode_state(self, state, device=None) -> torch.Tensor:
        if not torch.is_tensor(state):
            device = device or next(self.parameters()).device
            state = state_tensor(state, device)
        return state / self.state_scale

    def encode_query(self, query: dict, device=None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        return self.q_enc(query_tensor(query, device))

    def one_step(self, z, action: dict) -> torch.Tensor:
        op, i, j, k = action_tensors(action, z.device)
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
            {"state_dict": self.state_dict(), "T": self.T, "z_dim": self.z_dim},
            path / "register.pt",
        )

    @classmethod
    def load(cls, path: str | Path, device=None, residual: bool = True):
        path = Path(path)
        ckpt = torch.load(path / "register.pt", map_location=device or "cpu", weights_only=True)
        model = cls(T=int(ckpt["T"]), residual=residual)
        model.load_state_dict(ckpt["state_dict"])
        if device is not None:
            model.to(device)
        return model
