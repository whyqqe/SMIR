#!/usr/bin/env python3
"""SMIR Tier-B model: 7B LoRA encoder + latent flow on semantic chart coordinates."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

OP2ID = {"add": 0, "sub": 1, "mul": 2, "null": 3}


class SemanticFlow(nn.Module):
    """V_θ(z, u): residual MLP in chart coordinates (latent manifold chart)."""

    def __init__(self, z_dim: int, n_ops: int = 4):
        super().__init__()
        self.op_emb = nn.Embedding(n_ops, 64)
        self.net = nn.Sequential(
            nn.Linear(z_dim + 64, z_dim * 2),
            nn.SiLU(),
            nn.Linear(z_dim * 2, z_dim),
        )

    def forward(self, z: torch.Tensor, op_id: torch.Tensor) -> torch.Tensor:
        u = self.op_emb(op_id.clamp(0, self.op_emb.num_embeddings - 1))
        return z + self.net(torch.cat([z, u], dim=-1))


class SMIRModel(nn.Module):
    def __init__(self, model_name: str, z_dim: int = 256, T: int = 4):
        super().__init__()
        self.T = T
        self.z_dim = z_dim
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        base.config.use_cache = False
        self.backbone = get_peft_model(
            base,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
        )
        self.backbone.print_trainable_parameters()
        h = base.config.hidden_size
        self.to_z = nn.Linear(h, z_dim)
        self.V = SemanticFlow(z_dim)
        self.answer = nn.Sequential(nn.Linear(z_dim, z_dim), nn.SiLU(), nn.Linear(z_dim, 2))
        dev = next(self.backbone.parameters()).device
        self.to_z.to(dev)
        self.V.to(dev)
        self.answer.to(dev)

    @property
    def device(self):
        return next(self.to_z.parameters()).device

    def encode_text(self, text: str) -> torch.Tensor:
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=384)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.backbone(**enc, output_hidden_states=True, return_dict=True)
        hidden = out.hidden_states[-1]
        idx = enc["attention_mask"].sum(dim=1) - 1
        h = hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
        return self.to_z(h.float())

    def flow(self, z0: torch.Tensor, op_ids: torch.Tensor) -> torch.Tensor:
        z = z0
        for t in range(op_ids.size(1)):
            z = self.V(z, op_ids[:, t])
        return z

    def one_step(self, z: torch.Tensor, op_id: torch.Tensor) -> torch.Tensor:
        return self.V(z, op_id)

    def logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.answer(z)

    def save_light(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(path / "adapter")
        torch.save(
            {
                "to_z": self.to_z.state_dict(),
                "V": self.V.state_dict(),
                "answer": self.answer.state_dict(),
                "T": self.T,
                "z_dim": self.z_dim,
            },
            path / "heads.pt",
        )

    @classmethod
    def load_light(cls, model_name: str, path: str | Path, z_dim: int | None = None, T: int | None = None):
        """Load LoRA adapter + heads saved by save_light."""
        from peft import PeftModel

        path = Path(path)
        ckpt = torch.load(path / "heads.pt", map_location="cpu", weights_only=True)
        z_dim = int(z_dim if z_dim is not None else ckpt["z_dim"])
        T = int(T if T is not None else ckpt["T"])
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.T = T
        model.z_dim = z_dim
        model.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if model.tok.pad_token is None:
            model.tok.pad_token = model.tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        base.config.use_cache = False
        model.backbone = PeftModel.from_pretrained(base, str(path / "adapter"))
        h = base.config.hidden_size
        model.to_z = nn.Linear(h, z_dim)
        model.V = SemanticFlow(z_dim)
        model.answer = nn.Sequential(nn.Linear(z_dim, z_dim), nn.SiLU(), nn.Linear(z_dim, 2))
        model.to_z.load_state_dict(ckpt["to_z"])
        model.V.load_state_dict(ckpt["V"])
        model.answer.load_state_dict(ckpt["answer"])
        dev = next(model.backbone.parameters()).device
        model.to_z.to(dev)
        model.V.to(dev)
        model.answer.to(dev)
        model.eval()
        return model
