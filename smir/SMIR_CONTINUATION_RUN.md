# 反事实续算一致性：初步可行性实验

对齐：`new_theory.md`（单一 \(\mathcal L_{\mathrm{continuation}}\) + §11 证伪电池）。

## 验证什么

| 理论点 | 实验 |
|--------|------|
| 统一续算目标 | **只用** CE 续算损失（无 conj / route 辅助项） |
| 空 / 多步 / 反事实 | batch 混合 `empty` / `full` / `prefix` / `cfact` |
| 完整中间状态替换 | `replace_midstate_acc` |
| 长程 | OOD length / `task_ood` |
| 干预忠实 | Flow-F |
| 结构是否必要 | `hard` vs `mlp`（`structured_beats_mlp`） |
| 跨域 | 图域 `ghard` / `gdisc` |

成功：`claims.pass_continuation_feasibility`（寄存器 Hard 续算过线，且 Discrete 或图域 Hard 过线）。

## 正式提交

可与仍在跑的 pub/str **并行**（另一张卡）：

```bash
cd /project/peilab/why/NEW
mkdir -p smir/artifacts/slurm
sbatch smir/slurm/38_smir_continuation.sbatch
```

## 冒烟

```bash
SEEDS=0 KINDS=hard,discrete STEPS_1=400 STEPS_2=500 STEPS_V=400 EVAL_N=80 \
  SKIP_GRAPH=1 \
  sbatch smir/slurm/38_smir_continuation.sbatch
```

只冲寄存器 + 图 Hard：

```bash
SEEDS=0,1 KINDS=hard,discrete,mlp GRAPH_KINDS=ghard \
  sbatch smir/slurm/38_smir_continuation.sbatch
```

## 看结果

```bash
tail -f smir/artifacts/slurm/smir-cont-<JOBID>.out
cat smir/artifacts/runs/smir_continuation_pack/continuation_pack.json
```

## 期望解读

- **可行性成立**：Hard（冻结 \(V^*\)）+ 单目标即可过 F/OOD/替换 → 理论目标可实现。
- **可学习性**：Discrete / gdisc 在无 aux 下也能过 → 支持「统一目标可学」。
- **MLP 明显更差**：说明仍需动力学归纳偏置（写作上如实写，不装作无结构）。
- **本流水线不证明**有限测⇒长程一般定理；那是 `new_theory` §10 的开放问题。
