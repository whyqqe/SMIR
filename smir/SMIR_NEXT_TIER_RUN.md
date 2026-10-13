# SMIR Next-Tier 完整实验（冲击更高会）

修复点相对 AAAI Hard 主行：

1. **SoftV 退火**：PhaseA 高 bias≈Hard 训 Q → PhaseB bias→0 学 routing → PhaseC 保 OOD  
2. **Discrete V**：学离散路由 + 硬执行（STE）  
3. **Flow-F**：主判据用轨迹级干预，不只 oracle 状态 F  
4. **多种子**：默认 seeds `0,1,2`  
5. **g\* probe**：轻量验证图表距离与语义响应相关  
6. **MLP±teacher 消融**同表

成功（`pass_next_tier_core`）：Hard 过 M1，且 Soft **或** Discrete 过 M2（Flow-F≥0.45 ∧ OOD≥0.80 ∧ η/tub/S）。

## 提交

```bash
cd /project/peilab/why/NEW
mkdir -p smir/artifacts/slurm
sbatch smir/slurm/35_smir_next_tier.sbatch
```

冒烟（约数小时）：

```bash
SEEDS=0 EXPS=hard,soft STEPS_A=600 STEPS_B=800 STEPS_C=400 EVAL_N=100 \
  sbatch smir/slurm/35_smir_next_tier.sbatch
```

只冲 Soft M2（最快验证可学门）：

```bash
SEEDS=0,1,2 EXPS=soft \
  BIAS_START=5.0 BIAS_END=0.0 P_DIST=0.85 \
  sbatch smir/slurm/35_smir_next_tier.sbatch
```

Soft 仍差点时提高路由监督：

```bash
SEEDS=0,1 EXPS=soft,discrete \
  LAM_ROUTE=1.0 LAM_CONJ=1.0 BIAS_END=0.0 \
  sbatch smir/slurm/35_smir_next_tier.sbatch
```

## 看结果

```bash
tail -f smir/artifacts/slurm/smir-next-<JOBID>.out
cat smir/artifacts/runs/smir_next_pack/next_tier_pack.json
```

关注：`claims.pass_m2_soft` / `pass_m2_discrete` / `pass_next_tier_core`，以及 Soft 的 `F_flow.mean`。
