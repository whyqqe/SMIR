# SMIR 发表向完整流水线（最高成功率）

对齐：`SMIR_PUB_EXPERIMENTS.md` + 定理 5/6/7 + 命题 8。

设计要点：
1. **E1** 优先复用已过线 AAAI `lam_0.5` ckpt（seed0）  
2. **Discrete/Soft gated**：Hard→Q 转移后过门控再学 \(V\)（主看 Flow-F）  
3. **Soft joint**：无门控负对照  
4. **diag / charts**：Thm5 必要性 + Thm6/7 图表电池  

## 提交前

若旧作业 `smir-next` 仍在跑，先取消以免占卡：

```bash
scancel -n smir-next
```

## 正式提交

```bash
cd /project/peilab/why/NEW
mkdir -p smir/artifacts/slurm
sbatch smir/slurm/36_smir_pub_core.sbatch
```

## 冒烟

```bash
SEEDS=0 EXPS=hard_e1,discrete_gated,soft_joint,diag,charts \
  STEPS_E1_1=400 STEPS_E1_2=400 STEPS_V=600 EVAL_N=100 \
  sbatch smir/slurm/36_smir_pub_core.sbatch
```

## 只冲最高成功率 M2（Discrete）

```bash
SEEDS=0,1,2 EXPS=hard_e1,discrete_gated,soft_joint,diag,charts \
  sbatch smir/slurm/36_smir_pub_core.sbatch
```

## 看结果

```bash
tail -f smir/artifacts/slurm/smir-pub-<JOBID>.out
cat smir/artifacts/runs/smir_pub_pack/pub_pack.json
```

成功：`claims.pass_pub_core: true`（M1∧M2）。更强：`pass_pub_strong`（另含门控优于 joint / diag / charts）。
