# SMIR 缺陷补强实验（完整流水线）

针对审稿短板：

| 缺陷 | 本流水线 |
|------|----------|
| 单域 | **图距离第二语义域** `SMIR-Graph-Shadow` |
| 充分性 | 图域 **seeds 0,1,2**；寄存器域复用 `smir_pub_core` |
| LLM 仅负例 | **图表适配器弱正例**（文本特征→合法图表，对比随机文本图表） |
| 错误图表 | 图域 charts / diag |

> 理论增量与野外部署边界靠论文叙事收窄，不靠本作业“消掉”。

## 提交

可与仍在跑的 `smir-pub` **并行**（不同 job）：

```bash
cd /project/peilab/why/NEW
mkdir -p smir/artifacts/slurm
sbatch smir/slurm/37_smir_strengthen.sbatch
```

冒烟：

```bash
SEEDS=0 STEPS_1=600 STEPS_2=800 STEPS_V=600 ADAPTER_STEPS=400 EVAL_N=100 \
  sbatch smir/slurm/37_smir_strengthen.sbatch
```

若希望等寄存器多种子完成后再打包，可先跑完 `36`，再提交 `37`（打包会读 `smir_pub_core`）。

## 看结果

```bash
tail -f smir/artifacts/slurm/smir-str-<JOBID>.out
cat smir/artifacts/runs/smir_strengthen_pack/strengthen_pack.json
```

成功：`claims.pass_strengthen_core: true`  
（寄存器 M1∧M2 且图域 M1∧M2；弱正例见 `pass_weak_positive_adapter`。）
