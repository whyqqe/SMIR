# SMIR-AAAI 完整剩余实验（高成功率）

一次完成：

1. **P0 主表**：Hard V + 课程学习 + 高 distinguishing 比例 + `λ_flow∈{0.3,0.5,1.0}`  
2. **P1 对照表**：汇总历史 runs + 本轮 best λ  
3. **P2 LLM 负结果**：从 `smir_main` 整理 7B-SCF（F≈0）

成功标准（同一 ckpt）：`F≥0.5` 且 `OOD≥0.8` 且 `η<0.02` 且 `tub<0.05` 且 `S_gap≥0.15`。

## 提交

```bash
cd /project/peilab/why/NEW
mkdir -p smir/artifacts/slurm
sbatch smir/slurm/34_smir_aaai_complete.sbatch
```

冒烟：

```bash
STEPS_1=600 STEPS_2=800 EVAL_N=100 P_DIST=0.8 \
  sbatch smir/slurm/34_smir_aaai_complete.sbatch
```

若 F 仍偏低，可提高干预比例：

```bash
P_DIST=0.85 STEPS_1=4000 STEPS_2=5000 sbatch smir/slurm/34_smir_aaai_complete.sbatch
```

## 看结果

```bash
tail -f smir/artifacts/slurm/smir-aaai-<JOBID>.out
cat smir/artifacts/runs/smir_aaai_pack/aaai_pack.json
ls smir/artifacts/runs/smir_aaai_grid/
```

成功看：`aaai_ready_experimentally: true` 与 `table1_main`。
