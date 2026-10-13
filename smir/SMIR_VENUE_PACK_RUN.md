# 下一步：RegV+ 冲总闸 + 四档对照表

1. **RegV+**：更宽 \(Q\)、混合训练、微残差 → 冲 `pass_regv_plus`（F+η+tub+OOD）  
2. **Contrast**：汇总 7B-SCF / V* / RegV / RegV+ 主表（不重训旧实验）

```bash
cd /project/peilab/why/NEW
mkdir -p smir/artifacts/slurm
sbatch smir/slurm/32_smir_venue_pack.sbatch
```

冒烟：

```bash
STEPS_A=1000 STEPS_B=400 EVAL_N=100 sbatch smir/slurm/32_smir_venue_pack.sbatch
```

结果：

```bash
cat smir/artifacts/runs/smir_regv_plus/overall.json
cat smir/artifacts/runs/smir_venue_pack/contrast_table.json
```
