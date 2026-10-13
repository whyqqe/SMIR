# SMIR 正式实验（语义流形隐式推理）

对齐 `SMIR.md`：共轭流、管状吸引、度量对齐、法向无关、干预忠实；对照 ContCE / Outcome。  
模型：Qwen2.5-7B-Instruct + LoRA；**只存 adapter+heads**（避免再占 ~45G）。

## 提交

```bash
cd /project/peilab/why/NEW
mkdir -p smir/artifacts/slurm

J1=$(sbatch --parsable smir/slurm/20_build_shadow.sbatch)
sbatch --dependency=afterok:$J1 smir/slurm/21_smir_venue.sbatch
```

一键：

```bash
sbatch smir/slurm/21_smir_venue.sbatch
```

冒烟：

```bash
STEPS=600 EVAL_N=80 sbatch smir/slurm/21_smir_venue.sbatch
```

## 看结果

```bash
squeue -u $USER
tail -f smir/artifacts/slurm/smir-venue-<JOBID>.out
cat smir/artifacts/runs/smir_venue/overall.json
```

关注 `gates.pass_geometry` / `pass_faithfulness` / `pass`，以及 `conjugacy_mse`、`tubular_mse`、`on_minus_off`。
