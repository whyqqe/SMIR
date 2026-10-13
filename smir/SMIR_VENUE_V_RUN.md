# SMIR-Venue-V（冲击顶会下一步）

**策略：** 冻结已打通 F 的 Stage-A（`E`,`Q`），**只训 `V`**。  
全链教师强制共轭 + 终点管状 + 轻量 task；并报告 oracle 上界。

不重复：7B / ContCE / 联训 Stage-B。

## 提交

```bash
cd /project/peilab/why/NEW
mkdir -p smir/artifacts/slurm
sbatch smir/slurm/28_smir_venue_v.sbatch
```

冒烟：

```bash
STEPS=800 EVAL_N=100 sbatch smir/slurm/28_smir_venue_v.sbatch
```

## 看结果

```bash
squeue -u $USER
tail -f smir/artifacts/slurm/smir-vv-<JOBID>.out
cat smir/artifacts/runs/smir_venue_v/overall.json
```

成功：`gates.pass_venue_v`（几何 + 保 F + S/OOD）。
