# SMIR-RegV（成功率最高：寄存器空间结构化 V）

\(E(s)=s/40\in\mathbb R^5\)，\(V\) 在同一坐标上执行 add/sub/mul（mul 已按 scale 对齐）。

```bash
cd /project/peilab/why/NEW
mkdir -p smir/artifacts/slurm
sbatch smir/slurm/31_smir_regv.sbatch
```

纯硬结构（共轭最强）：

```bash
PURE=1 sbatch smir/slurm/31_smir_regv.sbatch
```

冒烟：`STEPS_A=800 STEPS_B=400 EVAL_N=100 sbatch smir/slurm/31_smir_regv.sbatch`

成功：`gates.pass_regv`（期望 `eta_abs`/`tub_abs` 接近 0，F 与 OOD 高）。
