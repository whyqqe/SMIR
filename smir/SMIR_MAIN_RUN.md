# SMIR 完善后主实验（统一 SCF）

对齐理论建议：**一个流、一个能量**。

\[
\mathcal L=\mathcal L_{\mathrm{task}}+\lambda\mathcal L_{\mathrm{flow}}
\]

| 方法 | 含义 |
|------|------|
| **SCF** | 主方法：`λ>0` 共轭流 |
| **ContCE** | 消融：同架构 `λ=0` |
| **Outcome** | 无潜在流，只从 `z0` 读出 |

主指标：`η`（共轭）、管状 MSE、干预 branch / on−off、长度曲线；任务 acc 为辅。  
`z_dim=128`（更强图表约束）。只存 adapter+heads。

## 提交

```bash
cd /project/peilab/why/NEW
mkdir -p smir/artifacts/slurm
sbatch smir/slurm/22_smir_main.sbatch
```

数据已存在时可直接上；若需重建：

```bash
J1=$(sbatch --parsable smir/slurm/20_build_shadow.sbatch)
sbatch --dependency=afterok:$J1 smir/slurm/22_smir_main.sbatch
```

冒烟：

```bash
STEPS=800 EVAL_N=100 LAM_FLOW=1.0 sbatch smir/slurm/22_smir_main.sbatch
```

## 看结果

```bash
squeue -u $USER
tail -f smir/artifacts/slurm/smir-main-<JOBID>.out
cat smir/artifacts/runs/smir_main/overall.json
```

成功看 `gates.pass_main`（几何 + 忠实 + OOD）与 `ablation_flow_necessary`。

## 说明

旧作业 `21_smir_venue`（多损失堆砌）可与本主实验并行或自行 `scancel`；正式解读以 **`smir_main`** 为准。
