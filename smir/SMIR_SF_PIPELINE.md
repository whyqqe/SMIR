# SMIR-SF 完整流水线（Exp-0 → Exp-1 → Exp-2）

**不重复**：不再跑 ContCE / Outcome / 纯 `L_task+λL_flow` SCF 三路对比。

| 阶段 | 内容 | 产物 |
|------|------|------|
| **Exp-0** | 冻结旧 SCF：E 可分性 / 线性探针 vs Q / 尺度管状 | `diag_scf.json` |
| **Exp-1** | Phase-S 状态可分预热 → Phase-F 共轭+终点锚定 `L_end` | `phase_s/` `full/` |
| **Exp-2** | 相对 tubular、离流形对抗、F/on−off、η、长度曲线 | `overall.json` gates |
| **Exp-0b** | 对新 `full` ckpt 再诊断 | `diag_full.json` |
| Exp-3（可选） | `METRIC_ALIGN=1` 打开 Phase-S 微扰项 | 同 `overall.json` |

训练目标（Phase-F）：

\[
\mathcal L=\mathcal L_{\mathrm{task}}+\lambda_{\mathrm{flow}}\mathcal L_{\mathrm{flow}}+\lambda_{\mathrm{end}}\|z_T-\mathrm{sg}(E(s^*))\|^2
\]

---

## 一键全流水线（推荐）

```bash
cd /project/peilab/why/NEW
mkdir -p smir/artifacts/slurm
sbatch smir/slurm/25_smir_sf_pipeline.sbatch
```

冒烟：

```bash
STEPS_S=400 STEPS_F=600 EVAL_N=80 N_DIAG=80 sbatch smir/slurm/25_smir_sf_pipeline.sbatch
```

可选打开 Exp-3 stub：

```bash
METRIC_ALIGN=1 sbatch smir/slurm/25_smir_sf_pipeline.sbatch
```

无旧 SCF、跳过诊断：

```bash
SKIP_DIAG=1 sbatch smir/slurm/25_smir_sf_pipeline.sbatch
```

---

## 分步提交

```bash
# Exp-0 only（依赖 smir_main/scf）
sbatch smir/slurm/23_smir_diag.sbatch

# Exp-1+2 only
sbatch smir/slurm/24_smir_sf.sbatch

# 仅 Phase-S 对照（消融：无流）
MODE=phase_s_only sbatch smir/slurm/24_smir_sf.sbatch
```

依赖串联：

```bash
J0=$(sbatch --parsable smir/slurm/23_smir_diag.sbatch)
sbatch --dependency=afterok:$J0 smir/slurm/24_smir_sf.sbatch
```

---

## 看结果

```bash
squeue -u $USER
tail -f smir/artifacts/slurm/smir-pipe-<JOBID>.out

cat smir/artifacts/runs/smir_sf/diag_scf.json
cat smir/artifacts/runs/smir_sf/overall.json
cat smir/artifacts/runs/smir_sf/diag_full.json
```

成功关注：`gates.pass_phase_s`、`gates.pass_full`；诊断关注 `verdict.recommend`。
