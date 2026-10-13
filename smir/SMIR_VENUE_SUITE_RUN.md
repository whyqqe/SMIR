# SMIR Venue Suite（多实验流水线）

## 详细背景（为何要做）

| 已有 | 缺口 |
|------|------|
| RegV：η≈0、F≥0.5 | OOD=0.77 未过 0.80（训练链偏短） |
| RegV+：g\*、S 变好 | 仍卡 OOD；硬 V 被质疑「写死规则」 |
| 对照表雏形 | 缺可学习 V / task-only 消融 / 长链数据 |

本 suite 一次跑齐：

1. **E1 RegV-OOD**：`train_lens=2..5` + hard V → 冲 OOD  
2. **E2 SoftV**：可学习 soft 路由（非平凡结构）  
3. **E3 MLP-TF**：无结构 MLP + 绝对教师强制（可学习性）  
4. **E4 MLP-Task**：仅任务（消融，η 应差）  
5. **Summary + contrast**（含历史 7B/V*/RegV）

## 提交

```bash
cd /project/peilab/why/NEW
mkdir -p smir/artifacts/slurm
sbatch smir/slurm/33_smir_venue_suite.sbatch
```

冒烟：

```bash
STEPS_E1=800 STEPS_E2=800 STEPS_E3=800 STEPS_E4=600 EVAL_N=100 \
  sbatch smir/slurm/33_smir_venue_suite.sbatch
```

只跑部分：

```bash
SKIP=e4 sbatch smir/slurm/33_smir_venue_suite.sbatch
```

## 结果

```bash
tail -f smir/artifacts/slurm/smir-suite-<JOBID>.out
cat smir/artifacts/runs/smir_venue_suite/summary.json
cat smir/artifacts/runs/smir_venue_suite/contrast_table.json
```

关注：`claims.pass_suite_core`（E1）、`pass_learnability`（E2/E3）、`pass_ablation`（E3≪E4 的 η）。
