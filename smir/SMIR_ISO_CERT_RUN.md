# SMIR-Iso-Cert（修复 Venue-V 尺度爆炸）

- **近等距图表**：`(s/40) → Orthogonal Linear → LayerNorm`
- **相对共轭**：`‖V(z_t,a)-z_{t+1}‖² / (‖z_t-z_{t+1}‖²+ε)`
- Stage A 解锁 F → Stage B 冻 Q，训 V(+轻 E)

已取消错误的 `smir-vv`（冻坏图表硬训 V）。

```bash
cd /project/peilab/why/NEW
mkdir -p smir/artifacts/slurm
sbatch smir/slurm/29_smir_iso_cert.sbatch
```

冒烟：`STEPS_A=600 STEPS_B=800 EVAL_N=100 sbatch smir/slurm/29_smir_iso_cert.sbatch`

看：`tail -f smir/artifacts/slurm/smir-iso-<JOBID>.out`  
成功：`gates.pass_iso_cert`；正常时 `chart_norm` 应 ≪20，`flow` 相对损失应降到 O(1) 而非上千。
