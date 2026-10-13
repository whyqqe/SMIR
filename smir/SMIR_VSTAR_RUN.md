# SMIR-V*（补齐最大缺口：共轭流）

**冻住** Iso Stage-A 的 `E`/`Q`，**只训 `V`**，用已知寄存器动力学做**绝对**教师强制：

\[
\sum_t \|V(E(s_t),a_t)-\mathrm{sg}(E(s_{t+1}))\|^2
\]

不再用相对共轭、不再解冻 `E`。

依赖：`smir/artifacts/runs/smir_iso_cert/stage_a`（已有即可）。

## 提交（请自行执行）

```bash
cd /project/peilab/why/NEW
mkdir -p smir/artifacts/slurm

# 可选：停掉仍在平台期的 iso Stage-B
# scancel 681180

sbatch smir/slurm/30_smir_vstar.sbatch
```

冒烟：

```bash
STEPS=1000 EVAL_N=100 sbatch smir/slurm/30_smir_vstar.sbatch
```

## 看结果

```bash
tail -f smir/artifacts/slurm/smir-vstar-<JOBID>.out
cat smir/artifacts/runs/smir_vstar/overall.json
```

正常：`[V*] flow/end` 应单调降到 ≪1（绝对 MSE），F 相对 init 掉幅 &lt;0.1。  
成功：`gates.pass_vstar`。
