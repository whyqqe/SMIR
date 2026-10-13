# NeurIPS/ICML 向：反事实续算一致性完整实验

对齐 `new_theory.md`。目标不是再刷 SMIR 多损失分数，而是验证：

| 阶段 | 问题 | 实验 |
|------|------|------|
| **A 可行性** | 单目标 \(\mathcal L_{cont}\) 能否实现 | A1 Hard 冻结 \(V^*\)；A2 Discrete 可学 \(T\)；**A3 采样消融**（短测 vs 全覆盖） |
| **B 可识别** | 有限/弱测试何时不够 | B1 长度曲线；**B2 非分离查询训练**；B3 误差随长度 |
| **C 瓶颈** | 是否真经 \(z\) 计算 | C1 中状态替换；**C2 旁路泄漏**；C3 反事实 follow |
| **D 范围** | 结构与跨域 | D1 Hard/Disc vs MLP；D2 图域；D3 错误 chart |

成功：
- **core**：`pass_nips_core` = A∧B∧C（冲高会最低实证包）
- **strong**：再加 D（`pass_nips_strong`）

> 本流水线**不自动证明** §10 一般定理；B 提供经验可识别证据，定理需另写。

## 正式提交（默认单 seed=0 + 全阶段 A–D）

建议等 pub/str 结束后再提，或接受 preempt 竞争：

```bash
cd /project/peilab/why/NEW
mkdir -p smir/artifacts/slurm
sbatch smir/slurm/39_smir_nips.sbatch
```

多种子：

```bash
SEEDS=0,1,2 sbatch smir/slurm/39_smir_nips.sbatch
```

## 冒烟（约数十分钟级，视集群而定）

```bash
SEEDS=0 PHASES=A,B,C STEPS_1=400 STEPS_2=500 STEPS_V=400 EVAL_N=80 \
  SKIP_GRAPH=1 \
  sbatch smir/slurm/39_smir_nips.sbatch
```

只要 A+B+C+D、单 seed：

```bash
SEEDS=0 PHASES=A,B,C,D STEPS_1=800 STEPS_2=1000 STEPS_V=800 EVAL_N=120 \
  sbatch smir/slurm/39_smir_nips.sbatch
```

## 看结果

```bash
tail -f smir/artifacts/slurm/smir-nips-<JOBID>.out
cat smir/artifacts/runs/smir_nips/seed_0/summary.json
cat smir/artifacts/runs/smir_nips_pack/nips_pack.json
```

关注：
- `A/pass_A3_sampling_matters`（测试设计论断）
- `B/pass_B2_identifiability`（弱查询失败）
- `C/pass_C2_leak`（旁路证伪）
- `pass_nips_core` / `pass_nips_strong`

## 时间粗估

单 seed 全 A–D（含图域）约 **4–8h**；3 seed 约 **12–24h**（preempt 可能更长）。
