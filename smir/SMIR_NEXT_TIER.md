# SMIR 冲击更高会（NeurIPS / ICML / AAAI-27）实验方案

目标从「证书在控制环境成立」升级为「理论可检验 + 可学习 + 跨模型可复现」。  
当前 AAAI 主行（`lam=0.5`）已过内部门槛；更高会必须补齐下表缺口。

## 缺口（审稿必打）

| 缺口 | 现状 | 更高会要求 |
|------|------|------------|
| Hard V 构造性 | η≈0 近乎内置 | Soft/MLP 也要接近主行 |
| 统计稳健 | 单 seed | ≥3 seeds + 均值/方差 |
| LLM 锚 | 仅 7B 负结果 | 多尺度负/正对照 + 公开合成任务 |
| 可学习共轭 | SoftV F≈0.36 | SoftV/MLP 同时 F≥0.45 且 OOD≥0.80 |
| 理论对齐 | 文档有、论文未封 | 定理↔指标一一对应表 |

## 三阶段实验（按优先级）

### P0 — SoftV 冲主门槛（最关键，2–3 GPU·天）

**假设：** Hard 主行证明协议可测；若 SoftV 也能 F∧OOD，则「结构先验」不再是唯一解释。

**设计：**
1. 数据：继续 `SMIR-Shadow-Long`（train_lens 2–5）。
2. 模型：`SuiteSMIR(kind=soft)`；图表仍 `E(s)=s/40`。
3. 损失：`L = L_task + λ_F L_dist + λ_flow L_flow + λ_η L_conj`  
   - Phase1（短链）：`p_dist=0.85`，`λ_η=1.0`，`λ_flow=0.2`，`λ_F=1.0`  
   - Phase2（混合长链）：`p_dist=0.75`，`λ_η=0.5`，`λ_flow∈{0.3,0.5,1.0}`  
   - Phase3（若 F 掉、OOD 仍 ≥0.78）：提高 `p_dist` 到 0.9、降低 `λ_flow`
4. 网格：`λ_η × λ_flow` 小网格（约 6–9 点），**联合 early-stop**：最大化  
   `score = 2·min(F,0.6)+2·min(OOD,0.9)+min(S_gap,0.4)-0.5·η-0.2·tub`
5. Seeds：`{0,1,2}`。

**通过线（同一 ckpt，报告 mean±std）：**
- SoftV：`F≥0.45`，`OOD≥0.80`，`η<0.05`，`tub<0.2`，`S_gap≥0.15`
- 理想线：`F≥0.50`（与 Hard 同级）

**对照必须同表：** Hard（已有）、Soft best、MLP+teacher、MLP task-only、7B text-E。

### P1 — 多种子 + 长度外推稳健性（1 GPU·天）

对 Hard best 与 Soft best：
- seeds `{0,1,2}` 完整复跑；
- 长度外推：`T_eval ∈ {T, T+2, T+4, T_max+2}`；
- 报告 F / OOD / η / tub / S_gap 的 mean±std；
- 显著性：相对 MLP-task-only 的 paired bootstrap（至少 F 与 η）。

### P2 — LLM / 公开任务锚（2–4 GPU·天）

**定位：负结果+局部可测，不是冲 MATH SOTA。**

1. **扩展负结果：** 同一 Shadow 协议，在 `{1B, 3B, 7B}`（或实验室可用尺度）上测 text last-token E：预期 F≈0、tub 爆炸 → 支撑「错误图表不可认证」。
2. **弱正结果（加分项）：** 在固定寄存器图上，用小型 Transformer **只学 Q**（或浅层 adapter），Hard/Soft V 冻结或蒸馏：若 F≥0.4 且 OOD≥0.75，则说明协议可迁移到神经读出。
3. **公开合成对齐：** 增加 ProsQA / 简易图可达任务的一列「干预翻转」指标（不必超过 SOTA，只要协议可跑通）。

### P3 — 理论–实验封板（写作，并行）

主 claim 收窄为：
> 隐式推理可认证 ⟺ 图表下轨迹落在管状邻域且对干预电池忠实；错误图表（文本 last-token）在同标签下不可认证。

必备表：
- Table1：Hard / Soft / MLP±teacher（多种子）
- Table2：长度外推
- Table3：LLM 锚（负为主）
- Figure：F–OOD Pareto（λ 网格）

## 建议时间线

| 周 | 产出 |
|----|------|
| W1 | SoftV 网格 + 3 seeds；选出 Soft best |
| W2 | Hard/Soft 稳健性 + 长度外推；封 Table1/2 |
| W3 | 多尺度 LLM 负结果 +（可选）神经读出弱正 |
| W4 | 定理–指标对照、论文初稿、开源代码复现脚本 |

## 提交命令（实现后）

```bash
# SoftV 冲主门槛（示例；实现 run_smir_next_soft.py 后）
cd /project/peilab/why/NEW
sbatch smir/slurm/35_smir_next_soft.sbatch

# 多种子稳健性
SEEDS=0,1,2 sbatch smir/slurm/36_smir_next_multiseed.sbatch

# LLM 锚
sbatch smir/slurm/37_smir_next_llm_anchor.sbatch
```

## 成功/止损

- **冲 NeurIPS/ICML 继续：** SoftV 达通过线 + 3-seed 稳健 + LLM 负结果多尺度一致。  
- **改投 AAAI-27 稳妥线：** SoftV 达通过线但未到 Hard 同级，主打证书协议与错误图表证伪。  
- **止损重写理论：** SoftV 全网格 F<0.35 且 η 无法压下 → 主文改为「结构化可认证推理」而非「任意 V 可学共轭」。
