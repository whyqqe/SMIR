# SMIR 提升发表价值：完整实验方案

对齐理论：`SMIR.md` 定理 5–7、命题 8、最小充分目标 (19)。  
目标：**用实验逐条证伪/支持理论主张**，不是冲 MATH/LLM SOTA。

---

## 0. 创新性确认（写作与实验必须守住）

### 0.1 相对前人，可辩护的增量

| 工作 | 给了什么 | SMIR 真正多出来的 |
|------|----------|-------------------|
| LIFIR / ICRM | S/F 判据 | **为何**需要 S/F 的几何充要 + 曲率组合界 |
| Coconut / 连续思维 | 连续格式 | 连续更新必须是**语义共轭流**，否则不可认证 |
| DAS / 因果抽象 | 变量干预 | 轨迹级共轭 + **图表合法性** |
| 忠实 CoT 文献 | 文本级干预 | **潜在图表 × 动力学 × 读出**分解 |

**主创新（论文应主打这三条，而不是「我们 F 更高」）：**

1. **证书分解（Thm 5）**：可认证 ⟺ 合法 \(E\) ∧ 共轭 \(V\) ∧ 忠实 \(Q\)；Oracle-F ≠ Flow-F。  
2. **错误图表必然失败（Thm 6）**：解释 last-token 负结果是理论预言，不是调参失败。  
3. **门控可学性（Prop 8）**：先 \(Q\) 后 \(V\) 有保证；无门控联合训练无保证。

### 0.2 创新性边界（避免被审稿打穿）

- Hard \(V\) 上 \(\eta\approx0\)：**不是**创新，只是读出腿上界。  
- 固定 \(E(s)=s/c\)：由 Thm 7 合法，但**不能**声称「学会了图表」。  
- 不声称无影子自然语言上的无条件证书。

### 0.3 发表价值公式

\[
\text{价值}
=
\underbrace{\text{Thm5 可分检验}}_{\text{主表}}
+
\underbrace{\text{Thm6 错误图表}}_{\text{负结果升格}}
+
\underbrace{\text{Prop8 门控 vs 无门控}}_{\text{可学性}}
+
\underbrace{\text{第二语义域}}_{\text{非玩具单一性}}
\]

缺任一项，更容易被看成「合成任务上训了个分类器」。

---

## 1. 总设计：四条实验轴

| 轴 | 理论 | 问题 | 主指标 |
|----|------|------|--------|
| **A 分解** | Thm 5 | 三腿是否可独立证伪/合成？ | \(\varepsilon_E,\eta_V,\delta_Q,\delta_Q^{flow}\) |
| **B 图表** | Thm 6/7 | 错误图表是否必败？尺度等价是否保持？ | Oracle-F, tub |
| **C 门控** | Prop 8 | 先 Q 后 V 是否显著优于联合退火？ | Flow-F, 过门控率 |
| **D 外推** | Thm 2 | 长度/域外是否符合组合界趋势？ | OOD, length curve |

全部实验 **3 seeds**，报 mean±std；主门控对可学习 \(V\) 用 **Flow-F**。

---

## 2. 数据

### D0：SMIR-Shadow-Long（已有，主域）
- train_lens `{2,3,4,5}`，OOD `{4,5}`  
- 保留 distinguishing 电池  

### D1：第二语义域（必做，抗「单一玩具」）
任选一个，2–3 GPU·天内可生成即可：

- **Stack-Shadow**：栈操作（push/pop/swap）+ 查询栈顶关系；或  
- **GraphReach-Shadow**：小图可达（ProsQA 风格）+ 边干预翻转标签  

要求：有影子状态、可定义 Hard \(V^*\)、可构造 distinguishing 对。

---

## 3. 实验清单（按论文图表编号）

### E1 — 读出腿上界（Thm 5(iii) / 已有可复用）
**设置：** 合法 \(E\)，Hard \(V^*\) 冻结，只训 \(Q\)（AAAI `lam=0.5` 配方：3000+5000+F-recovery）。  
**通过：** Oracle-F≥0.50，OOD≥0.80，η≈0，tub<0.05，S_gap≥0.15。  
**叙事：** 「证书在读出端可行」——**不要**写成学到共轭。  
**产出：** Table 1 第一行；可直接引用已有 `aaai_main_lam0.5`。

### E2 — 动力学腿：门控可学 vs 无门控（Prop 8，核心新实验）
**设置（同合法 \(E\)，同数据）：**

| 行 | 协议 |
|----|------|
| E2a Hard→Q→Discrete | 阶段A：Hard 上训 Q 至 Oracle-F≥0.55；载入 Q；阶段B：学离散路由+硬执行；主看 Flow-F |
| E2b Hard→Q→Soft | 同上，但学 soft 路由；**仅当** A 过门控才退火 bias→0 |
| E2c Joint-Soft（负对照） | 无门控，V+Q 联合 + 早退火（复现失败模式） |
| E2d MLP-TF / MLP-Task | 非结构化消融 |

**通过（E2a 或 E2b）：** Flow-F≥0.45，OOD≥0.80，η<0.05，tub<0.20。  
**关键对比：** E2a/b 的 Flow-F **显著高于** E2c（bootstrap）。  
**产出：** Table 2；Figure：门控前后 Flow-F 曲线。

### E3 — 分解可分检验（Thm 5 必要性）
固定另两腿、破坏一腿：

| 破坏 | 操作 | 预期 |
|------|------|------|
| 坏 \(E\) | 随机投影 / 打乱寄存器维 / 常数图表 | Oracle-F 崩（→E4） |
| 坏 \(V\) | 打乱 op 或零场 | η↑，Flow-F≪Oracle-F |
| 坏 \(Q\) | 随机读出或未训 Q | Oracle-F≈0，即使 Hard V |

**产出：** Figure「三腿诊断条形图」——审稿最爱的可证伪图。

### E4 — 错误图表电池（Thm 6，负结果升格）
**同标签影子协议**下比较图表：

| 图表 | 预期 |
|------|------|
| \(E(s)=s/c\)（合法） | F 高、tub 低 |
| \(E(s)=s/c'\) 尺度变体 | F 近似保持（Thm 7） |
| 随机高斯投影 \(P E(s)\) | F 降、符合 (18) |
| 维置换/折叠 | F 降 |
| LLM last-token（1B/3B/7B 能跑几个算几个） | F≈0，tub 爆炸 |

**产出：** Table 3「图表合法性」；把 7B 从「趣闻」变成定理 6 的实例。

### E5 — 组合界 / 长度外推（Thm 2）
在 E1 与 E2-best 上画：
- length ∈ `{T, T+2, T+4, T_max+2}` 的 task / Flow-F / tub  
- 报告是否随长度近似呈现式 (8) 的变差趋势（定性即可）

**产出：** Figure length curves。

### E6 — 第二域重复 E1+E2a+E4 子集（泛化叙事）
不必跑满；至少：Hard 上界 + 门控 Discrete + 一种错误图表。  
**产出：** Table 4「跨域」——显著提升非玩具说服力。

### E7 — 最小充分损失（式 19，轻量）
在门控 Soft/Discrete 上消融：
- full (19)：task + F + conj  
- w/o F、w/o conj、task-only  

**预期：** task-only η/F 差；w/o F 则 Flow-F 差；与推论 9 一致。

---

## 4. 论文表格草图（投稿时）

**Table 1（主结果）**  
Domain × {Hard-Q, Discrete-gated, Soft-gated, Joint-Soft, MLP-TF, MLP-Task}  
列：Oracle-F, Flow-F, η, tub, OOD, S_gap（mean±std）

**Table 2（图表）**  
合法 / 尺度 / 投影 / LLM-E × {F, tub, η}

**Table 3（门控）**  
Gated vs Joint：Flow-F 与过线率

**Figure 1** 证书分解示意图（理论）  
**Figure 2** 三腿破坏诊断  
**Figure 3** 门控训练曲线  
**Figure 4** 长度外推  

---

## 5. 成功标准与止损（发表策略）

### 冲 NeurIPS/ICML 的最低实验包
- [x] E1 过线（已有 AAAI Hard）  
- [ ] E2a **或** E2b 过 Flow-F 门控，且显著打赢 E2c  
- [ ] E3 三腿诊断图完整  
- [ ] E4 含 ≥2 种错误图表 + ≥1 个 LLM 尺度  
- [ ] E5 长度曲线  
- [ ] E6 第二域至少 Hard+Discrete+坏图表  

### 若 E2 Soft 不过、Discrete 过
→ 主文写「可学离散程序共轭 + 硬执行」，Soft 作连续松弛讨论；**仍可投**（更诚实，常更好中）。

### 若 E2 全不过
→ 收缩为 AAAI：**证书协议 + Thm6 错误图表 + 分解诊断**；不硬吹可学习 \(V\)。

### 明确不要做的
- 再扫 ambient MLP λ  
- 试图把 text-E F 拉高  
- 无第二域却声称「通用隐式推理」  
- 用 Hard η≈0 当主贡献  

---

## 6. 执行顺序与算力（约 1–2 周）

| 周 | 任务 | 预算 |
|----|------|------|
| D1–2 | 冻结 E1（复用 AAAI）；实现 Hard→Q→Discrete/Soft 门控训练；修 Flow-F 报告 | 1 GPU |
| D3–5 | E2 全表 × 3 seeds；E7 消融 | 1–2 GPU |
| D6–7 | E3 诊断 + E4 错误图表（含 LLM 若资源允许） | 1 GPU（LLM 另计） |
| D8–9 | D1 第二域生成 + E6 子集 | 1 GPU |
| D10 | E5 曲线、打包、与定理对照表写入论文 | CPU |

**已实现一键流水线：** 见 `SMIR_PUB_RUN.md`。

```bash
cd /project/peilab/why/NEW
scancel -n smir-next   # 若旧作业仍在
sbatch smir/slurm/36_smir_pub_core.sbatch
```

第二域（E6）仍为后续加分项，未打进 36 号作业。

---

## 7. 每条实验必须在论文中对应的一句话

| 实验 | 论文句 |
|------|--------|
| E1 | 合法图表下读出腿可认证（Thm 5 上界） |
| E2 gated | 门控使动力学腿可学（Prop 8） |
| E2 joint | 无门控无保证（Prop 8 反面） |
| E3 | 任一腿失败 ⇒ 整体不可认证（Thm 5 必要） |
| E4 | 错误图表必然不可认证（Thm 6） |
| E5 | 误差随深度按共轭界恶化（Thm 2） |
| E6 | 分解不依赖单一算术域 |
| E7 | 最小充分损失非任意堆 λ（式 19） |

---

## 8. 一句话收束

发表价值来自：**用可分实验证明「可认证 = E∧V∧Q」，用错误图表与门控对照证明理论不是事后解释。**  
现有 Hard 主行只是 E1；把 E2–E4（+E6）做实，才配得上已完善的定理 5–7 / 命题 8。
