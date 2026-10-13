# 干预完备推理机制理论（ICRM）

**英文名：** Intervention-Complete Reasoning Mechanisms  
**定位：** 面向“如何让模型真正学会推理”的新理论框架  
**相对 CCRO / CCRO-FTB：** 保留续推残差与混合界，但以**捷径可证伪性**与**干预完备分离族**取代过强的线性响应重构假设，并把经验中反复出现的失败模式写入公理与评价。

---

## 0. 先回答：原理论是否足够？

### 0.1 结论

**不够支撑“真正学会推理”这一总问题的顶会级完整答案；作为局部执行机制的条件证书，CCRO-FTB 方向正确但假设过强、问题过窄。**

### 0.2 足够的部分

| 要素 | 价值 |
|------|------|
| 用后续行为分歧定义局部错误（δ） | 正确：最终答案不足以暴露机制 |
| 混合执行界 | 正确：局部残差与长链误差可联系 |
| 冻结续推、输入梯度归因 | 正确：抑制“后续参数补锅” |
| 可识别性 κ | 正确：弱测试上的低残差无意义 |

### 0.3 不够的部分（来自理论缺口 + 我们的实验）

1. **FTB-LRR（有界线性响应重构）过强且难验证**  
   实测 \(\zeta\) 偏大时，主证书变空；理论把最难点假定为已成立。

2. **把“真正推理”缩成“给定操作序列的执行”过窄**  
   规划、抽象、迁移未被主定理覆盖，却常被口头承诺。

3. **缺少反捷径公理**  
   7B Outcome 也能抬高 branch；同答案捷径策略可模仿状态敏感表象。没有“相对捷径对手的优势”就无法定义“真正”。

4. **经典硬探针可失败而平均指标仍好看**  
   Anchor/转移近饱和、classic 仍不分 —— 说明平均 δ / 准确率不是机制充分统计量。

5. **半真实域探针若无 oracle 分离，不可作为理论推论**  
   MATH 启发式 `label_wrong` 失败，暴露“测量定义错了就没有外部效度”。

6. **与过程监督的边界仍易被收编**  
   若实现退化成状态 SFT，理论创新不落地。

因此：需要一套**以捷径可证伪与干预完备为核心**的新理论，CCRO 的 δ 与混合界降为其中模块，而非整篇论文的唯一支柱。

---

## 1. 问题重述：什么叫“真正学会推理”

### 1.1 操作性定义（ICRM）

> 模型在任务族 \(\mathcal D\) 上**真正学会推理**，当且仅当存在可观测（或受控）状态通道，使得模型实现了该族语义更新机制的一个**行为同构像**，并且对所有**答案匹配的捷径策略类** \(\mathcal H_{\mathrm{short}}\)，存在有限**干预电池** \(\mathcal B\)，使模型通过而 \(\mathcal H_{\mathrm{short}}\) 中容量不超过预算的策略无法同时通过。

要点：

- 不要求唯一算法，只要求与目标相关的**干预行为**正确；  
- “真正”被定义为**相对捷径类的可证伪优势**，不是更高的答对率；  
- 证书始终相对于明确的 \(\mathcal D,\mathcal B,\mathcal H_{\mathrm{short}}\)，禁止无范围口号。

### 1.2 三个层级（必须分开认证）

| 层级 | 名称 | 问题 |
|------|------|------|
| L1 | 局部执行机制 | 给定操作，状态是否按语义更新？ |
| L2 | 组合执行 | 未见操作序列上是否保持？ |
| L3 | 目标导向规划 | 是否选择完成目标的操作？ |

ICRM 主理论先闭合 **L1+L2**；L3 为扩展（需策略等价类，不能假装已被 L1 证明）。

---

## 2. 基本对象

### 2.1 语义系统

- 状态 \(s\in\mathcal S\)，操作 \(a\in\mathcal A\)，转移 \(s'=f(s,a)\)  
- 干预/续推 \(c\in\mathcal C\)，oracle 输出 \(y^*(s,c)\)  
- 合法任务族 \(\mathcal D=(\mathcal S,\mathcal A,f,\mathcal C,y^*)\)

### 2.2 神经系统与受控通道

- 编码 \(E_\theta(s)\)，更新 \(F_\theta(z,a)\)，续推核 \(P_\theta(Y\mid z,c)\)  
- **通道公理：** 评测分支除被干预状态外，共享信息必须一致；禁止原题旁路。

### 2.3 机制等价（因果商）

定义

\[
s\sim_{\mathcal C} s'
\iff
\forall c\in\mathcal C:\ y^*(s,c)=y^*(s',c).
\]

商空间 \(\mathcal S/\sim_{\mathcal C}\) 是“对未来干预而言可区分的机制状态”。  
**真正推理关心的是商上的更新，而不是任意潜在坐标。**

---

## 3. 核心概念 I：干预电池与分离完备性

### 3.1 干预电池

有限集合 \(\mathcal B=\{c_1,\ldots,c_K\}\subset\mathcal C\)。

对状态对 \((s,s')\)：

\[
\mathrm{Sep}_{\mathcal B}(s,s')
=\mathbf{1}\{\exists c\in\mathcal B:\ y^*(s,c)\ne y^*(s',c)\}.
\]

### 3.2 定义：\(\mathcal P\)-分离完备

对需区分的机制对集合 \(\mathcal P\subset\mathcal S\times\mathcal S\)（混淆对、近邻错误、部署敏感对）：

\[
\mathcal B\text{ 对 }\mathcal P\text{ 分离完备}
\iff
\forall (s,s')\in\mathcal P:\ \mathrm{Sep}_{\mathcal B}(s,s')=1.
\]

记完备时 \(\kappa_{\mathcal P}(\mathcal B)=1\)（或对随机电池给概率版 \(\kappa\)）。

### 3.3 命题 1（分离完备 ⇒ 无假低残差陷阱）

若 \(\mathcal B\) 对 \(\mathcal P\) 分离完备，则不存在 \((s,s')\in\mathcal P\) 使得所有 \(c\in\mathcal B\) 上 \(y^*\) 相同。  
因此“在 \(\mathcal B\) 上行为一致”不能再隐藏 \(\mathcal P\) 内的机制错误。

**与 CCRO-FTB 的关键差别：**  
我们**不**假设完整 \(\mathcal C\) 上的线性响应重构 (FTB-LRR)。  
我们只要求：对**预先声明的混淆类 \(\mathcal P\)**，电池可分离。  
完整 \(\mathcal C\) 的控制改为下面的**扩张引理**（构造性，而非解析线性假设）。

### 3.4 命题 2（电池扩张引理，构造性）

设 \(\mathcal B\) 对 \(\mathcal P\) 完备。若发现 \(c^\dagger\in\mathcal C\) 与某对 \((s,s')\in\mathcal P\) 上行为无法由 \(\mathcal B\)-签名预测（经验检验），则令 \(\mathcal B\leftarrow\mathcal B\cup\{c^\dagger\}\) 并更新 \(\mathcal P\)。  
在**有限** \(\mathcal P\) 与有限候选库上，该过程在有限步终止于对当前 \(\mathcal P\) 的完备电池。

这把“有限测试基”从强解析假设改成**对有限认证域的构造过程**——与我们实验中“搜 distinguishing continuation”一致，且可算法化。

---

## 4. 核心概念 II：捷径类与可证伪优势

### 4.1 答案匹配捷径类 \(\mathcal H_{\mathrm{short}}\)

一类假设策略/模型，满足：在分布 \(d_{\mathrm{ans}}\) 上最终答案风险不高，但**不实现**正确商更新。典型成员：

- Outcome-only 拟合；  
- 浅层模式匹配；  
- 可读状态但不具备干预因果角色的通道（“状态装饰”）。

### 4.2 干预风险

对模型 \(\theta\) 与电池 \(\mathcal B\)：

\[
R_{\mathcal B}(\theta)
=\mathbb E_{(s,a)\sim d}\max_{c\in\mathcal B}
\mathrm{TV}\big(
P_\theta(Y\mid F_\theta(E_\theta(s),a),c),\,
P_\theta(Y\mid E_\theta(f(s,a)),c)
\big).
\]

（局部残差的电池最坏版；可用熵正则软化。）

### 4.3 定义：相对捷径的 \(\varepsilon\)-机制优势

\[
\mathrm{Adv}(\theta;\mathcal H_{\mathrm{short}},\mathcal B)
=
\inf_{h\in\mathcal H_{\mathrm{short}}} R_{\mathcal B}(h)
- R_{\mathcal B}(\theta).
\]

若 \(\mathrm{Adv}\ge\varepsilon>0\)，称 \(\theta\) 在 \((\mathcal B,\mathcal H_{\mathrm{short}})\) 上具有**可证伪机制优势**。

### 4.4 公理（真正推理的最低要求）

**公理 A（反捷径）。**  
仅当存在声明的 \((\mathcal D,\mathcal P,\mathcal B,\mathcal H_{\mathrm{short}},\varepsilon)\) 使 \(\mathcal B\) 分离完备且 \(\mathrm{Adv}\ge\varepsilon\)，才允许声称“学到了推理机制（L1）”。

这把我们的实验教训写成理论门槛：**超过 Outcome / Process-SFT 不够；必须超过显式捷径类。**

---

## 5. 核心概念 III：局部残差、归因与组合（继承并收紧 CCRO）

### 5.1 局部因果续推残差

\[
\delta_\theta(s,a,c)
=\mathrm{TV}\big(
P_\theta(Y\mid F_\theta(E_\theta(s),a),c),\,
P_\theta(Y\mid E_\theta(f(s,a)),c)
\big).
\]

### 5.2 命题 3（混合执行界，同 CCRO）

对给定合法序列 \(a_{1:T}\)，

\[
\mathrm{TV}(H_0,P^*)
\le
\epsilon_{\mathrm{read}}+\sum_{t=1}^{T}\delta_\theta(s_{t-1},a_t,c_t).
\]

### 5.3 命题 4（电池最坏版组合界）

若评测续推属于 \(\mathrm{span}_{\mathrm{beh}}(\mathcal B)\)——即其行为由 \(\mathcal B\)-签名在认证域上决定（由扩张过程保证的有限域性质，而非全局线性假设）——则

\[
\sum_t\delta_\theta(s_{t-1},a_t,c_t)
\le
\sum_t\max_{c\in\mathcal B}\delta_\theta(s_{t-1},a_t,c).
\]

从而长链误差由**电池最坏残差**控制。  
**失败模式写入理论：** 若续推落在电池行为闭包之外，证书自动无效，必须扩张 \(\mathcal B\)。

### 5.4 冻结续推归因（方法层，保留）

局部损失用 \(D_{\mathrm{KL}}(p_{\mathrm{ref}}\Vert p_{\mathrm{pred}})\)；续推参数冻结、保留对 \(z_{\mathrm{pred}}\) 输入梯度，使信用归于 \(F_\theta\)。

### 5.5 命题 5（归因必要性，定性）

若允许续推参数与 \(F_\theta\) 同训，则存在投机解：错误 \(F\) + 补偿续推，使端到端答案对而 \(\delta\) 在评测冻结协议下暴露。  
故**评测与训练协议必须固定续推核**，否则机制主张不成立。  
（对应我们坚持 frozen continuation 的理由。）

---

## 6. 学习目标（ICRM 优化）

### 6.1 主问题

\[
\begin{aligned}
\min_\theta\quad
&R_{\mathcal B}(\theta)
+\lambda_{\mathrm{anc}}\mathcal L_{\mathrm{anchor}}(\theta)\\
\mathrm{s.t.}\quad
&\mathcal B\text{ 对 }\mathcal P\text{ 分离完备},\\
&\mathrm{Adv}(\theta;\mathcal H_{\mathrm{short}},\mathcal B)\ge\varepsilon,\\
&\text{（可选）长度 }T\le T_{\max}\text{ 上混合界经验 hold}.
\end{aligned}
\]

### 6.2 交替过程

1. **锚定**参考编码分支（oracle）；  
2. **残差优化**（冻结续推）；  
3. **挖混淆对**更新 \(\mathcal P\)；  
4. **oracle 搜测**扩张 \(\mathcal B\) 至分离完备；  
5. **对抗评测**估计 \(\mathrm{Adv}\) vs Outcome/Process/其它捷径；  
6. **长度曲线**检查组合；失败则加紧预算或扩电池。

这与我们已跑通的流水线同构，但评价中心从 `gates.pass` 准确率改为 **Adv + 分离完备 + 长度 hold**。

---

## 7. 从可执行域到自然语言：同态迁移原理

### 7.1 抽象状态代数

设可执行域 \(\mathcal D_{\mathrm{ex}}\) 与语言域 \(\mathcal D_{\mathrm{nl}}\)。  
若存在映射 \(\phi:\mathcal S_{\mathrm{nl}}\to\mathcal S_{\mathrm{ex}}\)（或相反）使在相关操作上近似交换：

\[
\phi(f_{\mathrm{nl}}(s,a))\approx f_{\mathrm{ex}}(\phi(s),\psi(a)),
\]

则称存在**机制同态**。

### 7.2 命题 6（迁移的必要条件，不是充分）

若同态不成立，则 \(\mathcal D_{\mathrm{ex}}\) 上的机制优势**不蕴含** \(\mathcal D_{\mathrm{nl}}\) 上的机制优势。  
因此 MATH/GSM 必须自建**oracle 可分离**的干预电池；不可用启发式标签冒充。

### 7.3 经验规则（上升为方法论公理）

**公理 B（测量诚实）。**  
外部效度实验的每个 YES/NO 或分支标签，必须由域内语义 oracle 导出；否则结果不得进入机制定理的经验对应物。

---

## 8. 可检验预言（理论 → 实验）

1. **平均准确率对齐时，Adv 仍可分离真正机制与捷径。**  
2. **电池未完备时，可制造假低 \(R_{\mathcal B}\)。**（κ 诊断）  
3. **放开续推同训会降低冻结协议下的 Adv。**  
4. **classic / 近邻混淆对比随机对更能预测组合 OOD。**  
5. **无同态时，可执行域 Adv 与 NL 分支相关接近 0。**

这些预言直接对应我们已见现象，并给出下一轮实验该测什么。

---

## 9. 与相关路线的边界

| 路线 | ICRM 多出来的要求 |
|------|-------------------|
| Outcome / RLVR | 必须过干预电池与 Adv |
| Process / PRM | 步骤对 ≠ 干预因果角色；需冻结续推协议 |
| CCRO-FTB | 用分离完备+扩张替换 FTB-LRR；加入捷径优势公理 |
| IIT / 因果抽象 | 加上有限电池构造、组合界与捷径类 |

---

## 10. 论文级贡献表述（若采用 ICRM）

**一句话：**  
我们提出干预完备推理机制理论：在受控状态通道上，用对混淆类分离完备的有限干预电池度量局部续推残差，并以相对答案匹配捷径类的机制优势作为“真正学会推理（执行层）”的可证伪判据；在此判据下，冻结续推归因与组合混合界给出可检验的 L1–L2 证书，并明确无机制同态时不可向自然语言外推。

**不宣称：** 已解决自主规划（L3）、一般智能、或无结构自然语言的无条件证书。

---

## 11. 闭合：够不够？新理论给了什么？

- **原 CCRO/CCRO-FTB：** 局部工具正确，但对“真正推理”**不够**——假设过强、反捷径缺失、范围过窄。  
- **ICRM：** 基于实验失败模式重写判据：  
  **分离完备干预电池 + 相对捷径的机制优势 + 冻结归因 + 组合界 + 同态才可迁移。**  

这使“真正学会推理”从口号变成**相对明确对象的可证伪命题**，并自然解释我们为何在 classic/ζ/MATH/Outcome 追上之处反复受挫，以及下一步理论与实验应改哪里。
