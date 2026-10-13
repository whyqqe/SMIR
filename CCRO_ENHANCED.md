# CCRO-FTB：有限续推测试基上的因果续推残差优化

**英文全称：** Causal Continuation Residual Optimization with Finite Test Bases（CCRO-FTB）  
**文档性质：** 增强版完整理论方案（面向顶会贡献边界）  
**相对原 CCRO.md 的增强目标：** 把“可发表的新东西”从工具拼装中剥离出来——**有限测试基证书 + 残差归因训练 + 可识别性联合学习**——并明确哪些只是已有技术。

---

## 0. 一句话主张

> 在具有稀疏局部依赖的可执行推理族上，局部更新错误应由其对**未来合法续推行为**的影响度量；若存在低成本有限续推测试基，使得完整续推响应可由测试响应有界重构，则可通过优化测试基上的续推残差，并在证书条件下控制未见组合上的执行误差。

这不是“多做状态监督”，也不是“过程奖励换皮”。核心新组合是：

1. **行为等价残差** \(\delta\)（非潜在距离）；  
2. **冻结续推、保留输入梯度**的局部归因；  
3. **有限测试基重构假设**下的组合证书；  
4. **测试可识别性与机制**的联合学习。

---

## 1. 贡献边界（增强后必须这样写）

### 1.1 已有基础（不宣称新）

| 工具 | 角色 |
|------|------|
| 混合执行 / 三角不等式误差分解 | 标准证明技术 |
| Pinsker：KL → TV | 标准不等式 |
| 熵正则分布鲁棒风险 | 已有 DRO 工具 |
| 原始–对偶约束更新 | 已有优化工具 |
| 干预 / 因果抽象（IIT 等） | 相关先验工作 |
| 过程监督、PRM、状态追踪 | 相关经验方法 |

### 1.2 本文主张的新增贡献

**C1（理论，条件性）。** 给出有限续推测试基（FTB）下的执行误差证书：若完整续推概率可由 \(K\) 个测试响应以有界 \(\ell_1\) 系数重构，则测试基上的最大残差控制任意允许续推上的行为距离，并进一步控制给定操作序列的输出 TV 误差。

**C2（方法）。** CCRO 局部训练：对当前更新 \(F_\theta\) 优化相对正确状态编码分支的续推 KL；续推模块参数冻结但**保留对当前状态输入的梯度**，使错误归因到当前更新而非被后续参数掩盖。

**C3（方法+问题）。** 联合学习：在语义区分度 \(\kappa(\mu)\) 约束下扩展/重加权测试分布 \(\mu\)，并与残差优化交替进行；给出可检验的“可识别性不足 ⇒ 假低残差”诊断。

**C4（经验协议，非口号）。** 在可执行域上验证：同答案准确率下更低残差 ⇒ 更强组合 OOD；报告重构误差 \(\zeta\)、覆盖与证书失效模式。半真实域仅作为外部效度，不替代机制证书。

### 1.3 明确不主张的内容

- 不主张对任意 LLM、任意自然语言 CoT 给出无条件推理证书。  
- 不主张自主规划（操作选择）已由执行证书自动解决（见第 10 节扩展）。  
- 不主张 DRO / 对偶更新本身是理论创新。  
- 不把“显式文本状态 SFT 有效”等同于 CCRO-FTB 已被验证。

---

## 2. 形式化设定

### 2.1 语义任务系统

固定规则 \(r\) 与目标 \(g\)：

- \(\mathcal S\)：语义状态空间；  
- \(\mathcal A\)：合法操作空间；  
- \(f_r(s,a)\in\mathcal S\)：确定性真实转移（随机情形需另写耦合）；  
- \(s_0(x)\)：由问题得到的初始语义状态；  
- \(\mathcal C\)：允许的完整续推集合（操作序列 + 终端查询）；  
- \(y^*(s,c)\)：从状态 \(s\) 执行续推 \(c\) 的正确输出。

非法操作映射到显式无效状态 \(\bot\)，不进入证书。

### 2.2 神经系统与受控状态通道

- \(E_\theta(s)\)：正确语义状态编码器（训练时可用 oracle 状态）；  
- \(F_\theta(z,a)\)：神经状态更新；  
- \(P_\theta(Y\mid z,c)\)：从内部状态执行续推 \(c\) 的输出分布；  
- \(M_\theta(x)\)：从原始问题到初始内部状态（可与 \(E_\theta(s_0)\) 不同）。

**关键实现约束（增强点）：** 必须存在**受控状态通道**：评估与训练时，比较的两条分支除被测状态外，其余可变信息一致；不得把原题全文作为旁路重新注入，使模型绕过状态。

对证书评估，优先使用二值（或有限标签）查询，以便 TV / 概率可测。复杂输出应拆成保留任务信息的有限事件。

### 2.3 操作性成功定义

模型“学会局部执行机制”当且仅当：在认证域 \(\mathcal Z_{\mathrm{adm}}\times\mathcal A_{\mathrm{adm}}\times\mathcal C_{\mathrm{adm}}\) 内，对合法操作序列，其输出行为与语义 oracle 一致到指定误差，并在未见但同族的操作组合上保持该误差界所允许的外推。

---

## 3. 因果续推残差

### 3.1 定义

对 \((s,a,c)\)：

\[
z_{\mathrm{pred}}=F_\theta\big(E_\theta(s),a\big),\qquad
z_{\mathrm{ref}}=E_\theta\big(f_r(s,a)\big),
\]

\[
p_{\mathrm{pred}}=P_\theta(Y\mid z_{\mathrm{pred}},c),\qquad
p_{\mathrm{ref}}=P_\theta(Y\mid z_{\mathrm{ref}},c).
\]

**局部因果续推残差：**

\[
\boxed{
\delta_\theta(s,a,c)=\mathrm{TV}(p_{\mathrm{pred}},p_{\mathrm{ref}}).
}
\tag{1}
\]

对离散输出 \(\mathrm{TV}(p,q)=\tfrac12\sum_y|p(y)-q(y)|\)。二值时等于两分支“真”概率差的绝对值。

### 3.2 性质与非性质

- \(\delta\) 是**行为**量，对保持输出核不变的潜在再参数化不变。  
- \(\delta=0\) **不蕴含**语义正确：两分支可一致地错。  
- 单个弱测试上 \(\delta\) 小 **不蕴含**机制正确：测试可能不可识别。

### 3.3 锚定损失（正确性配角，非主创新）

\[
\mathcal L_{\mathrm{anchor}}(\theta)
=\mathbb E_{s,c}\big[-\log P_\theta(y^*(s,c)\mid E_\theta(s),c)\big].
\tag{2}
\]

标签必须来自 oracle，不能用模型自洽代替。

---

## 4. 给定操作序列的混合执行界

### 4.1 命题 A（混合执行界）

设 \(a_1,\ldots,a_T\) 合法，\(s_t=f_r(s_{t-1},a_t)\)，\(c_t\) 为第 \(t\) 步后的剩余续推。定义

\[
H_t=P_\theta(Y\mid E_\theta(s_t),c_t),\quad t=0,\ldots,T,
\]

并令 \(\epsilon_{\mathrm{read}}=\mathrm{TV}(H_T,P^*)\)，其中 \(P^*\) 为真实最终输出分布。则

\[
\boxed{
\mathrm{TV}(H_0,P^*)
\le
\epsilon_{\mathrm{read}}+\sum_{t=1}^{T}\delta_\theta(s_{t-1},a_t,c_t).
}
\tag{3}
\]

（可再截断为 \(\le 1\)。）

**证明。** 三角不等式展开 \(\mathrm{TV}(H_0,P^*)\le\mathrm{TV}(H_T,P^*)+\sum_t\mathrm{TV}(H_{t-1},H_t)\)；相邻项恰为式 (1)。□

### 4.2 初始化误差

\[
\epsilon_{\mathrm{init}}(x,c_0)
=\mathrm{TV}\big(
P_\theta(Y\mid M_\theta(x),c_0),\,
P_\theta(Y\mid E_\theta(s_0(x)),c_0)
\big).
\]

\[
\mathrm{TV}\big(P_\theta(Y\mid M_\theta(x),c_0),P^*\big)
\le
\epsilon_{\mathrm{init}}+\epsilon_{\mathrm{read}}+\sum_{t=1}^{T}\delta_t.
\tag{4}
\]

### 4.3 该命题的定位

- **说明：** 局部残差与整链输出误差的可加联系。  
- **不说明：** 有限训练测试已覆盖部署；优化残差必收敛；自主选动作已解决。

---

## 5. 可微代理与冻结续推归因（方法核心）

### 5.1 KL 代理与 Pinsker

\[
\ell_\theta(s,a,c)=D_{\mathrm{KL}}(p_{\mathrm{ref}}\Vert p_{\mathrm{pred}}).
\tag{5}
\]

\[
\delta_\theta(s,a,c)\le\sqrt{\ell_\theta(s,a,c)/2}.
\tag{6}
\]

### 5.2 目标模型与局部损失

引入缓慢更新目标 \(\bar\theta\)：

\[
p_{\mathrm{ref}}=P_{\bar\theta}\big(Y\mid E_{\bar\theta}(f_r(s,a)),c\big),
\]
\[
p_{\mathrm{pred}}=P_{\bar\theta}\big(Y\mid F_\theta(E_{\bar\theta}(s),a),c\big).
\]

\[
\mathcal L_{\mathrm{local}}
=D_{\mathrm{KL}}\big(\mathrm{sg}[p_{\mathrm{ref}}]\Vert p_{\mathrm{pred}}\big).
\tag{7}
\]

参考分支必须定期用 oracle 锚定，防止“稳定但错误”的目标。

### 5.3 命题 B（冻结续推的局部梯度形式）—增强表述

在局部更新中强制：

1. 输入参考编码 \(\mathrm{sg}[E_{\bar\theta}(s)]\)；  
2. 仅 \(F_\theta\)（及必要时当前读出头的允许部分）可训练；  
3. 续推前向使用冻结参数 \(\bar\theta\)；  
4. **计算图保持对输入状态 \(z'=F_\theta(\cdot)\) 可微**；  
5. 参考概率停止梯度。

则

\[
\boxed{
\nabla_\theta\mathcal L_{\mathrm{local}}
= J_\theta F_\theta(z,a)^\top\,
\nabla_{z'}\mathcal L_{\mathrm{local}}.
}
\tag{8}
\]

其中 \(\nabla_{z'}\mathcal L_{\mathrm{local}}\) 编码“后续任务对当前状态误差的敏感性”。

**相对普通 process SFT 的方法区分点：** 禁止通过改后续参数在同一步掩盖当前更新错误；归因强制经过当前状态通道。

**实现警告：** 不能对整个续推包 `no_grad()`；应冻结参数副本/stop-grad 参数，同时保留对 \(z'\) 的梯度。

### 5.4 目标漂移

评估证书时用当前 \(\theta\) 重测。若续推核相对 \(\bar\theta\) 有 TV 漂移 \(\omega_{\mathrm{pred}},\omega_{\mathrm{ref}}\)，则

\[
\delta_\theta\le\delta_{\mathrm{target}}+\omega_{\mathrm{pred}}+\omega_{\mathrm{ref}}.
\tag{9}
\]

---

## 6. 测试可识别性

### 6.1 语义区分度

对测试分布 \(\mu\) 与需区分的状态对集 \(\mathcal P\)：

\[
d_\mu^*(s,s')=\Pr_{c\sim\mu}[y^*(s,c)\ne y^*(s',c)],
\]

\[
\boxed{
\kappa(\mu)=\inf_{(s,s')\in\mathcal P}d_\mu^*(s,s').
}
\tag{10}
\]

\(\kappa=0\) ⇒ 存在机制错误无法被当前测试暴露 ⇒ **低残差无意义**。

### 6.2 命题 C（条件性识别界，二值有限测试）

令签名 \(\sigma^*(s)=(y^*(s,c))_{c\in\mathcal C_{\mathrm{test}}}\)。若不同相关状态签名的最小 Hamming 距离 \(\ge\kappa>0\)，模型硬签名经最近合法投影得 \(\hat s\)，平均交叉熵为 \(\eta\)，则

\[
\Pr(\hat s\ne s)\le\min\Big\{1,\frac{2\eta}{\kappa\log 2}\Big\}.
\tag{11}
\]

该界说明 \(\kappa\) 的作用，**不是**任意连续潜在状态的无条件识别保证。

### 6.3 主动测试设计

\[
\max_{\mu\in\Delta(\mathcal C_{\mathrm{test}})}
\big[\kappa(\mu)-\nu\mathbb E_\mu\mathrm{Cost}(c)\big].
\tag{12}
\]

状态对优先取：模型混淆对、单测不可分对、部署敏感对。标签仅来自 oracle。

---

## 7. 有限续推测试基（FTB）与组合证书 — 理论核心增强

### 7.1 理想行为距离

\[
d_{\mathcal C,\theta}(z,z')
=\sup_{c\in\mathcal C}
\mathrm{TV}\big(P_\theta(Y\mid z,c),P_\theta(Y\mid z',c)\big).
\tag{13}
\]

不可直接计算；目标是用有限测试控制它。

### 7.2 结构假设（FTB-LRR：有界线性响应重构）

对二值（或事件）读出 \(q_\theta(z,c)=P_\theta(Y=1\mid z,c)\)，测试基 \(c_1,\ldots,c_K\)，签名

\[
\sigma_\theta(z)=(q_\theta(z,c_1),\ldots,q_\theta(z,c_K)).
\]

**假设 FTB-LRR。** 存在 \(B,\zeta\)，使得对所有 \(z\in\mathcal Z_{\mathrm{adm}}\)、\(c\in\mathcal C\)：

\[
q_\theta(z,c)=b_c+\alpha_c^\top\sigma_\theta(z)+r_c(z),
\quad
\|\alpha_c\|_1\le B,\quad |r_c(z)|\le\zeta.
\tag{14}
\]

### 7.3 命题 D（有限测试控制完整行为距离）

在假设 FTB-LRR 下，对任意 \(z,z'\in\mathcal Z_{\mathrm{adm}}\)：

\[
\boxed{
d_{\mathcal C,\theta}(z,z')
\le
B\max_{1\le j\le K}|q_\theta(z,c_j)-q_\theta(z',c_j)|+2\zeta.
}
\tag{15}
\]

**证明。** 常数项相消；Hölder 得
\(|q(z,c)-q(z',c)|\le\|\alpha_c\|_1\|\sigma(z)-\sigma(z')\|_\infty+|r_c(z)|+|r_c(z')|\)；对 \(c\) 取上确界。□

### 7.4 命题 E（主证书：测试基 KL ⇒ 长链输出界）

若对每步操作 \(a_t\)，测试基上 KL 一致 \(\le\rho_{a_t}\)，且剩余续推属于 \(\mathcal C\)，状态落在认证域，则由 (6)(15) 与命题 A：

\[
\sup_{c\in\mathcal C}\delta_\theta(s,a,c)\le B\sqrt{\rho_a/2}+2\zeta,
\tag{16}
\]

\[
\boxed{
\mathrm{TV}(P_\theta,P^*)
\le
\epsilon_{\mathrm{init}}+\epsilon_{\mathrm{read}}
+\sum_{t=1}^{T}\big(B\sqrt{\rho_{a_t}/2}+2\zeta\big).
}
\tag{17}
\]

**增强表述的要点：** 证书的“新”不在三角不等式，而在 **把不可计算的 \(d_{\mathcal C}\) 换成可优化的有限测试残差**，并显式暴露 \((B,\zeta)\) 作为科学对象：必须估计、必须报告失效。

### 7.5 必须研究的科学问题（论文理论主线）

1. 哪些推理族使 FTB-LRR 以小 \(K\)、可控 \(B\) 成立？（稀疏依赖 / 有限可观测商）  
2. 测试基是否覆盖**模型可达**异常状态，而非仅 oracle 编码状态？  
3. \(\zeta\) 是否随续推长度增长？如何监控？  
4. 有限样本下 \(B,\zeta\) 的保守估计与多重比较。  
5. 测试基–模型交替扩展的停止规则与假阴性风险。

**诚实声明：** “存在有限基” alone 不蕴含高效；维度可能与状态数同阶。

### 7.6 基扩展原则（可操作）

```text
短测试库 → 训编码/更新 → 搜索不可重构续推
→ oracle 合法性与标签 → 扩展基 → 重估 (κ,B,ζ)
```

有限数据未发现反例 = 经验支持 ≠ 完整证明。完整证书需要结构定理或有限域穷举条件。

---

## 8. 分布鲁棒残差与约束预算（工程必要，非主创新）

### 8.1 熵正则鲁棒风险

\[
\mathcal R_\tau(\theta)
=\max_q\big[\mathbb E_q\ell_\theta-\tau D_{\mathrm{KL}}(q\Vert\mu)\big]
=\tau\log\sum_c\mu(c)\exp(\ell_c/\tau).
\tag{18}
\]

\[
q^*(c)\propto\mu(c)\exp(\ell_c/\tau).
\tag{19}
\]

\(\nabla_\theta\mathcal R_\tau=\mathbb E_{c\sim q^*}\nabla_\theta\ell_c\)（权重作常数；或对 logsumexp 直接反传）。

\[
\max_c\ell_c\le\mathcal R_\tau+\tau\log(1/\mu_{\min}).
\tag{20}
\]

(20) 只控制**当前有限库**最坏项；进入 (17) 仍需 FTB-LRR。

### 8.2 约束优化

\[
\min_\theta\ \mathcal L_{\mathrm{goal}}
\quad\mathrm{s.t.}\quad
\mathcal R_a(\theta)\le\rho_a,\ 
\mathcal L_{\mathrm{anchor}}\le\rho_0,\ 
\kappa(\mu)\ge\kappa_0.
\tag{21}
\]

原始–对偶更新乘子 \(\lambda_a\)；非凸下不保证全局收敛。组合长度预算：

\[
\sum_a n_a\big(B\sqrt{\rho_a/2}+2\zeta\big).
\tag{22}
\]

若 \(\mathcal R_a\) 仅为平均风险，不能直接当统一 \(\rho_a\)。

---

## 9. 覆盖、统计与证书卫生

### 9.1 分布迁移

若 \(d_{\mathrm{dep}}/d_{\mathrm{ver}}\le C\) 在支持上成立，则 \(\mathbb E_{\mathrm{dep}}\delta\le C\mathbb E_{\mathrm{ver}}\delta\)。未覆盖状态 ⇒ 无有限 \(C\) ⇒ **不得外推证书**。

### 9.2 模型自生状态

可在 oracle 后台跟踪模型动作对应的真实状态，但不得把该状态偷喂回推理分支并与“单步正确执行”损失混用。分别报告：单步残差、自生状态续推、显式修复。

### 9.3 测量对象

TV/KL 是分布量。自由文本必须定义事件抽取；抽取错误计入测量误差。自适应选测可用于训练；**最终证书在预留状态/测试上重测**。样本均值 \(\hat\zeta\) ≠ 式 (14) 的一致上界，除非另有有限域论证。

---

## 10. 自主操作选择（扩展，非主文必证）

执行证书 \(\neq\) 规划证书。闭环混合需参考策略 \(\pi^*\)、策略 TV \(\beta_\theta\) 与闭环残差 \(\delta^{\mathrm{cl}}\)。主文默认只认证**给定/强制执行序列**；自主求解单独成章，缺数据则降级为讨论。

---

## 11. 完整训练流程（增强版）

### 阶段 A — 锚定  
训 \(E_\theta\) 与读出，使参考分支在短续推上 oracle 正确率达标。

### 阶段 B — 局部残差（方法必做）  
对 \((s,a)\) 建双分支；**冻结续推 + 输入梯度**优化 (7)；定期刷新 \(\bar\theta\) 并重验锚定。

### 阶段 C — 测试基扩展（理论对应）  
收集混淆对；oracle 搜索区分续推；更新 \(\mu\) 使 \(\kappa\) 上升；在留出长续推上估 \(\zeta,B\)。

### 阶段 D — 鲁棒与约束  
用 (18)–(21) 加压高残差测试与紧操作预算。

### 阶段 E —（可选）目标完成  
在执行约束下优化 \(\mathcal L_{\mathrm{goal}}\)；与固定序列认证分表报告。

### 概念伪代码

```python
# 概念代码：frozen_forward 冻结参数，保留对输入状态梯度

for round in range(R):
    # A: oracle 锚定
    L_anchor = oracle_supervised_continuation_loss(...)

    # B: 局部残差
    for s, a in sample_state_actions():
        s_next = oracle_transition(s, a)
        z0 = stop_grad(target_encode(s))
        z_ref = stop_grad(target_encode(s_next))
        z_pred = model.transition(z0, a)          # 可训练
        tests = sample_from(test_bank, mu)
        ell = []
        for c in tests:
            p_ref = stop_grad(frozen_forward(target, z_ref, c))
            p_pred = frozen_forward(target, z_pred, c, input_grad=True)
            ell.append(KL(p_ref, p_pred))
        L_local += entropic_risk(ell, mu, tau)    # 或均匀平均作消融

    L = L_goal + λ0*(L_anchor-ρ0) + sum λa*(Ra-ρa)
    update(θ); update(λ); maybe_refresh(target)

    # C: 扩展测试基 + 估 (κ,B,ζ)
    if expand_due:
        pairs = collect_confusions()
        bank, mu = oracle_expand_distinguishing(bank, mu, pairs)
        estimate_reconstruction_on_heldout_long_continuations()

    # 证书：用当前 θ 在预留集上重测 δ、ζ、组合长度曲线
```

---

## 12. 可检验预测与必做对照（增强后的经验契约）

### 12.1 预测

1. **同答案准确率**下，更低测试基残差 ⇒ 更高组合 OOD / 更长链可靠性。  
2. \(\kappa\) 过低时，残差可被优化到接近 0 但机制仍错（假证书）。  
3. 去掉冻结续推（允许后续参数同训）会提高投机解率：当前更新错、后续补锅。  
4. \(\zeta\) 增大时，式 (17) 变松，实测组合误差与上界关系恶化。

### 12.2 必做对照

| 对照 | 目的 |
|------|------|
| Outcome-only | 排除最终答案捷径 |
| Process / 状态 SFT（无 δ、无冻结续推） | 隔离方法增量 |
| 随机测试 vs 可识别测试 | 隔离 \(\kappa\) |
| 均匀残差 vs DRO 加权 | 隔离鲁棒项（次要） |
| 当前模型重测 vs 仅目标模型 | 隔离漂移 |

### 12.3 主图表（顶会最小集）

1. 残差–组合长度曲线（含证书上界与实测误差）；  
2. \(\kappa\)–假低残差诊断；  
3. 同准确率、不同机制的分支分离；  
4. \((B,\zeta)\) 估计与失效案例；  
5. 相对 Process-SFT / Outcome 的消融。

半真实（MATH 等）只报告**真 distinguishing oracle 探针**；启发式 `label_wrong` 不得进入主表。

---

## 13. 与原 CCRO.md 的差异摘要

| 项目 | 原方案 | 增强版 CCRO-FTB |
|------|--------|-----------------|
| 中心贡献 | 框架较全、边界偏宽 | **FTB 证书 + 残差归因 + κ 联合学习** |
| 方法最低实现 | 易被做成状态 SFT | **强制 δ + 冻结续推输入梯度** |
| 理论主定理 | 多命题并列 | **命题 D/E 作为主证书，其余为支撑** |
| 自主规划 | 同文易过度承诺 | **明确降级为扩展** |
| 经验契约 | 原则性 | **规定对照与主图，禁止弱代理冒充** |

---

## 14. 闭合陈述

CCRO-FTB 的可发表理论不是“我们用了续推”，而是：

> **在可陈述的 FTB-LRR 结构假设下，有限可识别续推测试上的局部残差，足以有界控制认证域内任意允许续推的行为距离，并由此给出给定操作序列的输出误差证书；训练上通过冻结续推的输入梯度归因与测试基联合扩展，使该证书对应的量可优化、可失效诊断。**

若实验只验证文本状态 SFT，则支持的是问题动机，**不支持**上述定理式主张。增强后的理论要求方法与评价与第 5、7、12 节对齐。
