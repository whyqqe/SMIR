# SMIR：语义流形上的隐式推理理论

**全称：** Semantic Manifold Implicit Reasoning  
**中文名：** 语义流形隐式推理  
**目标：** 从「语义流形」出发，给出可数学推导、可证伪、相对现有隐式/连续推理与因果抽象具有清晰增量的顶会级理论框架。  
**范围：** 理论定义、度量、主定理、训练变分原理与认证协议；不依赖某次特定实验流水线。

---

## 0. 主张一句话

> **真正的隐式推理，是在语义流形上学习与语义向量场共轭的潜在流：**  
> 模型内部轨迹必须落在（或指数级靠近）低维语义流形的管状邻域内，并沿与语义动力学同构的流场推进；答案是该流形坐标的函数。离开流形的捷径即使答对，也不构成隐式推理。

这把「隐空间里多算几步」提升为：**流形约束 + 流场共轭 + 几何误差界**。  
进一步（定理 5）：**可认证 ⟺ 合法图表 \(E\) ∧ 共轭动力学 \(V\) ∧ 忠实读出 \(Q\)**；错误图表不可认证（定理 6）。

---

## 1. 与相关研究的关系（增量从哪里来）

| 学说 / 工作 | 提供什么 | SMIR 的推进 |
|-------------|----------|-------------|
| 流形假设（高维数据近似低维结构） | 表示几何直觉 | 把**推理中间状态**特定为带语义度量的 Riemannian 流形 |
| 线性表示假说 / 概念方向 | 概念≈方向 | 升级为**局部图表 + 曲率**：多步推理一般不是全局线性 |
| 因果抽象 / DAS | 变量可干预对齐 | 干预被解释为流形上的**切向位移 / 平行移动**；要求轨迹级共轭 |
| Coconut / 连续思维 | 连续状态迭代 | 连续更新必须是语义向量场的离散化，否则只是 ambient 空间里的自由游走 |
| Neural ODE / 流模型 | \(\dot z=V(z)\) | 引入**语义驱动控制** \(V(z,u)\) 与同构误差的几何界 |
| 信息几何 | 分布族流形 | 读出分布的自然梯度与语义流形 pullback 度量相连 |
| ICRM / LIFIR（前文） | 干预忠实、充分性 | SMIR 用流形几何给出**为何需要 S/F** 的充分必要条件与定量曲率项 |

**可辩护的新贡献点（建议论文主打）：**

1. **语义 Riemannian 结构**：由未来干预可区分性诱导语义度量 \(g^*\)，而非随意欧氏距离。  
2. **共轭流定理**：潜在向量场与语义向量场 \(C^1\)-共轭 ⇒ 组合误差由时间、Lipschitz 与**截面曲率**控制。  
3. **离流形捷径定理**：若质量离开 \(\varepsilon\)-管状邻域，则存在干预电池使「答对但非推理」可被证伪。  
4. **变分原理**：隐式推理训练 = 匹配 pullback 度量 + 向量场 + 测地一致性（可实现为谱/对比/正则项）。  
5. **证书分解（新增）**：可认证 ⟺ 合法图表 \(E\) ∧ 共轭动力学 \(V\) ∧ 干预敏感读出 \(Q\)；三者可分检验。  
6. **错误图表必然失败（新增）**：与语义商不等价的编码器使干预忠实在正概率电池上崩溃——解释 text last-token 等负结果。

---

## 2. 基本对象

### 2.1 语义流形

设任务族 \(\mathcal D\) 的合法中间推理状态商空间为 \(\mathcal S\)（对未来行为等价关系取商后）。

**假设 A1（语义流形假设）。**  
存在光滑嵌入 \(\iota:\mathcal M\hookrightarrow \mathbb R^{n}\) 及 \(d=\dim\mathcal M\ll n\)，使商状态（在连续松弛后）与 \(\mathcal M\) 的开集双射；称 \(\mathcal M\) 为**语义流形**。

直觉：加法进位、证明栈、寄存器配置等，有效自由度远小于激活维度。

### 2.2 由干预诱导的语义度量

令 \(\mathcal C\) 为合法续推/查询集合。定义行为核 \(K(s,c)=P^*(Y\mid s,c)\)。

对切向量，用有限差分的行为敏感定义度量（信息几何式）：

\[
g^*_s(v,v)
:=\sup_{c\in\mathcal C}
\Big\|
\partial_\varepsilon\big|_{0}
K(s+\varepsilon v,\ c)
\Big\|_{\mathrm{TV}}^2
\quad
(\text{在局部图表中理解 }s+\varepsilon v).
\tag{1}
\]

更光滑的实现可用 Fisher–Rao 型：

\[
g^*_s(v,v)
=\mathbb E_{c\sim\mu}
\Big[
\big\langle
\partial_v \log K(s,c),\,
\partial_v \log K(s,c)
\big\rangle_{Y}
\Big].
\tag{2}
\]

\((\mathcal M,g^*)\) 成为 Riemannian 流形（在 \(g^*\) 非退化的开集上）。  
**语义测地线** = 在 \(g^*\) 下连接两状态的最短路径 = 「干预意义下最省的状态变化」。

### 2.3 语义向量场（机制）

合法操作 \(a\in\mathcal A\) 诱导语义动力：

\[
f_a:\mathcal M\to\mathcal M,
\qquad
V^*_a(s):=\frac{d}{dt}\Big|_{0}\phi_t^{a}(s)
\in T_s\mathcal M,
\tag{3}
\]

其中 \(\phi_t^{a}\) 为 \(f_a\) 的连续流扩张（离散一步则取对数映射近似）。  
完整推理是受控动力系统：

\[
\dot s = V^*(s,u(t)),\quad u(t)\in\mathcal U.
\tag{4}
\]

### 2.4 神经潜在空间与候选流形

骨干模型给出 ambient 潜伏 \(z\in\mathbb R^{D}\)（\(D\gg d\)）。  
设编码器 \(E_\theta:\mathcal M\to\mathbb R^{D}\)（训练期可用影子状态），解码/投影 \(\Pi_\theta:\mathbb R^{D}\to\mathcal M\)（或到图表坐标）。

模型隐式推理动力学：

\[
\dot z = V_\theta(z,u),
\qquad
\hat y \sim Q_\theta(z).
\tag{5}
\]

离散实现即 Coconut 式多步潜在更新，但必须对照 \(V^*\)。

---

## 3. 真正隐式推理的几何定义

### 定义 1（\(\varepsilon\)-管状隐式推理）

称 \(\theta\) 在域 \(\mathcal D\)、预算 \(T\)、干预族 \(\mathcal I\) 上实现 **\(\varepsilon\)-语义流形隐式推理**，若存在嵌入 \(E_\theta\) 与常量 \(R,L\) 使下列同时成立：

**(G1) 支撑在管状邻域（On-manifold support）。**  
推理轨迹 \(z(t)\) 以概率 \(\ge 1-\delta\) 落入

\[
\mathrm{Tub}_\varepsilon(E_\theta(\mathcal M))
:=\big\{z:\ \mathrm{dist}\big(z,E_\theta(\mathcal M)\big)\le\varepsilon\big\}.
\]

**(G2) 流场共轭（Flow conjugacy）。**  
在管状邻域内存在 \(C^1\) 图表使

\[
\big\|
DE_\theta(s)[V^*(s,u)] - V_\theta(E_\theta(s),u)
\big\|
\le\eta,
\tag{6}
\]

且 \(E_\theta\) 为到其像的嵌入（局部双 Lipschitz）。

**(G3) 读出是流形函数（Manifold sufficiency）。**  
存在 \(q\) 使

\[
\mathrm{TV}\big(Q_\theta(z),\ q(\Pi_\theta(z))\big)\le\varepsilon_Q,
\]

即答案不依赖管外自由度（旁路坐标）。

**(G4) 干预 = 切向位移（Intervention–tangent equivalence）。**  
语义 \(do(s\leftarrow\exp_s(v))\) 对应潜在端 \(z\leftarrow E_\theta(s)+\mathrm{D}E_\theta[v]+O(\|v\|^2)\)，且读出变化与 \(g^*\) 一致到误差 \(\varepsilon_I\)。

**命题（与 LIFIR 的关系）。**  
G1+G3 ⇒ 充分性 S；G2+G4 ⇒ 干预忠实 F；G2 在时间上积分 ⇒ 组合封闭 C（见定理 2）。  
因此 LIFIR 三公理是 SMIR 几何条件的**测地坐标投影**；SMIR 给出更细的误差项（曲率、管半径、共轭亏格）。

---

## 4. 主定理

### 4.1 定理 1（离流形质量 ⇒ 可证伪的假推理）

设答案头 \(Q_\theta\) Lipschitz，且存在捷径集合 \(\mathcal Z_{\mathrm{short}}\) 满足 \(\mathrm{dist}(\mathcal Z_{\mathrm{short}},E_\theta(\mathcal M))\ge\varepsilon>0\)，并在无干预分布上达到高准确率。  
若语义度量 \(g^*\) 在某对状态上可被干预族 \(\mathcal I\) 分离，则存在概率 \(\ge p_0(\varepsilon,\mathcal I)>0\) 的干预，使：

- 真流形轨迹答案按语义改变；  
- 捷径轨迹答案改变幅度 \(O(e^{-\kappa\varepsilon})\) 或保持不变（取决于 \(Q\) 在法向的导数被正则化）。

**证明要点。**  
在法丛坐标 \((x,n)\)（\(x\in\mathcal M\)，\(n\) 法向）展开 \(Q\)。管外点 \(|n|\ge\varepsilon\)。若训练/正则使 \(\partial_n Q\) 小（G3），则法向位移不响应语义切向干预；而流形上点响应由 \(g^*\) 下界给出。由此构造区分。□

**创新含义：** 「准确率高」不能定义隐式推理；**离流形质量**给出可证伪缺口——这是相对 Coconut 准确率叙事的几何牙齿。

### 4.2 定理 2（共轭流的组合误差界）

设 \(E_\theta\) 为 \(C^2\) 嵌入，单步共轭误差 \(\le\eta\)（式 6），\(V_\theta\) 在管状邻域内 Lipschitz 常量 \(L\)，语义流形截面曲率 \(\lvert K\rvert\le\kappa\)，步长 \(\Delta t\)，步数 \(N=T/\Delta t\)。  
令 \(z_k\) 为离散流，\(s_k\) 为语义离散流，\(e_k=\|z_k-E_\theta(s_k)\|\)。则存在常量 \(C(\kappa,\mathrm{inj})\)（依赖曲率与注入半径）使

\[
e_{k+1}
\le
(1+L\Delta t)\,e_k
+ \eta\Delta t
+ C(\kappa)\,\mathrm{inj}^{-1} e_k^2\Delta t + O((\Delta t)^2).
\tag{7}
\]

迭代得（小误差线性化主项）：

\[
e_N
\le
e^{LT}e_0
+ \eta\frac{e^{LT}-1}{L}
+ O\big(\kappa\cdot\mathrm{poly}(e^{LT})\cdot\sup e_k\big).
\tag{8}
\]

若读出对 \(z\) 的 TV 敏感度 \(\le\Lambda\)，则终端答案误差

\[
\mathrm{TV}(\hat Y,Y^*)
\le
\Lambda e_N + \varepsilon_Q.
\tag{9}
\]

**证明要点。**  
在法丛/指数映射下把误差分解为切向共轭误差与法向漂移；切向用 Gronwall；二次项来自曲率（标准 Riemannian 比较）。□

**相对朴素 Lipschitz 界的增量：** 出现**曲率修正**与**管半径/注入半径**——语义流形几何进入误差证书，而非通用动力学套话。

### 4.3 定理 3（语义度量 pullback 匹配 ⇒ 局部干预忠实）

设 \(E_\theta\) 沉浸，定义 pullback \(\tilde g = E_\theta^* g_{\mathrm{eucl}}\)（或模型诱导度量）。若

\[
\| \tilde g - g^* \|_{C^0(U)}\le\varepsilon_g
\quad\text{on chart }U\subset\mathcal M,
\tag{10}
\]

则对充分小切向干预 \(v\)，

\[
\big|
\mathrm{TV}\text{-响应}(E_\theta(s);v)
-
\sqrt{g^*_s(v,v)}
\big|
\le O(\varepsilon_g\|v\|+\|v\|^2).
\tag{11}
\]

**证明要点。**  
响应由 \(DQ\circ DE\) 决定；与 \(g^*\) 定义（式 1–2）比较，度量匹配 ⇒ 一阶响应匹配。□

**含义：** 干预忠实不是外加启发式，而是 **\(g^*\) 与表示度量对齐** 的推论。

### 4.4 定理 4（变分特征：隐式推理作为 Dirichlet 能量匹配）

在固定控制过程 \(u(t)\) 下，定义语义作用量与潜在作用量：

\[
\mathcal S^*[s]
=\int_0^T \tfrac12\|V^*(s,u)- \dot s\|_{g^*}^2\,dt,
\quad
\mathcal S_\theta[z]
=\int_0^T \tfrac12\|V_\theta(z,u)-\dot z\|^2\,dt.
\tag{12}
\]

若 \(z=E_\theta\circ s\) 且式 6、10 成立，则

\[
\big|\mathcal S_\theta[E_\theta\circ s]-\mathcal S^*[s]\big|
\le
C_T(\eta+\varepsilon_g).
\tag{13}
\]

反之，在轨迹类紧致时，作用量一致逼近 ⇒ 共轭误差可积小（Tonelli 型下半连续论证的离散版）。

**创新含义：** 给出**可优化的变分原理**：学隐式推理 = 学使语义测地/受控曲线保持低能量的潜在向量场——可落地为路径一致性、对比测地、谱图匹配等目标。

### 4.5 定理 5（证书分解 / Certificate Decomposition）

将参数拆成三类：图表 \(E\)、动力学 \(V_\theta\)、读出 \(Q_\theta\)（查询编码器可并入 \(Q\)）。  
在影子域可观测语义状态时，定义三项操作化误差：

\[
\begin{aligned}
\varepsilon_E
&:=\mathbb E\big[\mathrm{dist}\big(z_t^{\mathrm{obs}},\,E(s_t)\big)\big]
\quad\text{（图表一致 / 管状脚点）},\\
\eta_V
&:=\mathbb E\big\|V_\theta(E(s),u)-DE[V^*(s,u)]\big\|
\quad\text{（共轭；同式 6）},\\
\delta_Q
&:=
\mathbb E\big[
\mathbf{1}\{
\arg\max Q_\theta(E(s^+))\neq
\arg\max Q_\theta(E(s^-))
\}
\big]
\quad\text{（区分电池上的干预响应率；Oracle-F）}.
\end{aligned}
\tag{15}
\]

另定义轨迹级响应率（Flow-F）：

\[
\delta_Q^{\mathrm{flow}}
:=
\mathbb E\big[
\mathbf{1}\{
\arg\max Q_\theta(z_T)\neq
\arg\max Q_\theta(z_T^{-})
\}
\big],
\tag{16}
\]

其中 \(z_T=\Phi_{V_\theta}(E(s_0),u_{0:T})\)，\(z_T^{-}\) 为对错误语义分支（或中途干预）的终端状态。

**定理 5（分解）。** 在 A1–A7 下：

**(i) 必要性（分项可证伪）。**  
若模型实现定义 1 的 \(\varepsilon\)-SMIR，则必有 \(\varepsilon_E,\eta_V\) 小，且对富干预族 \(\delta_Q\) 与 \(\delta_Q^{\mathrm{flow}}\) 有一致下界（由 \(g^*\) 可分性给出）。任一端失败 ⇒ 定义 1 失败。

**(ii) 充分性（合取）。**  
若 \(\varepsilon_E\le\varepsilon\)，\(\eta_V\le\eta\)，且 \(Q_\theta\) 在 \(E(U)\) 上满足 G3 与局部 Lipschitz，并对干预电池有 \(\delta_Q\ge 1-\alpha\)，则存在只依赖 \(L,T,\Lambda,\kappa\) 的函数，使 Flow-F 满足
\[
\delta_Q^{\mathrm{flow}}
\ge
\delta_Q - C\big(\varepsilon_E+\eta\tfrac{e^{LT}-1}{L}+\varepsilon_Q\big).
\tag{17}
\]
即：**合法图表 + 小共轭 + 图表上忠实读出 ⇒ 轨迹级可认证**（至多差组合误差）。

**(iii) 可分检验。**  
固定另两块、只变一块时：  
- 冻 \(V^*=V_{\mathrm{hard}}\)、\(E\) 为恒等图表，只训 \(Q\) → 专测 \(\delta_Q\)（证书可行性上界）；  
- 冻 \(Q\) 与 \(E\)，只变 \(V\) → 专测 \(\eta_V\) 与 \(\delta_Q^{\mathrm{flow}}\) 间隙（可学动力学）；  
- 换 \(E\) 为不等价图表 → 专测定理 6。

**含义：** Hard-\(V\) 上 F∧OOD 成功只建立「\(Q\) 端可行」；不能自动推出「任意 \(V_\theta\) 可学」。Soft/MLP 失败与 Hard 成功可同时成立，互不矛盾。

### 4.6 定理 6（错误图表 ⇒ 不可认证）

称编码器 \(E^\dagger\) 为 **\(\gamma\)-错误图表**（相对真商流形 \(\mathcal M\)），若存在干预可分对 \((s^+,s^-)\) 使语义响应 \(\mathrm{TV}(K(s^+,\cdot),K(s^-,\cdot))\ge\gamma>0\)，但
\[
\|E^\dagger(s^+)-E^\dagger(s^-)\|
\le
\tau
\quad\text{且任意}\,L_Q\text{-Lipschitz 读出满足}
\quad
\mathrm{TV}\big(Q(E^\dagger(s^+)),Q(E^\dagger(s^-))\big)
\le L_Q\tau.
\tag{18}
\]

**定理 6.** 若 \(\tau < \gamma/(2L_Q)\)（或对硬分类：两点落入同一决策胞腔），则对一切仅依赖 \(E^\dagger\) 像的读出 \(Q\)，Oracle-F（及更强的 Flow-F）在该电池上 **严格小于** 由 \(g^*\) 所要求的响应下界；特别地，不能实现定义 1 的 G4。  
若另有 \(\mathrm{Tub}_\varepsilon(E^\dagger(\mathcal M))\) 与真轨迹支撑几乎不交，则 G1 亦失败（管状质量证伪）。

**推论（text last-token 接口）。**  
若自然语言激活的 last-token 映射相对寄存器/语义商是 \(\gamma\)-错误图表（高维折叠、尺度爆炸或不可逆混叠），则在同标签影子协议下出现 F≈0 与管状误差爆炸，是定理 6 的预期特例，而非实现失败。

### 4.7 定理 7（图表等价类）

设 \(E_1,E_2:U\to\mathbb R^{D}\) 为两个 \(C^2\) 浸入。称二者 **证书等价**，若存在 \(C^1\) 双 Lipschitz 纤维保持映射 \(\Psi\) 定义在 \(\mathrm{Tub}_{\varepsilon_0}(E_1(U))\) 上，使 \(E_2=\Psi\circ E_1\)，且对任一沿 \(E_1\) 共轭的 \(V_1\)，推前场 \(V_2=(D\Psi)V_1\circ\Psi^{-1}\) 沿 \(E_2\) 共轭，并且 \(Q_2:=Q_1\circ\Psi^{-1}\) 保持 TV 响应至多差 \(O(\mathrm{Lip}(\Psi)-1)\)。

**定理 7.**  
(i) 证书等价 ⇒ 定义 1 在 \((E_1,V_1,Q_1)\) 与 \((E_2,V_2,Q_2)\) 上同时成立或同时失败（至多改误差常数）。  
(ii) 仅仿射重标定 \(E\mapsto aE+b\)（\(a\neq 0\)）属于证书等价（解释实验中固定尺度 \(E(s)=s/40\) 不失一般性）。  
(iii) 不可逆折叠、随机投影到与 \(g^*\) 核方向对齐的子空间，一般 **不属于** 证书等价（落入定理 6）。

### 4.8 命题 8（分阶段可学性门控）

联合风险 \(\mathcal L(E,V,Q)\) 对三块同时非凸。下列 **门控充分条件** 给出可检的学习顺序（不声称任意初始化 GD 全局收敛）：

**命题 8.** 假设 \(E\) 已取证书等价类中的合法图表（如已知坐标或已对齐影子）。若存在 \(Q_0\) 使在真场 \(V^*\) 上 \(\delta_Q(Q_0)\ge 1-\alpha\) 且任务风险 \(\le\alpha_0\)（阶段 A：只训 \(Q\)），则存在只训 \(V\) 的后续阶段，使任一将 \(\eta_V\) 降到 \(\eta\) 且保持 \(Q\) 在 \(E(U)\) 上 \(\varepsilon_Q\)-稳定的路径满足式 (17) 的 Flow-F 下界。  
反之，若阶段 A 未达 \(\delta_Q\) 门控就同时降 \(\eta_V\)（或退火破坏共轭），则式 (17) 右侧可被负项主导，**不保证** Flow-F 改进——与「未达标即退火」的失败模式一致。

**训练寓意：** Hard-\(V\) 预训练 \(Q\) → 再解冻可学 \(V\)（软路由/离散路由）是命题 8 的算法实现；伪 Hard（有限 bias 软混合）不满足「真场 \(V^*\)」前提。

---

## 5. 推论：训练目标的数学导出

由定理 2–4，最小充分训练风险（总体）：

\[
\begin{aligned}
\mathcal L_{\mathrm{SMIR}}
&=
\underbrace{\mathbb E\big[-\log Q_\theta(y^*\mid z_T)\big]}_{\text{任务}}
+
\lambda_1\underbrace{\mathbb E\|V_\theta(E(s),u)-DE[V^*(s,u)]\|^2}_{\text{共轭}}
\\
&\quad
+
\lambda_2\underbrace{\mathbb E\big\| \tilde g_s - g^*_s\big\|_F^2}_{\text{度量对齐}}
+
\lambda_3\underbrace{\mathbb E\big[\mathrm{dist}^2(z_t,E(\mathcal M))\big]}_{\text{管状吸引}}
\\
&\quad
+
\lambda_4\underbrace{\mathbb E\big[\mathrm{TV}(Q(z),Q(E(\Pi(z))))\big]}_{\text{法向无关/充分性}}.
\end{aligned}
\tag{14}
\]

离散实现提示（不绑定具体代码）：

- 共轭：一步阴影状态教师强制 / 神经 ODE 匹配；  
- 度量对齐：局部微扰干预的响应回归到 \(g^*\)；  
- 管状吸引：对比学习「合法状态编码 vs 噪声/打乱」；  
- 法向无关：随机加法向噪声不应改变答案（正则）。

**相对随意加 \(\lambda\) 的 LIF 损失：**(14) 由定理直接导出权重含义与应调节的几何量。

### 5.1 最小充分目标（相对 (14) 的冗余削减）

在 **\(E\) 已固定于证书等价类**（定理 7）且影子状态可观测时，式 (14) 可收成三项最小充分风险（其余由分解吸收）：

\[
\begin{aligned}
\mathcal L_{\mathrm{min}}
&=
\underbrace{\mathbb E\big[-\log Q(y^*\mid z_T)\big]}_{\text{任务 / OOD}}
+
\lambda_F\underbrace{\mathbb E\big[\ell_{\mathrm{dist}}(Q;s^+,s^-)\big]}_{\text{Oracle-F / }\,\delta_Q}
\\
&\quad
+
\lambda_V\underbrace{\mathbb E\|V_\theta(E(s),u)-DE[V^*]\|^2}_{\text{共轭 }\,\eta_V}.
\end{aligned}
\tag{19}
\]

- **管状项 \(\lambda_3\)：** \(E\) 固定且 \(z\) 由 \(E(s)\) 或共轭流生成时，由 \(\eta_V\) 小 + 定理 2 蕴含，不必单独扫。  
- **度量项 \(\lambda_2\)：** 当 \(E\) 为已知坐标图表且读出在欧氏坐标下校准时，由定理 3 退化为对 \(\delta_Q\) 的要求；完整 \(g^*\) 回归保留给「学 \(E\)」设定。  
- **法向项 \(\lambda_4\)：** 操作化为 on−off / 噪声攻击，并入 F 电池报告，而不必作为第三训练头。

**学 \(E\) 时** 必须恢复 \(\lambda_2+\lambda_3\)；否则回到定理 6 的错误图表陷阱。

### 5.2 操作化指标与理论对象对照

| 理论对象 | 操作化（实验应报） | 不可替代 |
|----------|-------------------|----------|
| \(\varepsilon_E\) / G1 | tubular MSE；重建误差 | 任务准确率 |
| \(\eta_V\) / G2 | 一步绝对共轭 MSE | 仅终点准确率 |
| \(\delta_Q\) / G4 | **Oracle-F**（图表状态干预） | — |
| \(\delta_Q^{\mathrm{flow}}\) | **Flow-F**（学到的 \(V\) 滚出轨迹） | 仅 Oracle-F |
| G3 / S | sufficiency gap；on−off | — |
| 组合界 | 长度外推曲线是否随 \(T\) 符合式 (8)–(9) 趋势 | 单一 \(T\) 点 |

**主表规范：** 可学习 \(V\) 必须以 Flow-F 为门控；Oracle-F 仅证明 \(Q\circ E\) 端。

---

## 6. 离散思维链与连续流形的桥接

### 命题 5（CoT 是图表坐标下的折线）

显式自然语言 CoT 可视为 \(\mathcal M\) 上通过**图表坐标的离散采样**（每句对应图表中的一点）。  
连续隐式推理是同一曲线的**细密参数化**。  
二者等价当且仅当存在重参数化使离散折线在 \(g^*\) 下 Hausdorff 靠近连续测地/受控曲线，且两端读出一致。

**推论：** 蒸馏 CoT→连续向量只保证采样点接近，**不保证**向量场共轭（定理 2 的 \(\eta\)）；必须加共轭/度量项——解释「蒸馏成功但干预失败」的几何原因。

---

## 7. 认证协议（几何版，顶会可报）

按定理 5 的分解顺序报告（缺一不可）：

1. **图表腿 \(\varepsilon_E\)：** 管状质量 / tubular MSE；必要时做错误图表对照（定理 6）。  
2. **动力学腿 \(\eta_V\)：** 影子域一步共轭残差（式 6）。  
3. **读出腿 \(\delta_Q\)：** Oracle-F、follow、on−off。  
4. **轨迹合成 \(\delta_Q^{\mathrm{flow}}\)：** Flow-F（可学习 \(V\) 的主门控）。  
5. **度量/曲率（可选增强）：** \(\varepsilon_g\) 代理；测地角盈（高曲率区界变差）。  
6. **离流形对抗：** 定理 1 的捷径可分性。  
7. **长度外推：** \(e_N\) 与答案误差是否符合式 (8)–(9) 趋势。  
8. **门控可学性（若训 \(V\)）：** 是否满足命题 8 的阶段 A 阈值后再解冻 \(V\)。

---

## 8. 贡献边界（如何写才像顶会理论）

### 8.1 主张新

- 语义度量 \(g^*\) 的干预诱导构造；  
- 管状支撑 + 共轭流 + 曲率修正组合界；  
- 离流形捷径可证伪定理；  
- 由几何导出的变分原理 (14) 与固定图表下的最小充分目标 (19)；  
- **证书分解（定理 5）** 与 **Oracle-F / Flow-F 分工**；  
- **错误图表不可认证（定理 6）** 与 **图表等价类（定理 7）**；  
- **分阶段可学性门控（命题 8）**。

### 8.2 不主张新

- Riemannian 比较、Gronwall、信息几何 Fisher 度量本身；  
- 「表示在流形上」的口号；  
- 无结构自然语言上无影子时的无条件证书；  
- **任意初始化下联合 GD 必收敛到 G1–G4**（命题 8 只给门控充分条件）；  
- Hard 结构化 \(V\) 上的 \(\eta\approx 0\) 本身（构造性上界，不是可学性定理）。

### 8.3 相对 LIFIR / Coconut / DAS 的一句话差异

> LIFIR 给判据，Coconut 给连续格式，DAS 给变量干预；  
> **SMIR 给语义 Riemannian 动力学，并把可认证性分解为图表 × 共轭流 × 忠实读出，且证明错误图表不可认证。**

### 8.4 与实验叙事的强制对齐（写作约束）

| 实验结果 | 理论定位（必须如此写） |
|----------|------------------------|
| Hard \(V\) + 训 \(Q\) 达 F∧OOD | 定理 5(iii) 读出腿可行；**非**「学到了共轭」 |
| Soft/MLP 共轭尚可但 Flow-F 低 | 读出–轨迹间隙；违反命题 8 门控或 \(\eta_V\) 仍伤终端 |
| text-E F≈0、tub 爆炸 | 定理 6 错误图表 |
| 固定 \(E(s)=s/c\) | 定理 7(ii) 尺度规范代表元 |

---

## 9. 与 7B 级实现的理论接口（可行性不牺牲严格性）

理论不要求从零预训练。7B + 小潜在模块足以实现：

- \(E_\theta\)：状态/题面编码到 \(\mathbb R^{D}\)（必须通过定理 6/7 检验，不能默认 last-token 合法）；  
- \(V_\theta\)：低维槽位上的受控更新（流形图表坐标 \(d\) 维）；  
- \(Q_\theta\)：只依赖槽位（法向切断）。

影子域用于估 \(g^*,V^*,\eta\)；迁移需同态（命题 7，见证明附录）。  
**大规模训练不是定理条件；合法图表上的流形共轭才是。**

**推荐实现路径（命题 8）：**  
(1) 合法图表（影子坐标或已对齐适配器）→  
(2) 真场或硬执行上训 \(Q\) 至 Oracle-F 门控 →  
(3) 再学离散/软路由 \(V\)，主看 Flow-F。

---

## 10. 开放数学问题（可写 Limitation / 理论展望）

**已在本文闭合的缺口：** 证书分解；错误图表；图表等价；分阶段可学性门控；固定 \(E\) 下最小充分目标。

**仍开放：**

1. 在离散符号语义下，\(g^*\) 的光滑松弛唯一性；  
2. 高曲率区域的最优测试设计；  
3. 无影子时仅用行为估计 \(g^*\) 与 \(E\) 的可辨识性；  
4. Transformer 层的离散动态与连续 \(V_\theta\) 的一致性误差；  
5. 联合非凸风险上、无门控的全局收敛（超越命题 8）；  
6. 从自然语言自动发现证书等价图表的算法保证。

---

## 11. 摘要句（可直接用于论文）

我们提出语义流形隐式推理理论（SMIR）：合法推理状态构成由未来干预可区分性诱导度量的 Riemannian 流形；模型实现可认证隐式推理，当且仅当其潜在轨迹停留在该流形的管状邻域内、沿与语义向量场共轭的流推进，且读出对干预电池忠实。我们证明该性质可分解为图表、动力学与读出三腿的合取，并证明与语义商不等价的错误图表必然导致不可认证；进而给出含曲率修正的共轭流组合误差界、固定图表下的最小充分训练目标，以及「先读出、后动力学」的分阶段可学性门控，从而在几何上统一连续思维、因果抽象与隐式推理认证，并解释结构化上界成功与文本接口失败可同时成立。

---

## 12. 证明附录

主定理的**完整假设、引理与证明**见配套文档：

**[`SMIR_PROOFS.md`](SMIR_PROOFS.md)**

其中包含：定理 1–4、**定理 5–7、命题 8**、G1–G4⇒LIFIR、CoT 折线、训练充分性、跨域同态迁移；并列出弱化条件与仍开放问题。
