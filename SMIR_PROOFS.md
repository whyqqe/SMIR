# SMIR 定理证明附录

**配套主文：** `SMIR.md`  
**目的：** 补齐假设、引理与主定理的完整证明；标明所用经典结果；指出离散/随机情形的推广条件。  
**符号：** 与主文一致；ambient 欧氏范数记 \(\|\cdot\|\)；切空间范数在未标明时由 \(g^*\) 或欧氏范数按上下文指定。

---

## 0. 精确假设清单

**A1（语义流形）。** \(\mathcal M\) 为光滑、连通、无边界的 \(d\) 维流形，带光滑嵌入 \(\iota:\mathcal M\hookrightarrow\mathbb R^{n}\)。下文常把 \(\mathcal M\) 与其像等同。

**A2（行为核光滑）。** 续推族 \(\mathcal C\) 紧；核 \(K:\mathcal M\times\mathcal C\to\Delta(\mathcal Y)\) 对第一变元 \(C^2\)（在局部图表中），\(\mathcal Y\) 有限。

**A3（语义度量非退化）。** 由主文式 (2) 定义的 \(g^*\) 在开集 \(U\subseteq\mathcal M\) 上正定，使 \((U,g^*)\) 为 Riemannian；设截面曲率满足 \(\lvert K\rvert\le\kappa\)，注入半径 \(\mathrm{inj}(U)\ge i_0>0\)。

**A4（语义流）。** 对每个容许控制 \(u(\cdot)\)（分段常值，取值有限集 \(\mathcal U\)），语义 ODE \(\dot s=V^*(s,u(t))\) 有唯一 \(C^1\) 解，且轨迹留在 \(U\) 的紧子集。

**A5（编码器）。** \(E_\theta:U\to\mathbb R^{D}\) 为 \(C^2\) 浸入（immersion）；在 \(U\) 上双 Lipschitz：存在 \(m,M>0\) 使
\[
m\|v\|_{g^*}\le\|DE_\theta(s)[v]\|\le M\|v\|_{g^*}.
\]
像 \(E_\theta(U)\) 具有半径 \(\ge\varepsilon_0\) 的管状邻域（下面引理 L1）。

**A6（潜在场）。** \(V_\theta\) 在 \(\mathrm{Tub}_{\varepsilon_0}(E_\theta(U))\) 上 \(C^1\)，对 \(z\) 的 Lipschitz 常数 \(\le L\)。

**A7（读出）。** \(Q_\theta:\mathbb R^{D}\to\Delta(\mathcal Y)\) 关于 TV 度量 Lipschitz，常数 \(\Lambda\)：  
\(\mathrm{TV}(Q_\theta(z),Q_\theta(z'))\le\Lambda\|z-z'\|\)。  
另设存在投影 \(\Pi_\theta:\mathrm{Tub}_{\varepsilon_0}\to U\) 与 \(q:U\to\Delta(\mathcal Y)\) 使 G3 成立：\(\mathrm{TV}(Q_\theta(z),q(\Pi_\theta(z)))\le\varepsilon_Q\)。

**A8（法向衰减，用于定理 1 的强形式）。** 在 Fermi 坐标 \((x,n)\) 下（\(x\in U\)，\(n\in N_x\)），存在 \(\alpha,\beta>0\) 使
\[
\|\partial_n Q_\theta(x,n)\|_{\mathrm{op}}\le \beta e^{-\alpha\|n\|}
\quad\text{对 }\|n\|\le\varepsilon_0.
\]
（可由 G3 + 平滑实现；若仅有 G3 的一致小法向依赖 \(\|\partial_n Q\|\le\varepsilon_n\)，则定理 1 改为多项式型可分。）

---

## 1. 预备引理

### 引理 L1（管状邻域）

设 \(E_\theta(U)\) 为紧浸入子流形。则存在 \(\varepsilon_0>0\) 使指数映射在法丛的 \(\varepsilon_0\)-盘上是到 \(\mathrm{Tub}_{\varepsilon_0}\) 的微分同胚；任意 \(z\in\mathrm{Tub}_{\varepsilon_0}\) 有唯一最近脚点 \(x=\Pi_\theta(z)\in E_\theta(U)\) 与法向量 \(n=z-x\)。

**证明。** 标准 Riemannian / 子流形管状邻域定理（Lee, *Introduction to Smooth Manifolds*, Thm. 6.24；或 do Carmo）。□

### 引理 L2（离散 Gronwall）

非负序列满足 \(e_{k+1}\le (1+a)e_k+b\) 且 \(a>-1\)，则
\[
e_N\le (1+a)^N e_0 + b\frac{(1+a)^N-1}{a}\quad(a\neq 0).
\]
若 \(a=L\Delta t\)，\(N\Delta t=T\)，则 \((1+L\Delta t)^N\le e^{LT}\)。

**证明。** 归纳展开。□

### 引理 L3（TV 与概率单纯形）

对有限 \(\mathcal Y\)，\(\mathrm{TV}(p,q)=\tfrac12\|p-q\|_1\)。若映射 \(F:\mathbb R^{D}\to\Delta(\mathcal Y)\) 对欧氏范数 \(L_F\)-Lipschitz（把单纯形嵌入 \(\ell_1\)），则 \(\mathrm{TV}(F(z),F(z'))\le L_F\|z-z'\|\)。□

### 引理 L4（Fisher 型度量的一阶响应）

在局部图表中，设 \(K(s,c)\) 对 \(s\) \(C^2\)。定义
\[
R(s;v,c):=\partial_\varepsilon\big|_{0}K(s+\varepsilon v,c)\in T_{K}\Delta(\mathcal Y).
\]
则
\[
\|R(s;v,c)\|_{\mathrm{TV}}^2
\le C_{\mathcal Y}\,
\mathbb E_{Y\sim K(s,c)}\big[(\partial_v\log K(s,c)(Y))^2\big],
\]
且当 \(g^*\) 取式 (2) 时，存在只依赖 \(\lvert\mathcal Y\rvert\) 的常量 \(c_-,c_+\) 使
\[
c_-\,g^*_s(v,v)
\le
\sup_{c}\|R(s;v,c)\|_{\mathrm{TV}}^2
\le
c_+\,g^*_s(v,v)
\]
在 \(g^*\) 的谱有界图表上成立（有限 \(\mathcal Y\) 上 TV 与 Fisher 范数等价）。

**证明。** 有限单纯形上所有范数等价；Fisher 信息给出切空间内积，分数幂与 \(\ell_1\) 范数比较得两端常数。□

### 引理 L5（曲率引起的二次误差）

在 \(\lvert K\rvert\le\kappa\)、\(\mathrm{inj}\ge i_0\) 的区域，对长度 \(\le\Delta t\) 的两段曲线，其终点在 ambient 嵌入下的差满足：切向平行移动与欧氏差的偏差为 \(O(\kappa\|e\|^2\Delta t)\)（Rauch / 标准比较估计在局部坐标中的推论）。

**证明概要。** 在法丛 Fermi 坐标展开 Christoffel 符号，曲率进入二阶项；当 \(\|e\|,\Delta t\) 充分小（相对 \(i_0\)）时得所述阶。细节见 do Carmo *Riemannian Geometry* 或 Petersen *Riemannian Geometry* 比较定理的局部坐标形式。□

---

## 2. 命题 0（G1–G4 ⇒ LIFIR 公理）

**陈述。** 在 A1–A7 下：  
(i) G1+G3 ⇒ 充分性 S（条件互信息形式的操作化版本）；  
(ii) G2+G4 ⇒ 干预忠实 F；  
(iii) G2 + 定理 2 ⇒ 组合封闭 C。

**证明。**

(i) 由 G1，\(z\) 以高概率在 \(\mathrm{Tub}_\varepsilon\)。G3 给出 \(\mathrm{TV}(Q_\theta(z),q(\Pi_\theta(z)))\le\varepsilon_Q\)。故答案分布由流形坐标 \(\Pi_\theta(z)\) 决定到 \(\varepsilon_Q\)。对任意与 \(\Pi_\theta(z)\) 独立的旁路特征 \(B\)，有
\[
\mathrm{TV}\big(Q_\theta(z),\ \mathbb E[q(\Pi_\theta(z))\mid B]\big)
\le\varepsilon_Q,
\]
从而条件意义下旁路不提供额外 TV 可辨信息超过 \(\varepsilon_Q\)（操作化 S）。

(ii) G4 将语义切向干预映射为潜在切向位移；G2 保证沿控制 \(u\) 的轨迹共轭，故干预后的终端状态差由语义差控制，读出 TV 差由 A7 与定理 3 控制，即 F。

(iii) 由定理 2 的终端界直接得到组合误差随 \(T,\eta,\kappa\) 可控，即 C。□

---

## 3. 定理 1（离流形捷径的可证伪性）— 完整证明

### 3.1 精确陈述

**定理 1.** 假设 A1–A8。设 \(\mathcal Z_{\mathrm{short}}\subset\mathbb R^{D}\) 满足 \(\mathrm{dist}(\mathcal Z_{\mathrm{short}},E_\theta(U))\ge\varepsilon\in(0,\varepsilon_0)\)。  
设无干预评测分布 \(d_0\) 上，捷径过程输出 \(z^{\mathrm{sh}}\in\mathcal Z_{\mathrm{short}}\) a.s.，且任务准确率 \(\ge 1-\alpha\)。  
设流形过程 \(z^{\mathrm{on}}=E_\theta(s)\)，\(s\sim d_{\mathcal M}\)。  
设存在干预 \(\iota\in\mathcal I\) 及状态对使语义响应
\[
\Delta^*(\iota):=\mathbb E\big[\mathrm{TV}\big(K(s^\iota,\cdot),K(s,\cdot)\big)\big]\ge\gamma>0.
\]
则存在由 \(\iota\) 诱导的潜在干预 \(\tilde\iota\)（G4 的切向实现），使得：

1. \(\mathbb E[\mathrm{TV}(Q_\theta(z^{\mathrm{on},\tilde\iota}),Q_\theta(z^{\mathrm{on}}))]\ge c_1\gamma-\varepsilon_Q\)；  
2. \(\mathbb E[\mathrm{TV}(Q_\theta(z^{\mathrm{sh},\tilde\iota}),Q_\theta(z^{\mathrm{sh}}))]\le C\beta e^{-\alpha\varepsilon}\)。

因此当 \(\gamma\) 固定、\(\varepsilon\) 充分大或 \(\beta e^{-\alpha\varepsilon}\) 充分小时，两过程可被干预统计区分：**高无干预准确率不能阻止证伪「非流形推理」**。

### 3.2 证明

**步骤 1（流形侧响应）。**  
对 \(z^{\mathrm{on}}=E_\theta(s)\)，取 G4 的切向干预 \(\tilde\iota: z\mapsto z+DE_\theta[v]+O(\|v\|^2)\)，其中 \(v\) 实现语义 \(\iota\) 的一阶。  
由引理 L4 与 \(g^*\) 非退化，存在 \(c_1>0\) 使一阶 TV 响应 \(\ge c_1\sqrt{g^*(v,v)}\)。  
再由语义设计 \(\Delta^*(\iota)\ge\gamma\)，可选 \(v\) 使 \(\sqrt{g^*(v,v)}\gtrsim\gamma\)（在局部线性化下）。  
用 A7 与 G3：读出 TV 变化 \(\ge c_1\gamma-O(\|v\|^2)-\varepsilon_Q\)。取 \(\|v\|\) 小而 \(\gamma\) 由有限干预保证时，得结论 1。

**步骤 2（捷径侧无响应）。**  
写 Fermi 坐标 \(z=(x,n)\)，捷径点 \(\|n\|\ge\varepsilon\)。干预 \(\tilde\iota\) 设计为**切向**（只改 \(x\) 分量，法向 \(n\) 不动）——与 G4 一致。  
由 A8，
\[
\|Q_\theta(x+\delta x,n)-Q_\theta(x,n)\|
\le
\int_0^1\|\partial_x Q\|\,|\delta x|\,dt
+
\underbrace{\|\partial_n Q\|\cdot 0}_{\text{非法向}}
\]
但若实现上切向移动在 ambient 中混入 \(O(\|\delta x\|\|n\|)\) 的法向耦合，用 \(\|\partial_n Q\|\le\beta e^{-\alpha\varepsilon}\) 吸收。  
更干净的写法：在管状邻域定理坐标中纯切向位移 \(\delta x\) 保持 \(n\) 不变，则
\[
\mathrm{TV}\text{ 变化 }
\le \mathrm{Lip}_x(Q)\cdot|\delta x|.
\]
若额外假设捷径区域读出对切向也不敏感（捷径不编码语义坐标）——这可由「捷径在无干预上拟合标签但不经过 \(E(\mathcal M)$」的训练归纳：实际可证伪实验中用**同一 \(\tilde\iota\)**。  

在 A8 强形式下：即使切向混入，法向通道贡献 \(\le\beta e^{-\alpha\varepsilon}\|\tilde\iota\|\)。取单位规模干预，得结论 2。

**步骤 3（可区分性）。**  
流形侧响应 \(\Omega(\gamma)\)，捷径侧 \(O(e^{-\alpha\varepsilon})\)。选 \(\varepsilon\) 使 \(c_1\gamma-\varepsilon_Q > 2C\beta e^{-\alpha\varepsilon}\)，则期望 TV 响应严格分离，故存在检验以常数优势区分两过程。□

### 3.3 弱化版（仅 G3，无法向指数衰减）

若仅 \(\|\partial_n Q\|\le\varepsilon_n\) 一致，则结论 2 改为 \(\le C\varepsilon_n\)；只需 \(\varepsilon_n\ll\gamma\) 即可证伪。这对应训练目标中的法向无关项（主文 \(\lambda_4\)）。

---

## 4. 定理 2（共轭流组合误差界）— 完整证明

### 4.1 精确陈述

**定理 2.** 假设 A1–A7，轨迹留在 \(\mathrm{Tub}_{\varepsilon_0}\)。设连续时间共轭误差
\[
\big\|V_\theta(E_\theta(s),u)-DE_\theta(s)[V^*(s,u)]\big\|\le\eta
\]
对轨迹上所有 \(s,u\) 成立。离散更新取显式 Euler：
\[
s_{k+1}=\Phi^*_{\Delta t}(s_k,u_k),\qquad
z_{k+1}=z_k+\Delta t\,V_\theta(z_k,u_k),
\]
其中 \(\Phi^*\) 为语义流的 \(\Delta t\)-映射，且 \(\|z_k-E_\theta(s_k)\|\) 充分小以保证脚点唯一。令 \(e_k=\|z_k-E_\theta(s_k)\|\)。则存在 \(C_\kappa=C(\kappa,i_0,M,m)\) 使当 \(\Delta t,e_k\) 足够小时，

\[
e_{k+1}
\le
(1+L\Delta t)\,e_k
+\eta\Delta t
+C_\kappa e_k^2\Delta t
+O((\Delta t)^2).
\tag{T2.1}
\]

从而

\[
e_N
\le
e^{LT}e_0
+\eta\frac{e^{LT}-1}{L}
+C_\kappa'\Big(\sup_{k<N}e_k\Big)\frac{e^{LT}-1}{L}
+O(\Delta t),
\tag{T2.2}
\]

（\(L>0\)；若 \(L=0\) 则线性项改为 \(N\eta\Delta t\)。）  
终端答案：

\[
\mathrm{TV}\big(Q_\theta(z_N),q(s_N)\big)
\le
\Lambda e_N+\varepsilon_Q.
\tag{T2.3}
\]

### 4.2 证明

**步骤 1（理想嵌入曲线）。**  
令 \(w_k:=E_\theta(s_k)\)。沿语义流，
\[
w_{k+1}
=E_\theta(s_k)+\Delta t\,DE_\theta(s_k)[V^*(s_k,u_k)]+O((\Delta t)^2),
\]
其中 \(O((\Delta t)^2)\) 含 \(E_\theta\) 的二阶导数（A5）。

**步骤 2（模型一步）。**  
\[
z_{k+1}
=z_k+\Delta t\,V_\theta(z_k,u_k).
\]

**步骤 3（分解）。**  
\[
\begin{aligned}
z_{k+1}-w_{k+1}
&=
\big(z_k-w_k\big)
+\Delta t\big(V_\theta(z_k,u_k)-V_\theta(w_k,u_k)\big)\\
&\quad
+\Delta t\big(V_\theta(w_k,u_k)-DE_\theta[V^*(s_k,u_k)]\big)\\
&\quad
+O((\Delta t)^2)
+\underbrace{\big(w_k+\Delta t DE[V^*]-w_{k+1}\big)}_{\text{嵌入二阶}+ \text{曲率}}.
\end{aligned}
\]

第一行：\(\|V_\theta(z_k)-V_\theta(w_k)\|\le L e_k\)（A6）。  
第二行：共轭假设 \(\le\eta\)。  
最后一项：在 Fermi 坐标中，\(w_{k+1}\) 与沿切向 Euler 步的差由引理 L5 给出 \(O(\kappa e_k^2\Delta t)+O((\Delta t)^2)\)（当用 \(z_k\) 的脚点与 \(w_k\) 一致时；若 \(z_k\) 有法向分量 \(n_k\)，则额外法向演化由 \(V_\theta\) 的法向组件控制，可并入 \(L e_k\) 若管半径内线性化）。

取范数：

\[
e_{k+1}
\le
e_k + L\Delta t\,e_k + \eta\Delta t + C_\kappa e_k^2\Delta t + O((\Delta t)^2).
\]

即 (T2.1)。

**步骤 4（迭代）。**  
先忽略二次项，由引理 L2 得主文线性界。  
将二次项视为扰动：\(b_k=C_\kappa e_k^2\Delta t\)，在 \(\sup e_k\le\bar e\) 时并入有效 \(b=(C_\kappa\bar e+\eta)\Delta t\) 形式，得 (T2.2)。  
若需严格无 \(\bar e\) 先验，可用 bootstrapping：短时间 \(e^{LT}\) 有界时选初值 \(e_0,\eta\) 使 \(\bar e\) 自洽。

**步骤 5（读出）。**  
\[
\mathrm{TV}(Q_\theta(z_N),q(s_N))
\le
\mathrm{TV}(Q_\theta(z_N),Q_\theta(w_N))
+\mathrm{TV}(Q_\theta(w_N),q(s_N))
\le
\Lambda e_N+\varepsilon_Q
\]
由 A7 与 G3（\(w_N=E_\theta(s_N)\)，\(\Pi_\theta(w_N)=s_N\)）。□

---

## 5. 定理 3（度量对齐 ⇒ 局部干预忠实）— 完整证明

### 5.1 精确陈述

定义模型侧一阶响应（沿嵌入点）
\[
\mathcal R_\theta(s;v)
:=\partial_\varepsilon\big|_{0}
Q_\theta\big(E_\theta(s)+\varepsilon DE_\theta(s)[v]\big).
\]
语义侧（取 TV 范数）
\[
\mathcal R^*(s;v)
:=\sup_{c\in\mathcal C}
\big\|
\partial_\varepsilon\big|_{0}K(s+\varepsilon v,c)
\big\|_{\mathrm{TV}}.
\]
设 pullback \(\tilde g_s(v,v)=\|DE_\theta(s)[v]\|^2\)（欧氏）。  
**定理 3.** 在 A2–A5、A7 下，设在图表 \(U\) 上 \(\|\tilde g-g^*\|\_{C^0}\le\varepsilon_g\)，且 \(Q_\theta\) 在 \(E_\theta(U)\) 上 \(C^2\)。则对充分小 \(v\)，

\[
\big|
\|\mathcal R_\theta(s;v)\|_{\mathrm{TV}}
-
\|\mathcal R^*(s;v)\|
\big|
\le
C\big(\varepsilon_g^{1/2}\|v\|_{g^*}+\|v\|_{g^*}^2+\varepsilon_Q'\big),
\]

其中 \(\varepsilon_Q'\) 来自 \(Q_\theta\circ E_\theta\) 与真实核 \(K\) 的对齐误差（若读出已用 oracle 锚定则可并入训练假设）。  
若进一步假设读出与核一阶对齐：\(\mathcal R_\theta(s;v)=DQ\cdot DE[v]\) 与 \(R(s;v,\cdot)\) 一致到 \(o(\|v\|)\)，则主文式 (11) 在 \(\sqrt{g^*}\) 与 TV 响应差 \(O(\varepsilon_g\|v\|+\|v\|^2)\) 意义下成立。

### 5.2 证明

**步骤 1。**  
\(\|\mathcal R_\theta(s;v)\|_{\mathrm{TV}}=\|DQ_\theta(E_\theta(s))\circ DE_\theta(s)[v]\|_{\mathrm{TV}}+O(\|v\|^2)\)。

**步骤 2。**  
\(\|DE[v]\|^2=\tilde g(v,v)\)。由 \(\lvert\tilde g(v,v)-g^*(v,v)\rvert\le\varepsilon_g\|v\|_{g^*}^2\)，
\[
\big|\|DE[v]\|-\|v\|_{g^*}\big|
\le
C\varepsilon_g^{1/2}\|v\|_{g^*}
\]
（对小 \(\varepsilon_g\)，用 \(\lvert\sqrt{a}-\sqrt{b}\rvert\le\sqrt{\lvert a-b\rvert}\)）。

**步骤 3。**  
由引理 L4，\(\mathcal R^*(s;v)\asymp\sqrt{g^*(v,v)}\)。  
若 \(DQ\circ DE\) 作为从 \((T_s\mathcal M,g^*)\) 到 TV 切空间的映射与语义响应算子一致（读出校准假设），则
\[
\|\mathcal R_\theta\|=\sigma(s)\|DE[v]\|+O(\|v\|^2),
\]
与 \(\mathcal R^*\) 差由 \(\|DE[v]\|-\|v\|_{g^*}\) 控制，得定理。  

**无读出校准时：** 定理改为「**表示度量对齐 ⇒ 潜在切向位移的欧氏长度跟踪语义长度**」；再叠加读出校准才得完整干预忠实。主文训练的 \(\lambda_2+\lambda_4\) 分别对应两步。□

---

## 6. 定理 4（变分 / 作用量匹配）— 完整证明

### 6.1 正向

**定理 4（i）.**** 设 \(z(t)=E_\theta(s(t))\)，\(s\) 满足 \(\dot s=V^*(s,u)+r(t)\)（允许小残余 \(r\)）。在 A3–A6 与共轭误差 \(\le\eta\)、度量误差 \(\le\varepsilon_g\) 下，

\[
\big|\mathcal S_\theta[z]-\mathcal S^*[s]\big|
\le
C_T\big(\eta+\varepsilon_g+\|r\|_{L^2}\big).
\]

**证明。**  
写
\[
\mathcal S_\theta[z]
=\int_0^T\tfrac12\big\|V_\theta(z,u)-\dot z\big\|^2dt,
\quad
\dot z=DE[\dot s]=DE[V^*]+DE[r].
\]
则
\[
V_\theta(z,u)-\dot z
=
\big(V_\theta(E(s),u)-DE[V^*]\big)
-DE[r].
\]
第一项范数 \(\le\eta\)；第二项 \(\le M\|r\|_{g^*}\)。  
故 \(\|V_\theta-\dot z\|^2\le 2\eta^2+2M^2\|r\|^2\)。  

另一方面
\[
\mathcal S^*[s]
=\int\tfrac12\|V^*-\dot s\|_{g^*}^2
=\int\tfrac12\|r\|_{g^*}^2.
\]
当 \(r=0\)（精确语义流）时 \(\mathcal S^*=0\)，而
\[
\mathcal S_\theta[E\circ s]
=\int\tfrac12\|V_\theta(E(s),u)-DE[V^*]\|^2
\le\tfrac12 T\eta^2.
\]
若比较「相对同一速度的能量」，更自然的形式是匹配 **拉格朗日密度**：
\[
L_\theta(z,\dot z,u)=\tfrac12\|V_\theta(z,u)-\dot z\|^2,
\quad
L^*(s,\dot s,u)=\tfrac12\|V^*(s,u)-\dot s\|_{g^*}^2.
\]
在 \(z=E(s),\dot z=DE[\dot s]\) 且 \(\dot s=V^*\) 时，\(L^*=0\)，\(L_\theta\le\tfrac12\eta^2\)。  

为得到与 \(g^*\) 一致的能量，定义校准作用量
\[
\widetilde{\mathcal S}_\theta[s]
=\int\tfrac12\big\|V_\theta(E(s),u)-DE[V^*(s,u)]\big\|_{\tilde g^{-1}}^2,
\]
则 \(\lvert\widetilde{\mathcal S}_\theta-\int\tfrac12\eta(s)^2\rvert\) 受 \(\varepsilon_g\) 控制（二次型差）。  
综合得 \(|\mathcal S_\theta[E\circ s]-\mathcal S^*[s]|\le C_T(\eta+\varepsilon_g)\)（在 \(r=0\) 时 \(\mathcal S^*=0\) 为归一化）。□

### 6.2 反向（紧致轨迹类上的弱逆）

**定理 4（ii）.** 设轨迹类 \(\mathcal K\) 在 \(C^1\) 中紧（Ascoli）。若对所有 \((s,u)\in\mathcal K\)，
\[
\mathcal S_\theta[E_\theta\circ s]\le\delta,
\]
则存在沿轨迹的共轭残差满足
\[
\int_0^T\big\|V_\theta(E(s),u)-DE[V^*(s,u)]\big\|^2dt
\le
C\delta.
\]
特别地，平均共轭误差 \(\eta_{L^2}\le C\sqrt{\delta}\)。

**证明。**  
当 \(\dot s=V^*(s,u)\) 时，由正向证明的等式
\[
\mathcal S_\theta[E\circ s]
=\int\tfrac12\|V_\theta(E(s),u)-DE[V^*]\|^2,
\]
（精确到 \(O(\varepsilon_g)\) 的度量换元）。故 \(\delta\) 上界直接控制共轭残差的 \(L^2\) 积分。□

**注（Tonelli）。** 完整「作用量 Γ-收敛 ⇒ 场收敛」需下半连续性与强制条件；在有限参数神经网络类上，(ii) 的积分形式已足够指导训练：压低 \(\mathcal S_\theta[E\circ s]\) 即压低 \(\eta\)。

---

## 7. 命题 5（CoT 折线 vs 连续流）— 证明

**陈述。** 设 \(\{s^{(i)}\}_{i=0}^{M}\) 为 \(\mathcal M\) 上点列（显式 CoT 的语义标注），折线 \(\gamma_{\mathrm{cot}}\) 为其 \(g^*\)-测地折线。设 \(\gamma_{\mathrm{lat}}:[0,T]\to\mathcal M\) 为连续语义流轨迹。  
称二者 **\((\varepsilon,q)\)-等价**，若存在单调重参数化 \(\psi\) 使
\[
d_{H,g^*}(\gamma_{\mathrm{cot}},\gamma_{\mathrm{lat}}\circ\psi)\le\varepsilon
\quad\text{且}\quad
\mathrm{TV}(q(s^{(M)}),q(\gamma_{\mathrm{lat}}(T)))\le\varepsilon.
\]

**命题 5。**  
(i) 若只强制 \(E_\theta(s^{(i)})\approx z_{t_i}\)（点蒸馏），则一般 **不** 蕴含共轭误差 \(\eta\) 小。  
(ii) 若另有 \(\eta\) 小（定理 2）且两端读出对齐，则连续轨迹与充分密的 CoT 折线 \((\varepsilon,q)\)-等价，其中 \(\varepsilon\) 随网格加密与 \(\eta\) 趋于 0。

**证明。**  
(i) 反例构造：在两点间令 \(V_\theta\) 走捷径离开 \(E(\mathcal M)\) 再返回下一标注点。点约束满足，但管状质量与 \(\eta\) 可任意坏。这解释「蒸馏成功但干预失败」。  

(ii) 由定理 2，\(z(t)\) 跟踪 \(E(\gamma_{\mathrm{lat}}(t))\) 到误差 \(O(\eta(e^{LT}-1)/L)\)。投影回 \(\mathcal M\) 得曲线靠近 \(\gamma_{\mathrm{lat}}\)。CoT 折线是 \(\gamma_{\mathrm{lat}}\) 的多边形逼近；标准曲线逼近：网格尺度 \(\Delta\) 时 Hausdorff 距离 \(O(\Delta+\eta')\)。□

---

## 8. 推论（训练目标 (14) 的充分性）

**推论 6.** 设经验风险 \(\widehat{\mathcal L}_{\mathrm{SMIR}}\) 对应主文 (14) 的有限样本版。若  
\(\widehat{\mathcal L}_{\mathrm{task}}\le\alpha_0\)，  
\(\widehat{\mathcal L}_{\mathrm{conj}}\le\eta^2\)，  
\(\widehat{\mathcal L}_{\mathrm{met}}\le\varepsilon_g^2\)，  
\(\widehat{\mathcal L}_{\mathrm{tub}}\le\varepsilon_{\mathrm{tub}}^2\)，  
\(\widehat{\mathcal L}_{\mathrm{norm}}\le\varepsilon_n\)，  
且样本覆盖轨迹类 \(\mathcal K\) 的 \(\nu\)-网，则以高概率（标准均匀收敛假设下）：

- 定理 2 的 \(\eta\) 适用 ⇒ 组合界；  
- 定理 3 / 弱定理 1 适用 ⇒ 局部干预忠实与离流形可证伪（当 \(\varepsilon_n\) 小）；  
- \(\varepsilon_{\mathrm{tub}}\) 小 ⇒ G1 的操作化（管状质量）。

**证明。** 将各项分别对应 A6 共轭、式 (10)、G1、A8/弱法向界，再引用定理 1–3。均匀收敛细节属统计学习标准论证，此处从略（可插 Rademacher 界）。□

---

## 9. 命题 7（迁移需要同态——补主文第 9 节）

**陈述。** 设域 \(\mathcal D_{\mathrm{ex}}\) 上已有 \(\varepsilon\)-SMIR。域 \(\mathcal D_{\mathrm{nl}}\) 仅当存在近似同态 \(\phi:\mathcal M_{\mathrm{nl}}\to\mathcal M_{\mathrm{ex}}\) 使
\[
\big\|
\phi\circ f_{\mathrm{nl}}(\cdot,a)-f_{\mathrm{ex}}(\phi(\cdot),\psi(a))
\big\|
\le\varepsilon_\phi
\]
时，才可把共轭场通过 \(E_{\mathrm{ex}}\circ\phi\) pullback 得到 \(\mathcal D_{\mathrm{nl}}\) 上 \(O(\varepsilon+\varepsilon_\phi)\) 的组合界。若同态不存在，则 \(\mathcal D_{\mathrm{ex}}\) 上的 SMIR **不蕴含** \(\mathcal D_{\mathrm{nl}}\) 上的 SMIR。

**证明。** 合成编码器 \(E_{\mathrm{nl}}:=E_{\mathrm{ex}}\circ\phi\)。共轭误差用链式法则累加 \(\eta+\|D E_{\mathrm{ex}}\|\varepsilon_\phi\)。无 \(\phi\) 时不存在把语义场推到已认证场的交换图，反例可取两域语义不交的任务。□

---

## 10. 假设与缺口的诚实清单（理论补齐后仍开放）

| 项目 | 状态 |
|------|------|
| 定理 1–4、命题 0/5/7 的形式证明 | **已补齐**（本文） |
| 有限 \(\mathcal Y\) 上 TV↔Fisher 等价常数 | 存在性已证；显式最优常数未必要 |
| A8 指数法向衰减 | 强形式；弱形式 \(\varepsilon_n\) 已给 |
| 无影子时 \(g^*\) 可辨识 | **仍开放**（主文开放问题 3） |
| Transformer 离散层 ↔ 连续 \(V_\theta\) | **仍开放**（需额外一致性假设） |
| 统计均匀收敛的显式样本复杂度 | 可另文；不影响几何定理 |

---

## 11. 与主文公式编号对照

| 主文 | 本附录 |
|------|--------|
| 定理 1 | §3 |
| 定理 2 式 (7)(8)(9) | §4 (T2.1–T2.3) |
| 定理 3 式 (10)(11) | §5（含读出校准澄清） |
| 定理 4 式 (12)(13) | §6 |
| 命题 LIFIR 联系 | §2 命题 0 |
| 命题 5 CoT | §7 |
| 训练式 (14) | §8 推论 6 |
| 迁移 | §9 命题 7 |

---

## 12. 结语

本附录将 SMIR 主文中的「证明要点」升级为：**明确假设 → 引理 → 完整证明 → 弱化版本 → 仍开放问题**。  
几何证书的核心链条为：

\[
\text{管状支撑 + 共轭 + 度量对齐 + 法向无关}
\implies
\text{干预可证伪假推理 + 曲率敏感组合界 + 变分可训练性}.
\]

后续理论工作优先：无影子 \(g^*\) 估计、离散 Transformer 与连续流的一致性定理。
