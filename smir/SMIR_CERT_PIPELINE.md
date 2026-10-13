# SMIR-Cert：剩余目标完整流水线

针对顶会缺口：**F / 读出绑定 / 管状 / 离流形证伪 / \(g^*\) 代理 / OOD**。  
**不重复** 7B ContCE、Outcome、纯 SCF、失败的文本 Phase-S。

## 关键修复

先前 F=0 有两层原因：

1. 7B 末 token 对近乎相同寄存器文本 **E 坍缩**  
2. 标签 \(y=y^*(s,c)\) **依赖 continuation/query**，却用无 query 的 `Q(E(s))` 测 F → **协议不适定**

本流水线：

- **结构化图表** \(E:\mathbb R^5\to\mathbb R^d\)（影子域合法坐标）  
- **条件读出** \(Q(z,q)\)  
- Stage A 先解锁 F，失败则中止（不空转 Stage B）  
- Stage B：\(\mathcal L_{\mathrm{task}}+\lambda_f\mathcal L_{\mathrm{flow}}+\lambda_e\|z_T-\mathrm{sg}(E(s^*))\|^2\) + 轻量保真  

## 阶段

| 阶段 | 目标 |
|------|------|
| 数据 | `SMIR-Shadow-Struct`（含 `continuation` / `state_final`） |
| A | 干预忠实 warmup |
| B | 共轭 + 终点管状 |
| C | 认证门控：F、on−off、η、tub_rel、S-gap、adv、g\*proxy、OOD、长度 |

## 提交

一键（无数据则自动构建）：

```bash
cd /project/peilab/why/NEW
mkdir -p smir/artifacts/slurm
sbatch smir/slurm/27_smir_cert_pipeline.sbatch
```

分步：

```bash
J1=$(sbatch --parsable smir/slurm/26_build_shadow_struct.sbatch)
sbatch --dependency=afterok:$J1 smir/slurm/27_smir_cert_pipeline.sbatch
```

冒烟：

```bash
STEPS_A=500 STEPS_B=800 EVAL_N=100 sbatch smir/slurm/27_smir_cert_pipeline.sbatch
```

本地快速（MLP，可 CPU）：

```bash
cd /project/peilab/why/NEW/smir
python data/build_shadow.py --out-dir artifacts/data/SMIR-Shadow-Struct \
  --n-train 4000 --n-val 400 --n-test-id 400 --n-test-ood 400 --seed 0
cd train
python run_smir_cert.py --data-dir ../artifacts/data/SMIR-Shadow-Struct \
  --out-dir ../artifacts/runs/smir_cert --steps-a 1500 --steps-b 3000 --seed 0
```

## 看结果

```bash
squeue -u $USER
tail -f smir/artifacts/slurm/smir-cert-<JOBID>.out
cat smir/artifacts/runs/smir_cert/overall.json
```

成功：`gates.pass_cert == true`（含 `pass_faithfulness` + `pass_geometry` + S/adv/g\*/OOD）。

## 与理论的对应

| 剩余目标 | 门控 |
|----------|------|
| G4 / 定理 3 干预忠实 | `F_ge_0_40`, `follow_ge_0_30` |
| G3 读出 | `on_minus_off`, `S_gap` |
| 定理 2 共轭+组合 | `eta_lt_0_05`, `tub_rel_lt_1` |
| 定理 1 离流形 | `adv_lt_0_55` |
| \(g^*\) 代理 | `gstar_proxy_gt_0` |
| 外推 | `ood_ge_0_75`, `length_T4` |
