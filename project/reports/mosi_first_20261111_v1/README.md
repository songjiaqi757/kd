# MOSI First 2026-11-11：几何矩阵、`diagonal_mse` 与逐 checkpoint 复现包

> 本目录是 `mosi_first_20261111_v1` 的冻结复现包，补齐六类信息：
> (1) 几何矩阵精确构造公式；(2) `diagonal_mse` 完整损失实现；
> (3) matched-spectrum 矩阵生成与校验代码；(4) 18 个方法的完整训练配置；
> (5) 54 个 checkpoint 的 MAE / Acc-2 / Pearson / F1；
> (6) 数据指纹、源码指纹与 checkpoint SHA256。
>
> 结论仍然不变：本 campaign 是**单 seed（13）探索性实验**，测试集只报告、不参与选择；
> `diagonal_mse` 在 `valid_mae` 与 `valid_acc2_nonzero` 两种选择下都是本组最低 MAE，
> 但没有超过既有 `E3_ea_kd_full`（0.668253）或 `F7`（0.6812）。
> `normalized_icd` 的 `last` checkpoint 为 0.685065，但 `last` 不是验证选择，不能当作主结果。

## 0. 目录与文件

| 路径 | 内容 |
|---|---|
| `source/mechanisms.py` | 冻结的 `transform()` / `terms()` / `compute_loss()` 原始实现 |
| `source/train_mosi.py` | 完整训练循环、优化器、LoRA、早停与三种 checkpoint 保存逻辑 |
| `source/prepare.py` | 数据/教师 target/归一化资产准备 |
| `source/evaluate_mosi.py` | 单 checkpoint 测试与指标计算 |
| `source/summarize.py` | 从 `tests/*/report.json` 汇总 |
| `source/build_tables.py` | 从冻结 campaign 输出重建本目录三个 CSV |
| `source/verify_geometry.py` | 重建并校验全部几何矩阵、奇异值、条件数与 SHA256 |
| `source/plan.json` | 冻结的 18 个 job 配置 |
| `assets/geometry.json` | Möbius 矩阵、奇异值、条件数、Gram 对角 |
| `assets/matched_matrices.json` | 三个 matched-spectrum 矩阵及校验值 |
| `assets/normalization.json` | `normalized_icd` / `normalized_subset` 使用的归一化统计量 |
| `assets/teacher_manifest.json` | 教师资产清单 |
| `tables/method_configs.csv` | 18 个方法的紧凑训练配置（27 列） |
| `tables/full_run_configs.json` | 18 个方法的完整 `run_config.json`（78 字段/方法，含路径、教师资产、冻结指纹等） |
| `tables/per_checkpoint_metrics.csv` | 54 行 = 18 方法 × 3 策略，含完整 SHA256 与验证/测试指标 |
| `tables/transform_matrices.csv` | 18 行，矩阵 JSON、SHA256、奇异值、条件数 |
| `results/summary.md` / `results/summary.json` | 冻结汇总结果 |
| `fingerprints.json` | 数据、教师资产、源码、变换、54 个 checkpoint 的 SHA256 清单 |

---

## 1. 数据、教师与评测协议

### 1.1 数据划分

| split | 原始话语数 | 窗口数 | 说明 |
|---|---:|---:|---|
| train | 1,284 | 1,286 | 官方 MOSI train；窗口化后 2 条话语产生额外窗口 |
| valid | 229 | 229 | 官方 MOSI valid |
| test | 686 | 688 | 官方 MOSI test；688 个窗口，指标按 686 条话语聚合 |

- 数据集：CMU-MOSI。
- 输入：文本 + 音频 + 视频；学生模型为 0.82B TAV 模型。
- 学生骨干：Qwen3-0.6B-Base + WavLM-Base-Plus + VideoMAE-Base。
- 教师：Qwen3-Omni-30B-A3B-Instruct，七个 subset probe：`T/A/V/TA/TV/AV/TAV`，hidden 2048 维，probe seeds `2026/2027/2028`。
- 训练/验证/测试严格按官方 split；测试集只报告，不参与早停或 checkpoint 选择。

### 1.2 训练预算与 checkpoint 策略

| 项目 | 值 |
|---|---|
| epoch | 30 |
| min_epochs | 1 |
| patience | 7 |
| batch size | 8 |
| learning rate | 1e-4 |
| weight decay | 0.01 |
| lr schedule | constant |
| seed | 13 |
| 视频 LoRA | layers=12, rank=8, alpha=16, dropout=0.05 |
| 总参数 / 可训练参数 | 819,309,176 / 42,650,120 |

每个方法保存三个 checkpoint：

| 策略 | 文件 | 选择依据 | 用途 |
|---|---|---|---|
| `valid_mae` | `best_valid_mae.pt` | 验证集 MAE 最低 | 主报告 |
| `valid_acc2_nonzero` | `best_valid_acc2_nonzero.pt` | 验证集非零 Acc-2 最高 | 主报告 |
| `last` | `last.pt` | 最后一个 epoch | 仅作敏感性参考，**不是验证选择** |

### 1.3 数据指纹

完整 SHA256 与行数见 `fingerprints.json`。关键值：

| 资产 | 行/大小 | SHA256 | 说明 |
|---|---:|---|---|
| `dataset/cmu_mosi/manifests/train.jsonl` | 1,284 | `73e84cabfdd53970ddda182f6da7e0730198b6d602cde2a2b771e934f86e9fda` | 原始 train manifest |
| `dataset/cmu_mosi/manifests/valid.jsonl` | 229 | `a4ad595c7de78a607fbd0fbb8759dc428c96b739bddf8f9f7eb2b51a90f40cb7` | 原始 valid manifest |
| `dataset/cmu_mosi/manifests/test.jsonl` | 686 | `109df07fea788b1c14d2fc8d555f187b134f611249cd06320777628995dce51c` | 原始 test manifest |
| `dataset/cmu_mosi/manifests/official_train_valid_windowed.jsonl` | 1,515 | `ff94c37bcab1729a2e948c3f161c90fd92168229e203144f70de00e0440d7d3e` | 归一化路径后的 train+valid 窗口 |
| `dataset/cmu_mosi/manifests/official_test_windowed.jsonl` | 688 | `912c7ee2050ad9ec3b0163a4efa3657450555e062a9d13a28242a3c70781b127` | 归一化路径后的 test 窗口 |
| campaign `assets/official_test_windowed.jsonl` | 688 | raw `12b415ba4f4a8a8771503d9b4450178596356084cd3de8dc546520f9d77fb3c0` | 含 `/ai/sjq/kd` 绝对路径 |
| campaign train+valid 窗口 | 1,515 | raw `ba93c84cd206204744f43ba67a2be9ad9936572453c02eb9c9f5c183b261b498` | 含 `/ai/sjq/kd` 绝对路径 |
| split 过滤后 train 行拼接 | 1,286 | `9c07c00102530b9aae5a5fdc82f492c59d7211270f73945c487e1bbe21515c05` | 按 `split=train` 过滤 |
| split 过滤后 valid 行拼接 | 229 | `3026d56581b5da7ef56ecbd1c52eb4cc15131984c1e964fc7c9f884b3ce106ab` | 按 `split=valid` 过滤 |
| split 过滤后 test 行拼接 | 688 | `12b415ba4f4a8a8771503d9b4450178596356084cd3de8dc546520f9d77fb3c0` | 与 test 窗口 raw hash 相同 |
| 教师 ensemble `teacher_ensemble_n2.jsonl` | 1,513 | `2ded7373b55fec3db4fd1851e14b5d40badb8661f9ff6d35d4f34e7b56602510` | 七 probe 教师输出 |
| `interaction_targets.jsonl` | 1,513 | `f49fa257f21cd2bfc14dd883113b38c2c5ce778b42dff290eb16f60da2892231` | 7 坐标 target |
| `teacher_features/features.npy` | 42 MB | `13521c945179a8843441560fdfa6620736b3b10f427ac29f324c4741e2f985fb` | 教师特征缓存 |
| `teacher_feature_index.jsonl` | 77 KB | `dc8eec5431de858d70c0dabb9dee9ca9bbb66cb480858cccff03709c06917db8` | 特征索引 |
| `assets/normalization.json` | 684 B | `999268a2e3b3bcbca5719ea2470a582973028ac725cb257e264ff409c766465b` | 归一化统计量 |

> 路径归一化说明：历史迁移把 `/data/sjq/kd` 改成了 `/ai/sjq/kd`。`fingerprints.json` 同时记录 raw 文件 hash 和“把 `/ai/sjq/kd` 替换回 `/data/sjq/kd` 后”的 normalized hash；后者与历史 manifest hash 对齐。

---

## 2. 几何矩阵的精确构造

### 2.1 子集顺序与定义

七个交互坐标按固定顺序排列：

```text
T, A, V, TA, TV, AV, TAV
```

记 `f_S` 为教师/学生在子集 `S` 上的回归输出，`f_∅ = 0`。对任意非空子集 `S ⊆ {T,A,V}`，Anchored Möbius 交互坐标为

$$
\delta_S \;=\; \sum_{R \subseteq S} (-1)^{|S|-|R|}\, f_R,
\qquad \delta_\varnothing = 0.
$$

七个 `δ_S` 按上述顺序组成向量 `δ = [δ_T, δ_A, δ_V, δ_TA, δ_TV, δ_AV, δ_TAV]^T`。
学生与教师之间的交互误差为

$$
e \;=\; A\,(f_s - f_t) \;=\; A\,\Delta,
$$

其中 `A` 是下面的 7×7 Anchored Möbius 矩阵，行/列顺序均为 `T,A,V,TA,TV,AV,TAV`。

### 2.2 矩阵 `A`

```text
A =
[[ 1,  0,  0,  0,  0,  0,  0],   # T
 [ 0,  1,  0,  0,  0,  0,  0],   # A
 [ 0,  0,  1,  0,  0,  0,  0],   # V
 [-1, -1,  0,  1,  0,  0,  0],   # TA
 [-1,  0, -1,  0,  1,  0,  0],   # TV
 [ 0, -1, -1,  0,  0,  1,  0],   # AV
 [ 1,  1,  1, -1, -1, -1,  1]]   # TAV
```

逐行展开：

```text
δ_T   =  f_T
δ_A   =  f_A
δ_V   =  f_V
δ_TA  = -f_T - f_A + f_TA
δ_TV  = -f_T - f_V + f_TV
δ_AV  = -f_A - f_V + f_AV
δ_TAV =  f_T + f_A + f_V - f_TA - f_TV - f_AV + f_TAV
```

### 2.3 精确性质

| 性质 | 值 |
|---|---|
| 秩 | 7（可逆） |
| 奇异值 | `[3.451967523471161, 1.618033988749895, 1.6180339887498942, 1.0, 0.6180339887498948, 0.6180339887498947, 0.289689863302781]` |
| 条件数 | `11.916079783099619` |
| `A^T A` 对角 | `[4, 4, 4, 2, 2, 2, 1]` |
| `max(abs(A^{-1}A - I))` | `1.2212453270876722e-15` |
| `A` 的 SHA256（float64 小端字节） | `6d53177914ddc4e7de3a777e523df8059c1a2c1a66a3f70a0faa60cca0fe3ae5` |

### 2.4 生成代码

以下与 `source/mechanisms.py::transform()`、`project/src/rdid_mosei/interaction.py::mobius_matrix()` 完全一致：

```python
NAMES = ("t", "a", "v", "ta", "tv", "av", "tav")
SUBSET_SETS = tuple(frozenset(name) for name in NAMES)

def mobius_transform(values, empty_baseline):
    if values.shape[-1] != 7:
        raise ValueError(f"expected seven subset values, got {values.shape}")
    baseline = torch.as_tensor(empty_baseline, dtype=values.dtype, device=values.device)
    terms = []
    for target in SUBSET_SETS:
        result = torch.zeros_like(values[..., 0])
        for index, source in enumerate(SUBSET_SETS):
            if source.issubset(target):
                result = result + ((-1) ** (len(target) - len(source))) * values[..., index]
        result = result + ((-1) ** len(target)) * baseline
        terms.append(result)
    return torch.stack(terms, dim=-1)

def mobius_matrix(dtype=torch.float64):
    return mobius_transform(torch.eye(7, dtype=dtype), 0.0).T
```

---

## 3. 各机制对应的变换矩阵

`mechanisms.py::transform(args, dtype, device)` 的分支：

| mechanism | 变换 `M` | 说明 | SHA256 |
|---|---|---|---|
| `none` / `subset` | `I_7` | 直接对七个 subset 输出做误差 | `75f0bfc29ab9286f1d56c6eeca5c1d5abbeff1df15ed44032aacf7829ba28b72` |
| `icd` | `A` | Anchored Möbius 交互坐标 | `6d53177914ddc4e7de3a777e523df8059c1a2c1a66a3f70a0faa60cca0fe3ae5` |
| `diagonal` | `D = diag(sqrt(diag(A^T A)))` | 逐坐标按列范数缩放，`MSE` | `2e65afe9f22c8f8ea74ec0d680d07fc40be8d0e1c53427484df9e5892881451f` |
| `weighted` | `D_w = diag(sqrt(diag(A^T A) / mean(diag(A^T A))))` | 先归一化再按列范数缩放 | `e0b1adac138aae82a154975c0a7b85df10fdce77c944dab3755442c8a751b7d9` |
| `orthogonal` | `Q` | `random_orthogonal_matrix(seed)` | `9c6729d84d235866e81a5c123b09cbc7b9bcad7762e9eb2a207690e9d58d69c1` |
| `normalized_icd` | `diag(1/max(std, ε)) @ A` | 用坐标标准差归一化后再 Möbius | `e7f8ef5dff88a82384ee3b32d771a51ed7906cdf1f449948babe7a08ec580ae6` |
| `normalized_subset` | `diag(1/max(std, ε))` | 只做 subset 标准化 | `76aa994101aff9fe70e06a49c2f56706ea93eeff905df131dccbdab3ff2c5fe2` |
| `matched` seed 20261111 | `Q1 @ diag(σ) @ Q2^T` | 与 `A` 同奇异值 | `2e4445a6201f9b4fdec6e62bf805f59caa5bcb7b6012e05dc27224c254a6cb0c` |
| `matched` seed 20261112 | 同上 | 同上 | `117a5c013e061cd9e54de3c5538d9dda6a06abcec2a0ae55cd511eb57a6d6952` |
| `matched` seed 20261113 | 同上 | 同上 | `5c809d096710aed706ce0aaea1f60a9b36564233d98556f56632c9810e015964` |

其中：

```text
D     = diag([2, 2, 2, √2, √2, √2, 1])
D_w   = D / sqrt(19/7)
σ     = svdvals(A)
```

`normalized_icd` 的 `std` 来自 `assets/normalization.json` 的 `coordinate_std`；`normalized_subset` 使用 `subset_std`；`epsilon` 用于防止除零。`transform_matrices.csv` 保存了每个方法的完整矩阵 JSON、奇异值和条件数。

---

## 4. `diagonal_mse` 的完整损失实现

### 4.1 数学形式

`diagonal_mse` 的变换矩阵为

$$
D = \operatorname{diag}\!\left(\sqrt{\operatorname{diag}(A^\top A)}\right)
  = \operatorname{diag}(2,2,2,\sqrt2,\sqrt2,\sqrt2,1).
$$

误差与辅助损失为

$$
e_n = D\,(f_{s,n} - f_{t,n}), \qquad
\ell_{n,k} = \frac12 e_{n,k}^2,
$$

$$
L_{\text{aux}} = \frac{1}{7}\sum_{k=1}^{7}
\frac{\sum_{n=1}^{N} w_n\,\ell_{n,k}}{\sum_{n=1}^{N} w_n},
$$

其中 `w_n` 是该窗口的聚合权重（`weights`），`N` 是 batch 内窗口数。总损失为

$$
L = L_{\text{reg}} + c_{\text{ce}}L_{\text{ce}} + L_{\text{kd}}
  + c_{\text{kl}}L_{\text{kl}}
  + \lambda_{\text{interaction}} L_{\text{aux}}.
$$

本 campaign 中 `diagonal_mse` 的配置为：

```text
mechanism=diagonal, loss_kind=mse, task_ce=0.0,
lambda_interaction=0.3, kd_kl=1.0, lambda_feature=2.0, lambda_subset=1.0
```

### 4.2 冻结代码

以下代码逐字来自 `source/mechanisms.py`。`transform()` 的 `normalized_*` 分支读取同目录 `assets/normalization.json`；`terms()` 同时计算回归/CE/KD/KL 与交互辅助项。

```python
def transform(args, dtype=torch.float64, device='cpu'):
    m = mobius_matrix(dtype=torch.float64)
    name = args.mechanism
    if name == 'icd':
        matrix = m
    elif name == 'matched':
        sigma = torch.linalg.svdvals(m)
        q1 = random_orthogonal_matrix(seed=args.matrix_seed)
        q2 = random_orthogonal_matrix(seed=args.matrix_seed + 1000003)
        matrix = q1 @ torch.diag(sigma) @ q2.T
    elif name == 'orthogonal':
        matrix = random_orthogonal_matrix(seed=args.matrix_seed)
    elif name in ('diagonal', 'weighted'):
        weights = (m.T @ m).diag()
        if name == 'weighted':
            weights = weights / weights.mean()
        matrix = torch.diag(weights.sqrt())
    elif name in ('normalized_icd', 'normalized_subset'):
        stats = json.loads((BASE / 'assets/normalization.json').read_text())
        key = 'coordinate_std' if name == 'normalized_icd' else 'subset_std'
        std = torch.tensor(stats[key], dtype=torch.float64).clamp_min(stats['epsilon'])
        matrix = torch.diag(1 / std) @ (m if name == 'normalized_icd' else torch.eye(7, dtype=torch.float64))
    else:
        matrix = torch.eye(7, dtype=torch.float64)
    return matrix.to(device=device, dtype=dtype)

def terms(outputs, batch, device, args, assets):
    output = outputs['tav'] if 'tav' in outputs else outputs
    weights = batch['weights'].to(device)
    reg = weighted_mean(F.smooth_l1_loss(output['regression'].float(), batch['sentiment'].to(device), reduction='none'), weights)
    ce = weighted_mean(F.cross_entropy(output['classification_logits'].float(), batch['classes'].to(device), reduction='none'), weights)
    teacher = batch['teacher_subset_scores'].to(device)
    kd = weighted_mean(F.smooth_l1_loss(output['regression'].float(), teacher[:, -1], reduction='none'), weights)
    kl = weighted_mean(probability_kd_per_sample(output['classification_logits'], batch['teacher_ensemble_probabilities'].to(device), assets['kd_temperature']), weights)
    if 'tav' in outputs:
        student = torch.stack([outputs[s]['regression'].float() for s in NAMES], -1)
        error = (student - teacher.float()) @ transform(args, torch.float32, device).T
        losses = 0.5 * error.square() if args.loss_kind == 'mse' else F.smooth_l1_loss(error, torch.zeros_like(error), reduction='none')
        coordinates = (losses * weights[:, None]).sum(0) / weights.sum().clamp_min(1e-8)
    else:
        error = reg.new_zeros((len(weights), 7))
        coordinates = reg.new_zeros(7)
    return reg, ce, kd, kl, coordinates, error

def compute_loss(model, outputs, batch, device, args, assets, epoch):
    reg, ce, kd, kl, coords, error = terms(outputs, batch, device, args, assets)
    base = reg + args.task_ce * ce + kd + args.kd_kl * kl
    loss = base + args.lambda_interaction * coords.mean()
    step = getattr(args, '_diagnostic_step', 0) + 1
    args._diagnostic_step = step
    if step == 1 or step % args.diagnostic_every == 0:
        row = dict(epoch=epoch, step=step, base_loss=float(base.detach()), auxiliary=float(coords.mean().detach()),
                   total=float(loss.detach()), coordinate_losses=coords.detach().cpu().tolist(),
                   transformed_error_abs_quantiles=torch.quantile(error.detach().abs().float(), torch.tensor([.25,.5,.75,.95], device=device)).cpu().tolist(),
                   quadratic_region_fraction=(error.detach().abs() < 1).float().mean(0).cpu().tolist())
        with (args.output / 'loss_diagnostics.jsonl').open('a') as h:
            h.write(json.dumps(row) + '\n')
    return loss
```

对 `diagonal_mse` 而言，关键是：

```python
name = 'diagonal'
column_norms = (m.T @ m).diag()            # [4, 4, 4, 2, 2, 2, 1]
matrix = torch.diag(column_norms.sqrt())   # diag([2, 2, 2, sqrt(2), sqrt(2), sqrt(2), 1])
error = (student - teacher.float()) @ matrix.T
losses = 0.5 * error.square()
batch_weights = batch['weights'].to(device)
coordinates = (losses * batch_weights[:, None]).sum(0) / batch_weights.sum().clamp_min(1e-8)
loss = base + args.lambda_interaction * coordinates.mean()
```

因此 `diagonal_mse` **不是**对 `δ` 做普通 MSE，而是先按 `D` 缩放七个坐标，再在每个坐标内做加权 MSE，最后对七个坐标取等权平均。`D` 的对角来自 `A` 的列范数，用来抵消 Möbius 坐标量纲差异。

---

## 5. matched-spectrum 矩阵生成与校验

### 5.1 生成算法

目标：构造一个与 `A` **奇异值完全相同**、但特征方向不同的 7×7 矩阵，作为“同谱随机对照”。

```text
σ = svdvals(A)                                  # 7 个奇异值
Q1 = random_orthogonal_matrix(seed)              # QR + 符号修正
Q2 = random_orthogonal_matrix(seed + 1000003)
M  = Q1 @ diag(σ) @ Q2.T
```

`random_orthogonal_matrix` 的精确实现：

```python
def random_orthogonal_matrix(size=7, seed=2026, dtype=torch.float64):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(size, size, generator=generator, dtype=dtype)
    q, r = torch.linalg.qr(matrix)
    signs = torch.where(torch.diag(r) >= 0, 1.0, -1.0).to(dtype=dtype)
    return q * signs.unsqueeze(0)
```

因为 `Q1`、`Q2` 正交，`M` 的奇异值严格等于 `σ`；但 `M` 的 Gram 矩阵一般不等于 `A^T A`，所以它只控制“谱”，不控制“坐标几何”。

### 5.2 三个 matched 矩阵

| 方法 | matrix_seed | 奇异值 | 条件数 | SHA256 |
|---|---:|---|---:|---|
| `matched0_huber` / `matched0_mse` | 20261111 | 与 `A` 相同 | 11.916079783099619 | `2e4445a6201f9b4fdec6e62bf805f59caa5bcb7b6012e05dc27224c254a6cb0c` |
| `matched1_huber` | 20261112 | 与 `A` 相同 | 11.916079783099619 | `117a5c013e061cd9e54de3c5538d9dda6a06abcec2a0ae55cd511eb57a6d6952` |
| `matched2_huber` | 20261113 | 与 `A` 相同 | 11.916079783099619 | `5c809d096710aed706ce0aaea1f60a9b36564233d98556f56632c9810e015964` |

完整矩阵保存在 `assets/matched_matrices.json`；每个矩阵的 `singular_values`、`condition_number`、`sha256` 都在同一文件里。

### 5.3 校验代码

`source/verify_geometry.py` 会重建 `A`、三个 matched 矩阵、`diagonal` / `weighted` / `normalized_*` / `orthogonal`，并断言：

```python
torch.testing.assert_close(torch.linalg.svdvals(m),
                           torch.tensor(expected_singular_values, dtype=torch.float64),
                           rtol=0.0, atol=1e-10)
assert abs(float(torch.linalg.cond(m)) - expected_condition) < 1e-9
assert hashlib.sha256(m.numpy().tobytes()).hexdigest() == expected_sha256
```

实际运行（需要 torch 环境）：

```bash
cd project/reports/mosi_first_20261111_v1
/ai/sjq/miniconda3/bin/python source/verify_geometry.py
```

本机已验证输出：

```text
A: sha256=6d53177914ddc4e7de3a777e523df8059c1a2c1a66a3f70a0faa60cca0fe3ae5 cond=11.916079783100
matched 20261111: sha256=2e4445a6201f9b4fdec6e62bf805f59caa5bcb7b6012e05dc27224c254a6cb0c cond=11.916079783100
matched 20261112: sha256=117a5c013e061cd9e54de3c5538d9dda6a06abcec2a0ae55cd511eb57a6d6952 cond=11.916079783100
matched 20261113: sha256=5c809d096710aed706ce0aaea1f60a9b36564233d98556f56632c9810e015964 cond=11.916079783100
OK: all matrices reproduce the recorded hashes and spectra
```

---

## 6. 完整训练配置

### 6.1 全局配置

除特别说明外，18 个方法共享以下配置（来自各 `runs/*/run_config.json`）：

| 字段 | 值 | 字段 | 值 |
|---|---|---|---|
| epochs | 30 | min_epochs | 1 |
| patience | 7 | batch_size | 8 |
| learning_rate | 1e-4 | weight_decay | 0.01 |
| lr_schedule | constant | seed | 13 |
| video_layers | 12 | video_rank | 8 |
| video_alpha | 16 | video_dropout | 0.05 |
| lambda_feature | 2.0 | lambda_subset | 1.0 |
| kd_kl | 1.0 | icd_warmup | false |
| coordinate_weights | `[1,1,1,1,1,1,1]` | checkpoint_policies | `valid_mae;valid_acc2_nonzero;last` |

### 6.2 18 个方法的差异配置

| 方法 | mechanism | task_ce | λ_interaction | loss | matrix_seed | transform SHA256（前12位） |
|---|---|---:|---:|---|---:|---|
| `A_full_no_ce` | `none` | 0 | 0 | `huber` | 20261111 | `75f0bfc29ab9` |
| `B_N2_icd030` | `icd` | 0 | 0.3 | `huber` | 20261111 | `6d53177914dd` |
| `C_full_ce` | `none` | 0.5 | 0 | `huber` | 20261111 | `75f0bfc29ab9` |
| `D_icd030_ce` | `icd` | 0.5 | 0.3 | `huber` | 20261111 | `6d53177914dd` |
| `subset7` | `subset` | 0 | 0.3 | `huber` | 20261111 | `75f0bfc29ab9` |
| `matched0_huber` | `matched` | 0 | 0.3 | `huber` | 20261111 | `2e4445a6201f` |
| `matched1_huber` | `matched` | 0 | 0.3 | `huber` | 20261112 | `117a5c013e06` |
| `matched2_huber` | `matched` | 0 | 0.3 | `huber` | 20261113 | `5c809d096710` |
| `diagonal_mse` | `diagonal` | 0 | 0.3 | `mse` | 20261111 | `2e65afe9f22c` |
| `weighted_mse` | `weighted` | 0 | 0.3 | `mse` | 20261111 | `e0b1adac138a` |
| `weighted_huber` | `weighted` | 0 | 0.3 | `huber` | 20261111 | `e0b1adac138a` |
| `normalized_icd` | `normalized_icd` | 0 | 0.3 | `huber` | 20261111 | `e7f8ef5dff88` |
| `normalized_subset` | `normalized_subset` | 0 | 0.3 | `huber` | 20261111 | `76aa994101af` |
| `orthogonal_huber` | `orthogonal` | 0 | 0.3 | `huber` | 20261111 | `9c6729d84d23` |
| `icd_mse` | `icd` | 0 | 0.3 | `mse` | 20261111 | `6d53177914dd` |
| `matched0_mse` | `matched` | 0 | 0.3 | `mse` | 20261111 | `2e4445a6201f` |
| `F7_icd025` | `icd` | 0 | 0.25 | `huber` | 20261111 | `6d53177914dd` |
| `E4_icd050_ce` | `icd` | 0.5 | 0.5 | `huber` | 20261111 | `6d53177914dd` |

紧凑的 27 列配置（含 `transform_singular_values`、`transform_condition_number`、`coordinate_weights` 等）见 `tables/method_configs.csv`；**每个方法的完整 78 字段 `run_config.json`**（含模型路径、教师资产、校准温度、冻结指纹、checkpoint 保留策略等）见 `tables/full_run_configs.json`。

---

## 7. 逐 checkpoint 指标

`tables/per_checkpoint_metrics.csv` 共 **54 行** = 18 个方法 × 3 个 checkpoint 策略。每行包含：

- 选择策略、选中 epoch、checkpoint 完整 SHA256 与字节数；
- 测试集 `count / MAE / Pearson / Acc-2(nonzero) / weighted F1(nonzero) / Acc-7`；
- 对应验证集的同一组指标。

下面三张表按策略列出测试集指标；checkpoint SHA256 只显示前 12 位，完整 64 位在 CSV 的 `checkpoint_sha256` 列和 `fingerprints.json` 中。所有 Acc-2 / F1 均为 **non-zero 版本**，单位为百分比。

### 按验证集 MAE 选择 (`best_valid_mae.pt`)

| 方法 | epoch | Test MAE ↓ | Test Acc-2 ↑ | Test Pearson ↑ | Test F1 ↑ | checkpoint SHA256 (前12位) |
|---|---:|---:|---:|---:|---:|---|
| `A_full_no_ce` | 7 | 0.72596 | 82.32 | 0.7783 | 82.35 | `4aafdd53292d` |
| `B_N2_icd030` | 10 | 0.72962 | 84.30 | 0.7904 | 84.39 | `f03b576fc3f6` |
| `C_full_ce` | 16 | 0.71000 | 83.08 | 0.7896 | 82.99 | `8f9b6121eaa3` |
| `D_icd030_ce` | 15 | 0.69849 | 84.60 | 0.8002 | 84.54 | `d94b46bff817` |
| `subset7` | 19 | 0.71283 | 85.52 | 0.8076 | 85.58 | `caa9f89924b7` |
| `matched0_huber` | 11 | 0.72357 | 85.06 | 0.7951 | 85.10 | `579fb5b8fe91` |
| `matched1_huber` | 9 | 0.72034 | 84.60 | 0.7895 | 84.61 | `89a2489ed48b` |
| `matched2_huber` | 9 | 0.71935 | 84.76 | 0.7910 | 84.69 | `a70340cdc15f` |
| `diagonal_mse` | 13 | 0.69313 | 84.15 | 0.8169 | 84.17 | `348a6da0083f` |
| `weighted_mse` | 20 | 0.71540 | 85.06 | 0.7975 | 85.14 | `dd7f571a6b11` |
| `weighted_huber` | 15 | 0.75362 | 84.76 | 0.7931 | 84.83 | `76c179a34d7c` |
| `normalized_icd` | 17 | 0.71778 | 85.67 | 0.7995 | 85.73 | `9096541112b6` |
| `normalized_subset` | 3 | 0.81990 | 82.47 | 0.7656 | 82.57 | `dc72267298b8` |
| `orthogonal_huber` | 10 | 0.75560 | 84.45 | 0.7933 | 84.52 | `8a5aad5801d6` |
| `icd_mse` | 14 | 0.70382 | 85.67 | 0.7991 | 85.66 | `c7dfb203ee41` |
| `matched0_mse` | 10 | 0.73252 | 86.43 | 0.8050 | 86.46 | `185ff122468e` |
| `F7_icd025` | 6 | 0.75423 | 83.69 | 0.7830 | 83.71 | `9ad33bf2f686` |
| `E4_icd050_ce` | 13 | 0.72743 | 85.37 | 0.8013 | 85.45 | `b40397f7b460` |

### 按验证集 Acc-2 选择 (`best_valid_acc2_nonzero.pt`)

| 方法 | epoch | Test MAE ↓ | Test Acc-2 ↑ | Test Pearson ↑ | Test F1 ↑ | checkpoint SHA256 (前12位) |
|---|---:|---:|---:|---:|---:|---|
| `A_full_no_ce` | 5 | 0.75222 | 84.76 | 0.7898 | 84.80 | `24cc1b464e7a` |
| `B_N2_icd030` | 7 | 0.74995 | 84.60 | 0.7954 | 84.57 | `9c9bbef713f3` |
| `C_full_ce` | 19 | 0.71676 | 83.84 | 0.7938 | 83.87 | `73564f991e71` |
| `D_icd030_ce` | 21 | 0.72562 | 83.69 | 0.7901 | 83.55 | `f90303b74fc8` |
| `subset7` | 8 | 0.72320 | 85.21 | 0.8028 | 85.14 | `1581976c60dc` |
| `matched0_huber` | 14 | 0.68970 | 85.52 | 0.8124 | 85.35 | `7ccfe8cc3783` |
| `matched1_huber` | 3 | 0.75038 | 83.08 | 0.7795 | 82.96 | `2155b670f1bb` |
| `matched2_huber` | 11 | 0.70288 | 84.91 | 0.7997 | 84.76 | `690f11bbad25` |
| `diagonal_mse` | 10 | 0.68862 | 85.06 | 0.8121 | 84.92 | `e839243bcde8` |
| `weighted_mse` | 8 | 0.72532 | 84.45 | 0.7926 | 84.45 | `343368b1a7fc` |
| `weighted_huber` | 22 | 0.71451 | 84.30 | 0.7947 | 84.25 | `700e2690280d` |
| `normalized_icd` | 12 | 0.70989 | 85.82 | 0.8053 | 85.83 | `63921578246f` |
| `normalized_subset` | 4 | 0.76542 | 84.30 | 0.7919 | 84.20 | `091fa4ad5fac` |
| `orthogonal_huber` | 11 | 0.74491 | 84.91 | 0.8010 | 84.97 | `9bc397bf29d9` |
| `icd_mse` | 3 | 0.76965 | 84.30 | 0.7840 | 84.22 | `13973c5600c1` |
| `matched0_mse` | 5 | 0.77097 | 85.06 | 0.8030 | 84.94 | `2314e04347a5` |
| `F7_icd025` | 6 | 0.75423 | 83.69 | 0.7830 | 83.71 | `2a87d3689c2c` |
| `E4_icd050_ce` | 12 | 0.71897 | 84.76 | 0.8013 | 84.85 | `378dd76013d2` |

### 最后一个 epoch (`last.pt`，非验证选择)

| 方法 | epoch | Test MAE ↓ | Test Acc-2 ↑ | Test Pearson ↑ | Test F1 ↑ | checkpoint SHA256 (前12位) |
|---|---:|---:|---:|---:|---:|---|
| `A_full_no_ce` | 15 | 0.72804 | 83.69 | 0.7834 | 83.74 | `a455bde3d089` |
| `B_N2_icd030` | 18 | 0.69732 | 84.91 | 0.8038 | 84.84 | `c042474147c8` |
| `C_full_ce` | 23 | 0.72319 | 82.47 | 0.7855 | 82.36 | `1084f71ada51` |
| `D_icd030_ce` | 22 | 0.75039 | 84.76 | 0.7923 | 84.72 | `a31487bfb485` |
| `subset7` | 26 | 0.72146 | 84.76 | 0.8027 | 84.67 | `40ccd5dcabac` |
| `matched0_huber` | 18 | 0.73772 | 84.60 | 0.7907 | 84.65 | `791e6607138f` |
| `matched1_huber` | 16 | 0.77780 | 84.76 | 0.7989 | 84.84 | `21308e110bf8` |
| `matched2_huber` | 16 | 0.70211 | 85.21 | 0.8087 | 85.15 | `c47d8893d4bb` |
| `diagonal_mse` | 20 | 0.71201 | 84.60 | 0.8147 | 84.68 | `fd5dffe1f1ce` |
| `weighted_mse` | 27 | 0.77340 | 84.15 | 0.7964 | 84.24 | `3bfac96aeb9b` |
| `weighted_huber` | 22 | 0.71451 | 84.30 | 0.7947 | 84.25 | `dbbafdf95e47` |
| `normalized_icd` | 24 | 0.68506 | 85.98 | 0.8079 | 85.98 | `b0e67dddd1d4` |
| `normalized_subset` | 10 | 0.72350 | 82.01 | 0.7853 | 81.71 | `a17fabfbc62a` |
| `orthogonal_huber` | 17 | 0.72838 | 84.45 | 0.7950 | 84.36 | `281bf8eeb768` |
| `icd_mse` | 21 | 0.72401 | 86.13 | 0.8013 | 86.14 | `402df2edc1ec` |
| `matched0_mse` | 17 | 0.76313 | 85.67 | 0.7917 | 85.58 | `5c9591085f03` |
| `F7_icd025` | 13 | 0.84634 | 83.99 | 0.7979 | 84.09 | `6114c862d369` |
| `E4_icd050_ce` | 20 | 0.70882 | 84.76 | 0.7987 | 84.83 | `ff5f2fa30890` |


---

## 8. 源码、数据与 checkpoint 指纹

### 8.1 源码指纹

| 文件 | 字节 | SHA256 |
|---|---:|---|
| `source/mechanisms.py` | 8,246 | `bcc0c44f1c25977728af0ecc7c508c33aef2c5887e18d63829262c142cf4a7be` |
| `source/train_mosi.py` | 61,433 | `75547af5a8a89aea87055068549825e980c269709faaf63606f388bbd9864e89` |
| `source/prepare.py` | 10,344 | `aafd755a6f10df9cbaa38a04016fa5453d6ab6e66e6563f2e760aab8376db3e4` |
| `source/evaluate_mosi.py` | 12,018 | `8068e9b1daab0e19f1c572ba76f6cfaf6cbecfa688bed7efd3fe6fc02eacb9bf` |
| `source/summarize.py` | 2,676 | `2a588078fcb4ddb0ef6db7366082b391c4c0baf051a2d7f769309765b7120830` |
| `source/plan.json` | 6,102 | `42ccb34a0bacec5f62ad2bfed8f7fc433da8b4de2b0dc5d4100ff62f32c3f114` |
| `source/verify_geometry.py` | 5,288 | `76d8164a98919ebfb49b695cf357c271b019b05d27d66e25dc02181faf2f483f` |
| `source/build_tables.py` | 5,224 | `eba54d686c42609b6954595ecf3bbda0d6636031aa5ff74b048c369586063499` |
| `source/worker.py` | 7,452 | `c4bfd13571e4d429c6cefd37bcbb6d1f7da63314310e4cb08ef8d908dd72442d` |
| `source/运行说明.md` | 4,247 | `a1cfbd3752222cf1b13e3fd2d6aa9c2d8aee3d5a53bedc5d664dc82261e06979` |
| `assets/geometry.json` | 6,603 | `564298d80bf06259afe86183bcfb193f86cc96cf29347a727e175c9d71d1e800` |
| `assets/matched_matrices.json` | 6,796 | `23ddec325ecafa7cb055e31e7b82fda2fe42085d7f40b5839dcce5da42997f76` |
| `assets/normalization.json` | 684 | `999268a2e3b3bcbca5719ea2470a582973028ac725cb257e264ff409c766465b` |
| `assets/teacher_manifest.json` | 11,119 | `a5639a43607184256cbb8e9c782b1612e938203c4ef26b9373465f652b675c34` |
| `results/summary.md` | 5,365 | `cfdf0ab3900e346ff947b8c383bda2d7f30602a07757298221994ba7fbd7deb1` |
| `results/summary.json` | 35,333 | `7f1f9720d652d18a83c9470392a713f0550ea64357094c47e1910543b4660d63` |

> `source/mechanisms.py`、`train_mosi.py`、`prepare.py`、`evaluate_mosi.py`、`summarize.py` 与冻结 campaign 中的同名文件逐字相同；`source/plan.json` 是冻结的 18 组配置。

### 8.2 数据与教师资产

详见第 1.3 节与 `fingerprints.json` 的 `raw_manifests`、`windowed_manifests`、`split_hashes`、`teacher_and_assets` 四个对象。`fingerprints.json` 还记录了每个资产的字节数、行数和 raw/normalized 两种 hash。

### 8.3 checkpoint SHA256

完整 64 位 SHA256 在 `tables/per_checkpoint_metrics.csv` 的 `checkpoint_sha256` 列（54 行），以及 `fingerprints.json` 的 `checkpoints` 对象（按方法名和文件名分组）。CSV 同时给出 `checkpoint_bytes`、`selected_epoch`、测试指标和验证指标。本次整理已对 54 个 `.pt` 文件逐个重算 SHA256 与字节数，**54/54 与 CSV 完全一致**。

示例（`A_full_no_ce`）：

| policy | epoch | bytes | checkpoint SHA256 |
|---|---:|---:|---|
| `valid_mae` | 7 | 170,824,699 | `4aafdd53292dcfcfee70b416b5b7b29230c35884ede0369aa86d81401195e229` |
| `valid_acc2_nonzero` | 5 | 170,830,217 | `24cc1b464e7ae93703e943facd66fcb13a655ce5cd51766b795d6087dca0388e` |
| `last` | 15 | 512,497,981 | `a455bde3d0891490deadf986ee86716d7e58104b40a5574f0b95bce350357025` |

`fingerprints.json` 的结构：

```json
{
  "checkpoints": {
    "A_full_no_ce": {
      "best_valid_mae.pt": {"sha256": "...", "bytes": 170824699, "epoch": 7},
      "best_valid_acc2_nonzero.pt": {"sha256": "...", "bytes": 170830217, "epoch": 5},
      "last.pt": {"sha256": "...", "bytes": 170..., "epoch": 30}
    }
  }
}
```

---

## 9. 复现命令

以下命令假定当前目录为 `project/reports/mosi_first_20261111_v1`，且使用带 torch 的环境 `/ai/sjq/miniconda3/bin/python`。

### 9.1 校验几何矩阵

```bash
cd project/reports/mosi_first_20261111_v1
/ai/sjq/miniconda3/bin/python source/verify_geometry.py
```

期望输出以 `OK: all matrices reproduce the recorded hashes and spectra` 结尾。

### 9.2 重建三个 CSV

需要有冻结 campaign 输出目录（含 `plan.json`、`runs/*`、`tests/*`）：

```bash
cd project/reports/mosi_first_20261111_v1
/ai/sjq/miniconda3/bin/python source/build_tables.py \
  --campaign-root /ai/sjq/kd/outputs/experiments/mosi_first_20261111_v1 \
  --output tables
```

会写出：

```text
tables/per_checkpoint_metrics.csv   54 rows
tables/method_configs.csv           18 rows
tables/transform_matrices.csv       18 rows
```

### 9.3 单个方法重训（示例：`diagonal_mse`）

完整协议见 `source/运行说明.md`；调度命令映射见 `source/worker.py`。`none` 机制使用 `--method full_kd`，其他机制使用 `--method uniform_interaction`：

```bash
/ai/sjq/miniconda3/bin/python source/train_mosi.py \
  --method uniform_interaction \
  --mechanism diagonal \
  --loss-kind mse \
  --matrix-seed 20261111 \
  --experiment-id diagonal_mse_seed13 \
  --assets /ai/sjq/kd/outputs/experiments/mosi_first_20261111_v1/assets/protocol.json \
  --output /ai/sjq/kd/outputs/experiments/mosi_first_20261111_v1/runs/diagonal_mse_seed13 \
  --seed 13 --device cuda:0 --batch-size 8 --epochs 30 --patience 7 \
  --retain-best-mae-and-acc2 --checkpoint-selection valid_mae \
  --learning-rate 1e-4 --weight-decay 0.01 \
  --task-ce 0 --lambda-interaction 0.3
```

> 该命令需要原始 MOSI 数据、教师缓存、三个学生骨干权重和 GPU；本复现包不包含这些大文件。不同机器的路径需按实际位置替换。

### 9.4 单 checkpoint 测试

```bash
/ai/sjq/miniconda3/bin/python source/evaluate_mosi.py \
  --run /ai/sjq/kd/outputs/experiments/mosi_first_20261111_v1/runs/diagonal_mse_seed13 \
  --output /ai/sjq/kd/outputs/experiments/mosi_first_20261111_v1/tests/diagonal_mse_seed13__valid_mae \
  --test-manifest /ai/sjq/kd/outputs/experiments/mosi_first_20261111_v1/assets/official_test_windowed.jsonl \
  --checkpoint-selection valid_mae \
  --allow-official-test
```

`--checkpoint-selection` 取 `valid_mae`、`valid_acc2_nonzero` 或 `last`，分别对应三个已保存的 checkpoint。测试集只生成报告，不参与任何选择。

---

## 10. 解释边界与注意事项

1. **单 seed**：本 campaign 只跑学生 seed 13；没有跨 seed 均值、标准差或置信区间。表里的 `±` 若出现在其他汇总中，是样本标准差，不是 CI。
2. **`last` 不是主结果**：`normalized_icd` 的 `last` 为 0.685065，低于 `diagonal_mse` 的验证选择结果；但 `last` 没有经过验证集选择，不能与 `valid_mae` / `valid_acc2_nonzero` 混在一起排名。
3. **测试集只报告**：18 个方法的测试结果都是一次性报告；没有用测试集调参、选 checkpoint 或选方法。
4. **matched-spectrum 的边界**：三个 matched 矩阵只匹配 `A` 的奇异值，不匹配 `A^T A` 的非对角结构；因此它排除的是“谱效应”，不是所有几何效应。
5. **`diagonal_mse` 的边界**：`D` 只按 `A` 的列范数缩放，未使用验证/测试统计量；它是固定线性变换，不是可学习矩阵。
6. **跨 campaign 比较**：`E3_ea_kd_full`（0.668253）和 `F7`（0.6812）来自不同 campaign / 协议；与本 campaign 的 0.693131 直接比较时，应同时看数据窗口、教师缓存、训练预算与 checkpoint 策略差异。
7. **路径迁移**：历史路径 `/data/sjq/kd` 已迁移到 `/ai/sjq/kd`；manifest 的 raw hash 与 normalized hash 不同是预期现象，比较时使用同一归一化口径。
8. **教师权重缺失**：三个原始 Probe 权重缺失，但缓存的教师预测已核验；不能从本包重新训练教师，只能复现学生侧和几何分析。

---

## 11. 结论摘要

- `diagonal_mse` 在本 campaign 的 18 个方法中取得最低测试 MAE：
  - `valid_mae` 选择：epoch 13，MAE **0.693131**，Acc-2 **84.15%**，Pearson **0.8169**，F1 **84.17%**；
  - `valid_acc2_nonzero` 选择：epoch 10，MAE **0.688624**，Acc-2 **85.06%**，Pearson **0.8121**，F1 **84.92%**。
- `matched0_huber` 的 `valid_acc2_nonzero` 选择为 MAE 0.689688，与 `diagonal_mse` 非常接近；这说明“低 MAE”不能单独归因于对角缩放，仍需多 seed 验证。
- 标准 ICD（`B_N2_icd030`）的 `valid_mae` 结果为 MAE 0.729619，说明几何辅助项本身不足以超过强 full-KD 基线。
- `normalized_icd` 的 `last` 为 0.685065，但 `last` 非验证选择；其 `valid_mae` / `valid_acc2_nonzero` 分别为 0.7178 / 0.7099，不能据此宣称 normalized ICD 更优。
- 本 campaign 没有任何学生模型超过教师模型的任务指标；也没有方法超过既有 `E3_ea_kd_full` 0.668253 或 `F7` 0.6812。
- 下一步优先级：对 `diagonal_mse`、`matched0_huber`、`normalized_icd` 做多 seed（至少 13/42/2026）复验，并固定同一训练预算后再比较。

---

## 12. 引用

- 主汇总报告：`docs/reports/实验汇总报告_20261111.md`（第 5.7.1 节 MOSI First 矩阵）。
- 冻结 campaign：`outputs/experiments/mosi_first_20261111_v1/`。
- 复现包：`project/reports/mosi_first_20261111_v1/`。
