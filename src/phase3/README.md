# Phase 3: Model Training / 模型训练

**Deliverable**：训练好的 LightGBM 基线模型 + MLP 神经网络，Test 集上的对比评估报告。

---

## Pipeline Flow / 流水线

```
data/processed/
  X_train.npy (82K × 38)    ──┐
  X_val.npy   (18K × 38)    ──┤ Step 1: Load       ──┐
  X_test.npy  (18K × 38)    ──┤                      │
  y_train/val/test.npy      ──┘                      │
                                                     ├──> Step 2: LightGBM ──> lgbm_baseline.pkl
                                                     │         ├─ Train: 500 trees, early stop after 50 rounds no improvement
                                                     │         ├─ Output: AUC, LogLoss, feature importances
                                                     │         └─ Time: ~5 seconds on CPU
                                                     │
                                                     ├──> Step 3: MLP ──> mlp_baseline.pt
                                                     │         ├─ Architecture: 38 → [128, 64, 32] → 1 (sigmoid)
                                                     │         ├─ Training: Adam lr=0.001, BatchNorm, Dropout 0.3
                                                     │         ├─ Early stop: patience=15 epochs
                                                     │         └─ Time: ~45 seconds on CPU
                                                     │
                                                     └──> Step 4: Evaluate ──> evaluation_report.json
                                                               ├─ ROC-AUC comparison
                                                               ├─ LogLoss comparison
                                                               └─ Confusion matrices (threshold=0.5)
```

## Step-by-Step / 分步说明

### Step 1: Load / 加载数据

直接 `np.load()` Phase 2 产出的 6 个 `.npy` 文件 + `feature_names.json`。验证维度对齐、标签平衡。

### Step 2: LightGBM Baseline / 基线树模型

| 参数 | 值 | Why |
|:---|:---|:---|
| `n_estimators` | 500 | 足够多，early stopping 自动截断 |
| `learning_rate` | 0.05 | 适中，不冒进 |
| `max_depth` | 6 | 38 个特征不需要深树 |
| `num_leaves` | 31 | 每层最多 31 个叶子 |
| `reg_alpha/lambda` | 0.1 | L1 + L2 双重正则化 |
| `subsample` | 0.8 | 行采样防过拟合 |
| `colsample_bytree` | 0.8 | 列采样防过拟合 |
| `early_stopping` | 50 rounds | Val AUC 50 轮不涨就停 |

**Output**：
- Train / Val / Test AUC + LogLoss
- Top 15 特征重要性（gain-based，越高越重要）
- `models/lgbm_baseline.pkl`

### Step 3: MLP Neural Network / 神经网络

```
Input (38)
  -> Linear(128) -> BatchNorm -> ReLU -> Dropout(0.3)
  -> Linear(64)  -> BatchNorm -> ReLU -> Dropout(0.3)
  -> Linear(32)  -> BatchNorm -> ReLU -> Dropout(0.3)
  -> Linear(1)   -> Sigmoid
  -> Output: P(CT wins) ∈ [0, 1]
```

| 参数 | 值 | Why |
|:---|:---|:---|
| 隐藏层 | [128, 64, 32] | 金字塔压缩，逐步提取高层模式 |
| Dropout | 0.3 | 每层随机丢 30% 神经元 |
| BatchNorm | 每层后 | 稳定训练，加快收敛 |
| 优化器 | Adam (lr=0.001) | 自适应学习率 |
| Loss | Binary Cross-Entropy | 二分类标准损失 |
| Batch Size | 512 | 平衡速度 & 稳定性 |
| Early Stop | patience=15 | Val loss 不降就停 |

**Output**：
- Train / Val / Test AUC + LogLoss
- `models/mlp_baseline.pt`（含权重 + 配置）
- `models/mlp_history.json`（每 epoch 的 loss & AUC）

### Step 4: Evaluation & Comparison / 评估对比

| 指标 | 含义 | 为什么用 |
|:---|:---|:---|
| ROC-AUC | 排序能力：随机抽一个 CT 局和一个 T 局，模型给 CT 局更高分的概率 | 不依赖阈值 |
| LogLoss | 概率校准：预测概率和真实结果的差距 | 匹配系统需要精确的 P(胜率) |
| Confusion Matrix | TP/TN/FP/FN（threshold=0.5） | 直观看到预测错误类型 |

**Output**：`models/evaluation_report.json`

## Current Results / 当前结果 (2026-06-07)

```
Metric        LightGBM      MLP
────────────────────────────────
Test AUC      0.8919      0.8833
Test LogLoss  0.4004      0.4133

Winner: LightGBM (AUC +0.0087, LogLoss -0.0129)
```

**Top 5 Feature Importances (LightGBM)**：

| Rank | Feature | Meaning |
|:---|:---|:---|
| 1 | `money_per_player_ct` | CT 人均经济 — ECO or full buy is everything |
| 2 | `money_per_player_t` | T 人均经济 — same |
| 3 | `delta_armor` | Armor differential — armor vs no armor |
| 4 | `time_remaining_pct` | Time pressure — T running out of time to plant |
| 5 | `equip_score_t` | T equipment quality — armor + helmet investment |

Economy dominates — confirming CS:GO's money system is the strongest predictor.

## Usage / 运行方式

```bash
python -m src.phase3.run_all
# 或
python phase3_eda.py
```

## Output Files / 输出文件

```
models/
├── lgbm_baseline.pkl        # LightGBM model (production-ready baseline)
├── mlp_baseline.pt          # MLP weights + config
├── mlp_best.pt              # MLP best-epoch weights
├── mlp_history.json         # Training curves per epoch
└── evaluation_report.json   # Final comparison report
```

## What We Skip & Why / 跳过的内容

| Skipped | Reason |
|:---|:---|
| XGBoost | LightGBM faster on sparse features, same performance tier |
| Hyperparameter tuning (Grid/Random Search) | Validate feasibility first; manual params are close enough |
| K-Fold CV | Independent val/test sets already exist |
| Transformer / RNN | Phase 4 scope |
| Class weights / SMOTE | Labels are 49/51 — already balanced |
| Multi-task (rounds prediction) | Phase 4 scope |
