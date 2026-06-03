# Neural-Elo

基于深度学习的匹配引擎，用高维特征替代传统一维 Elo 评分，为竞技游戏提供更公平、更智能的匹配系统。

[English version](./README.md)

---

## 概述

传统 Elo 用单一数字评价每个玩家。Neural-Elo 用神经网络取而代之——输入丰富的队伍和玩家特征（经济、武器、地图动态、玩家嵌入向量），预测任意分队方案下的胜率，然后搜索胜率最接近 50% 的分队方案。

---

## 系统架构

```
排队队列 (N 名玩家)
  |
  v
Elo 粗筛                    -- 将玩家按 Elo ±200 分入候选池
  |
  v
候选分队生成器               -- 对每个 10 人候选池，枚举 126 种 5v5 分法
  |
  v
神经网络打分器               -- 批量推理所有分队方案 → 用 |P(Team A) - 0.50| 打分
  |
  v
最佳匹配                    -- 选出胜率最接近 50% 的分队方案 → 送入服务器
```

### 神经网络打分器内部

```
  队伍 A 特征               队伍 B 特征
  (Elo 均值/标准差,          (Elo 均值/标准差,
   CT-T 差值,                 CT-T 差值,
   地图 One-Hot,              地图 One-Hot,
   玩家 Embedding*)           玩家 Embedding*)
         |                        |
         +-----> 共享 MLP <-------+
                    |
            队伍嵌入 zA, zB
                    |
            P = σ(w · [zA - zB] + b)
                    |
             P(队伍 A 获胜)

*玩家 Embedding 属于阶段四；Baseline 仅使用队伍聚合特征。
```

---

## 关键设计决策

| 决策点 | 选择 | 理由 |
|:---|:---|:---|
| 匹配场景 | **天梯排位 (队列中组 5v5)** | 首要场景；队伍间匹配后续再做 |
| 玩家建模 | **混合方案**：玩家 Embedding + 队伍聚合统计 | 兼顾个体实力与团队协同 |
| 数据策略 | **两阶段**：先做基线模型，后加玩家嵌入 | 现有数据集缺少玩家 ID；先交付可运行的完整流水线 |
| 匹配搜索 | **Elo 粗筛 + 神经网络细排** | 能应对大规模队列；Elo 提供廉价的初筛 |
| Elo / 网络关系 | **Elo 作为输入特征** | 模型自行学习何时信 Elo、何时信其他特征 |

---

## 训练阶段

### 阶段二～三：基线模型（当前）

- **输入**：仅队伍聚合特征（CT-T 差值、地图 One-Hot 等）
- **模型**：XGBoost → MLP 基准 → MLP + RNN（逐步升级）
- **数据**：CS:GO 回合快照数据集（Kaggle，约 12.2 万行）
- **输出**：P(队伍 A 获胜) — 无需玩家 ID
- **目标**：端到端可运行流水线（训练 → 预测 → 匹配）

### 阶段四：玩家 Embedding 升级（未来）

- **输入**：队伍特征 + 可学习的玩家 Embedding 向量（维度 ~64–128）
- **模型**：MLP 编码器 + Embedding 层，每队 5 名玩家池化
- **数据**：需要玩家级对局历史（FACEIT API、Steam 对局记录等）
- **输出**：相同的胜率预测，但针对每个玩家做个性化

---

## 项目结构

```
Neural-Elo/
├── README.md
├── README_CN.md                   # 本文件
├── requirements.txt
├── phase1_eda.py                  # EDA：数据加载、清洗、相关性分析
├── data/
│   ├── download_dataset.py        # Kaggle 数据集下载器（kagglehub）
│   └── csgo_round_snapshots.csv   # ~12.2万行 × 97列（git-ignored）
└── outputs/
    └── phase1/                    # EDA 图表与摘要报告
```

---

## 快速开始

```bash
pip install -r requirements.txt
python data/download_dataset.py   # 从 Kaggle 下载约 48 MB 数据
python phase1_eda.py              # 运行完整 EDA，图表输出到 outputs/phase1/
```
