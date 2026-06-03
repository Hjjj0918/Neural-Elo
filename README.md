# Neural-Elo

A Deep Learning-based matchmaking engine replacing traditional 1D Elo with high-dimensional features for fairer and smarter competitive gaming.

[中文版](./README_CN.md)

---

## Overview

Traditional Elo rates every player with a single number. Neural-Elo replaces that with a neural network that ingests rich team and player features — economy, weapons, map dynamics, player embeddings — and predicts win probability for any team split. The matchmaker then searches for the split closest to 50%.

---

## System Architecture

```
Queue (N players)
  |
  v
Elo Coarse Filter          -- group players within ±200 Elo band → candidate pools
  |
  v
Candidate Split Generator  -- for each 10-player pool, enumerate 126 ways to split into 5v5
  |
  v
Neural Scorer              -- batch-infer all splits → score by |P(Team A wins) - 0.50|
  |
  v
Best Match                 -- select split with P closest to 50% → send to server
```

### Inside the Neural Scorer

```
  Team A Features          Team B Features
  (Elo mean/std, CT-T      (Elo mean/std, CT-T
   deltas, map one-hot,     deltas, map one-hot,
   player embeddings*)      player embeddings*)
         |                        |
         +-----> Shared MLP <-----+
                    |
            Team Embeddings zA, zB
                    |
            P = σ(w · [zA - zB] + b)
                    |
               P(Team A wins)
```

*\*Player embeddings are Phase 4; Baseline uses team-aggregated features only.*

---

## Key Design Decisions

| Decision | Choice | Rationale |
|:---|:---|:---|
| Match scenario | **Ranked Ladder (5v5 from queue)** | Primary use case; team-vs-team deferred |
| Player modeling | **Hybrid**: Player embeddings + team aggregate stats | Best of both: individual skill + team synergy |
| Data strategy | **Two-stage**: Baseline model first, embeddings later | Current dataset lacks player IDs; deliver working pipeline now |
| Match search | **Elo pre-filter + neural fine-rank** | Scales to large queues; Elo provides cheap coarse filter |
| Elo / NN relationship | **Elo as input feature** | Model learns when to trust Elo vs. when to trust other features |

---

## Training Phases

### Phase 2-3: Baseline Model (current)

- **Input**: Team-aggregated features only (CT-T deltas, map one-hot, etc.)
- **Model**: XGBoost → MLP benchmark → MLP + RNN (progressive)
- **Data**: CS:GO Round Snapshots (Kaggle, ~122K rows)
- **Output**: P(Team A wins) — no player IDs required
- **Goal**: End-to-end working pipeline (train → predict → match)

### Phase 4: Player Embedding Upgrade (future)

- **Input**: Team features + learnable player embedding vectors (dim ~64–128)
- **Model**: MLP encoder + embedding layer, pooled across 5 players per team
- **Data**: Requires player-level match history (FACEIT API, Steam logs, etc.)
- **Output**: Same win probability, but personalized per player

---

## Project Structure

```
Neural-Elo/
├── README.md
├── README_CN.md
├── requirements.txt
├── phase1_eda.py                 # EDA: data loading, cleaning, correlation
├── data/
│   ├── download_dataset.py       # Kaggle dataset downloader (kagglehub)
│   └── csgo_round_snapshots.csv  # ~122K rows x 97 cols (git-ignored)
└── outputs/
    └── phase1/                   # EDA charts & summary report
```

---

## Quick Start

```bash
pip install -r requirements.txt
python data/download_dataset.py   # downloads ~48 MB from Kaggle
python phase1_eda.py              # runs full EDA, outputs charts to outputs/phase1/
```
