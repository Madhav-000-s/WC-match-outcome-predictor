# World Cup 2026 Match Outcome Predictor

Predicts Win/Draw/Loss probabilities for international football matches and simulates the 2026 FIFA World Cup using a Dixon-Coles Bivariate Poisson model.

---

## Overview

The model estimates latent attack and defense strength parameters for each national team, then uses those to generate scoreline distributions for any head-to-head fixture. A walk-forward backtest on 15 years of historical data validates the approach before projecting the 2026 tournament.

Key design choices:
- **Dixon-Coles correction** — adjusts probabilities for low-scoring outcomes (0-0, 1-0, 0-1, 1-1), which a plain Poisson model systematically misprices
- **Exponential time decay** — matches from 3+ years ago contribute far less to the fit; grid search found a 1,095-day half-life optimal
- **Competition weights** — World Cup finals (1.0) weighted higher than qualifiers (0.8) and friendlies (0.3)
- **Home advantage** — a single multiplicative parameter (γ) fitted across all home sides; host-nation override applied for USA, Mexico, and Canada

---

## Dataset

[International Football Results 1872-2025](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) from Kaggle.

| File | Description |
|------|-------------|
| `football-data/results.csv` | Match results (date, teams, scores, tournament, venue) |
| `football-data/goalscorers.csv` | Individual goal events (not used in v1) |
| `football-data/shootouts.csv` | Penalty shootout outcomes (not used in v1) |
| `football-data/former_names.csv` | Historical team name aliases |

**Training window**: 2010–2026 · **15,135 matches** · **217 teams**

---

## Results

### Backtest performance (walk-forward validation)

| Metric | Model | Base-rate baseline |
|--------|-------|--------------------|
| Ranked Probability Score (RPS) | **0.1657** | 0.2283 |
| Improvement over baseline | **26.9%** | — |

*Lower RPS is better. Base-rate uses historical win/draw/loss frequencies.*

### Fitted parameters

| Parameter | Value |
|-----------|-------|
| Home advantage multiplier | **1.31×** (γ = 0.245) |
| Dixon-Coles dependence (ρ) | −0.045 |
| Best attack | Colombia, Brazil, Argentina (~1.39) |
| Best defense | Brazil (−1.33), Argentina (−1.29), Spain (−1.25) |

### 2026 World Cup championship probabilities

30,000 Monte Carlo simulations across the full 48-team bracket.

| Team | Win probability |
|------|----------------|
| Spain | 13.6% |
| Argentina | 11.9% |
| Brazil | 10.3% |

---

## Repo structure

```
WC match outcome predictor/
├── README.md
└── football-data/
    ├── results.csv
    ├── goalscorers.csv
    ├── shootouts.csv
    └── former_names.csv
```

---

## Usage

**Dependencies**: `numpy`, `scipy`, `pandas`

```bash
pip install numpy scipy pandas
```

The full implementation lives in the Kaggle notebook. To run locally, open the notebook in Jupyter and point the data paths at the `football-data/` directory.
