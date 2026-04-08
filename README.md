# 🏎️ Lights Out: Predicting F1 Podiums with Machine Learning

> Can we predict, using only pre-race information, whether a driver will finish on the podium in a Formula 1 Grand Prix?

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Overview

This project builds a binary classification model to predict F1 podium finishes (top 3) using only pre-race data — qualifying times, driver form, constructor performance, and championship standings. No post-race data is used at any point.

**Key results:**
- **93.4% ROC-AUC** on a held-out 2023–2024 test set (vs 79.6% naive baseline)
- **2/3 correct podium picks per race** across the full 2024 season (24 races)
- **5x improvement** over random baseline (Average Precision: 0.74 vs 0.15)

---

## 📁 Project Structure

```
f1-podium-predictor/
│
├── data/                          # Kaggle F1 dataset CSVs (14 files)
│   ├── races.csv
│   ├── results.csv
│   ├── qualifying.csv
│   ├── drivers.csv
│   ├── constructors.csv
│   ├── circuits.csv
│   ├── driver_standings.csv
│   ├── constructor_standings.csv
│   └── ...
│
├── plots/                         # All generated visualizations (14 plots)
│   ├── podium_share.png
│   ├── grid_vs_podium.png
│   ├── feature_correlations.png
│   ├── roc_pr_curves.png
│   ├── confusion_matrices.png
│   ├── feature_importance.png
│   ├── race_by_race_2024.png
│   └── ...
│
├── f1_podium_prediction.ipynb     # Main notebook (all phases)
└── README.md
```

---

## 🗂️ Dataset

**Source:** [Kaggle — Formula 1 World Championship (1950–2024)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) by Rohanrao

**Scope:** 2018–2024 (post-hybrid era, consistent regulations)

| Stat | Value |
|---|---|
| Races | 149 Grand Prix |
| Race entries | 2,979 |
| Drivers | 40 |
| Constructors | 16 |
| Circuits | 31 |
| Target (podium) rate | 15.0% |

---

## ⚙️ Feature Engineering

23 features engineered across 6 groups — all using strictly past data (`.shift(1)`) to prevent leakage:

| Group | Features |
|---|---|
| Qualifying & Grid | `grid`, `qual_position`, `qual_gap_to_pole_pct`, `qual_session`, `reached_q3` |
| Driver Form | `driver_form_5` (EWMA), `driver_form_3`, `driver_circuit_avg`, `age` |
| Constructor | `constructor_form_5`, `constructor_reliability` (rolling DNF rate) |
| Championship | `champ_points_before`, `champ_pos_before`, `champ_wins_before`, `con_points_before`, `con_pos_before` |
| Track | `is_street_circuit`, `is_high_altitude`, `drs_zones`, `alt` |
| Contextual | `home_race`, `round`, `grid_x_champ`, `points_gap_to_leader` |

**Key engineering decisions:**
- Qualifying times normalized as % gap to pole — comparable across all circuits
- Championship standings shifted by 1 race — standings *before* the current race only
- Rolling features use `.shift(1)` — current race result never used as input

---

## 🤖 Models

Two classifiers compared:

**1. Logistic Regression (Lasso L1 + Ridge L2)**
- `LogisticRegressionCV` auto-tunes regularization strength `C`
- `class_weight='balanced'` handles 85/15 class imbalance
- Lasso zeroed 4 of 23 features — automatic feature selection

**2. Random Forest**
- `RandomizedSearchCV` with 40 iterations for hyperparameter tuning
- Handles multicollinearity and non-linear interactions natively
- Gini impurity feature importance for interpretability

**Validation strategy:** `TimeSeriesSplit` (5 folds) — train on past seasons, validate on future. The final test set (2023–2024) was strictly held out throughout all development.

---

## 📊 Results

### Test Set Performance (2023–2024)

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| Naive Baseline (grid ≤ 3) | 0.796 | 0.655 | 0.657 | 0.652 |
| Logistic Regression (Lasso) | 0.926 | 0.599 | 0.451 | 0.891 |
| Logistic Regression (Ridge) | 0.923 | 0.595 | 0.444 | 0.899 |
| **Random Forest** | **0.934** | **0.649** | **0.520** | **0.862** |

### 2024 Season Race-by-Race

| Model | Avg Hits/Race | Hit Rate |
|---|---|---|
| Random Forest | 1.92 / 3 | 63.9% |
| Lasso LR | 1.96 / 3 | 65.3% |

**Notable miss:** São Paulo 2024 (0/3) — Ocon and Gasly podiumed after a safety car in wet conditions, one of the most chaotic results in recent F1 history. No pre-race model could reasonably predict this.

### Top Features (Random Forest)

| Feature | Importance |
|---|---|
| `qual_position` | 0.172 |
| `grid_x_champ` | 0.171 |
| `qual_gap_to_pole_pct` | 0.135 |
| `grid` | 0.122 |
| `champ_pos_before` | 0.088 |

> **Key finding:** Qualifying performance dominates over championship standings — what happens on Saturday predicts Sunday better than the points table.

---

## 📈 Key EDA Findings

- **P1 starters podium 79% of the time** — grid position is the single strongest predictor
- **Q3 drivers podium at 29.4% vs 0.8%** for Q1-only drivers — a 37x difference
- **Home race advantage: +3.87pp** podium probability
- **Street circuit upsets are a myth** — upset rate from P4+ is identical on street (4.7%) vs normal circuits (4.8%)
- **Championship leaders dominant:** drivers within 10 points of the leader podium 70% of the time

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/f1-podium-predictor.git
cd f1-podium-predictor
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn fastf1
```

### 3. Download the data
Download the Kaggle F1 dataset and place all CSVs in the `data/` folder:
[kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)

### 4. Run the notebook
```bash
jupyter notebook f1_podium_prediction.ipynb
```
Run all cells top to bottom. Each phase is clearly labelled.

---

## 🔮 Future Work

- **XGBoost + SHAP** for per-prediction interpretability
- **FastF1 telemetry** — Friday practice long-run pace as race pace proxy
- **Two-stage pipeline** — predict DNF first, then finishing position
- **Real-time in-race updating** — live lap time and safety car integration
- **2026 regulation monitoring** — new power units will cause concept drift

---

## 📚 References

- Rohanrao. *Formula 1 World Championship (1950–2024)*. Kaggle.
- El Haber et al. (2025). *F1 Race Winner Prediction Using Random Forest and SHAP*. IEEE IC2AI.
- Sasikumar et al. (2025). *Data-driven Pit Stop Decision Support for F1*. Frontiers in AI.
- Polishchuk, I. (2025). *Predicting F1 Podiums with 78% Accuracy*. Medium.
- Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*DAT402 Project 2 · Built with Python, scikit-learn, and too many late nights watching qualifying sessions*
