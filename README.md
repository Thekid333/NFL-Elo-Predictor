# NFL Elo Predictor

A machine learning system that predicts NFL game outcomes by combining a custom dual Elo rating engine with an XGBoost classifier trained on a decade of historical data.

---

## How It Works

The system has two layers working together:

**Layer 1 — Elo Rating Engine (`model.py`)**

Every team and player carries a live Elo rating that updates after each game:

- **Team Elo** — overall strength, updated using a Margin of Victory multiplier so blowouts shift ratings more than 1-point wins
- **Defensive Elo** — tracked separately based on points allowed vs. the league average
- **Player Elo** — updated each week using an EPA (Expected Points Added) approximation from box score stats, with soft decay toward 1500 to prevent stale ratings

**Layer 2 — XGBoost Classifier (`processor.py`)**

After simulating the full 2015–present history, the Elo states at the moment before each game are captured as features and fed into an XGBoost model:

| Feature | Description |
|---|---|
| `elo_diff` | Home Elo + HFA − Away Elo |
| `home_elo` / `away_elo` | Raw team ratings |
| `home_def_elo` / `away_def_elo` | Defensive ratings |
| `home_recent_score` / `away_recent_score` | Rolling 5-game scoring average |
| `week` | Week number (captures late-season fatigue) |
| `is_postseason` | Playoff game flag |

The model is trained on an 85/15 chronological split with early stopping, then retrained on the full dataset for live predictions.

**Injury Adjustments**

Before each prediction the system looks up `Out`/`IR`/`Doubtful` players for both teams and calculates a point-spread penalty using Value Over Replacement (VOR):

```
VOR = (Player Elo − Replacement Level) × Position Weight
Penalty = VOR / 15  (converts Elo diff → approx spread points)
```

QB injuries apply a 3× multiplier. The adjusted Elos are then passed into the XGBoost prediction.

---

## Project Structure

```
NFL-Elo-Predictor/
├── main.py                  # Entry point — trains, saves, and runs predictions
├── model.py                 # Team, Player, and NFLModel Elo classes
├── processor.py             # Feature engineering, XGBoost training, prediction
├── data_loader.py           # Loads historical data + live scrape integration
├── scraper.py               # ESPN schedule/stats + CBS injury scrapers
├── backtester.py            # Walk-forward backtesting framework
├── simulate_remainder.py    # Simulates remaining season week-by-week
├── tuner.py                 # Hyperparameter search for Elo parameters
├── debug_data.py            # Debugging utilities
├── data/
│   └── games_combined.csv   # Historical game data (2015–present)
└── requirements.txt
```

---

## Getting Started

**1. Clone and set up the environment**

```bash
git clone https://github.com/Thekid333/NFL-Elo-Predictor.git
cd NFL-Elo-Predictor
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Train and predict**

```bash
python main.py
```

On first run, the system will:
1. Load historical game + player data (2015–present)
2. Run a quick logic verification on 2015 Weeks 1–3
3. Simulate the full history to build Elo ratings
4. Train the XGBoost model
5. Save everything to `nfl_system.pkl`
6. Output a win probability for the configured matchup

On subsequent runs it loads from `nfl_system.pkl` and skips straight to predictions.

**3. Change the matchup**

Edit the bottom of `main.py`:

```python
predict_upcoming(
    model=trained_elo_model,
    ai_model=ai_model,
    home='KC',
    away='BUF',
    season=2025,
    week=22,
    injury_df=injuries
)
```

**4. Simulate the rest of the season**

```bash
python simulate_remainder.py
```

Configure `TARGET_SEASON`, `START_WEEK`, and `SIMULATED_INJURIES` at the top of the file to model "what-if" scenarios (e.g. Mahomes misses Week 14).

---

## Tuned Parameters

| Parameter | Value | Description |
|---|---|---|
| Home Field Advantage | 45 Elo pts | ~2.5 point spread boost |
| Team K-Factor | 16 | How fast team ratings move |
| Player K-Factor | 12 | How fast player ratings move |
| Replacement Level (QB) | 1350 | Floor for backup QB value |
| Replacement Level (Skill) | 1450 | Floor for WR/RB/TE value |

---

## Requirements

- Python 3.9+
- pandas, numpy
- xgboost
- scikit-learn
- requests, lxml

See `requirements.txt` for pinned versions.

---

## Data Sources

- **Historical game data** — bundled in `data/games_combined.csv`
- **Live schedules & player stats** — [ESPN API](https://site.api.espn.com/apis/site/v2/sports/football/nfl/)
- **Injury reports** — [CBS Sports](https://www.cbssports.com/nfl/injuries/)
