# NBA Performance & Injury Predictor

Predicts NBA player injury risk using per-season statistics from 2000–2024. Built with SQLite, scikit-learn, and FastAPI.

**Team:** Tyler Bertrand · Kuthab Ibrahim · Yusuf Aminu

---

## How It Works

1. **Data Collection** — NBA per-game stats are fetched from the NBA API (2000–2024). Injury transaction records come from a Kaggle dataset.
2. **Cleaning** — Player names are normalized, stats are converted to per-game, traded players are deduplicated, and injury records are matched to players by name.
3. **Feature Engineering** — Five features are computed per player per season and written to the database:
   - **Workload Score** — `(MPG / season_max_MPG) × (games_played / season_length)`, normalized 0–1
   - **PER Change** — simplified PER this season minus PER last season
   - **Injury Flag** — 1 if the player missed 10+ games, 0 otherwise
   - **Age Risk Factor** — age binned into under 25, 25–29, 30–33, 34+
   - **Games Missed Last Season** — lag feature from the previous year
4. **Model** — A Random Forest / Logistic Regression classifier trained on those features predicts injury risk. The best model by F1 score is saved.
5. **API** — A FastAPI server loads the trained model and serves predictions over HTTP.

---

## Project Structure

```
Performance-and-Injury-Predictor/
├── data/
│   ├── raw/                  # CSVs from NBA API + Kaggle (one per season)
│   └── processed/            # Cleaned CSVs written by cleaner.py
├── database/
│   ├── schema.sql            # Table definitions
│   └── nba_predictor.db      # SQLite database (auto-created)
├── models/
│   ├── injury_model.pkl      # Trained model
│   ├── scaler.pkl            # Feature scaler
│   └── model_metadata.json   # Training summary
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_eda_visualization.ipynb
│   ├── 05_machine_learning.ipynb
│   └── 06_model_analysis.ipynb
├── outputs/
│   ├── figures/              # Saved chart PNGs
│   └── report.md
├── src/
│   ├── api.py                # FastAPI REST server
│   ├── cleaner.py            # Data cleaning functions
│   ├── data_fetcher.py       # NBA API + Kaggle downloaders
│   ├── db_manager.py         # SQLite utilities
│   ├── features.py           # Feature engineering
│   ├── model_loader.py       # Model deserialization
│   ├── model_trainer.py      # Standalone training script
│   ├── pipeline.py           # End-to-end ETL orchestrator
│   └── visualizer.py         # Reusable plot functions
└── README.md
```

---

## Setup

### 1. Install dependencies

```
py -3 -m pip install pandas numpy scikit-learn matplotlib seaborn fastapi uvicorn joblib nba_api kagglehub notebook requests pydantic
```

### 2. Run the data pipeline

Fetches stats from the NBA API (takes several minutes due to rate limiting), downloads the injury dataset from Kaggle, cleans both, and loads everything into the database.

```
py -3 src/pipeline.py
```

> **Note:** Kaggle requires authentication. Make sure you have a Kaggle account and API token configured before running.

### 3. Build features

Computes all ML features from the database and writes them to the `player_season_features` table.

```
py -3 src/features.py
```

### 4. Train the model

Trains Logistic Regression and Random Forest, picks the best by F1, and saves it to `models/`.

```
py -3 src/model_trainer.py
```

### 5. Start the API

```
py -3 -m uvicorn src.api:app --reload
```

The API will be live at `http://127.0.0.1:8000`.

---

## API Endpoints

### `GET /players`
Returns a paginated list of all players.

```
GET /players?page=1&page_size=50
```

### `GET /players/{player_id}/stats`
Returns full season-by-season stats for a player.

```
GET /players/203076/stats
```

### `GET /players/{player_id}/prediction`
Looks up the player's most recent season features and returns their injury risk prediction.

```
GET /players/203076/prediction
```

**Response:**
```json
{
  "injury_probability": 0.72,
  "injury_flag": 1,
  "risk_label": "high"
}
```

`risk_label` is `"high"` (≥ 0.6), `"medium"` (0.4–0.6), or `"low"` (< 0.4).

### `POST /predict`
Manually submit stats to get a prediction.

```json
{
  "workload_score": 0.75,
  "per_change": -0.12,
  "age": 31,
  "games_missed_last_season": 20,
  "points_per_game": 24.5,
  "minutes_per_game": 34.2
}
```

---

## Interactive Docs

With the server running, open `http://127.0.0.1:8000/docs` in your browser for a full interactive UI to test all endpoints.

---

## Notable Player IDs

| Player | ID |
|---|---|
| LeBron James | 2544 |
| Anthony Davis | 203076 |
| Stephen Curry | 201939 |
| Kevin Durant | 201142 |

Use `GET /players` to search for others.

---

## Model Performance

| Model | F1 Score |
|---|---|
| Random Forest | 0.94 |
| Logistic Regression | 0.94 |

Features used: `workload_score`, `per_change`, `age`, `games_missed_last_season`, `points_per_game`, `minutes_per_game`

**Limitation:** The model evaluates same-season injury risk rather than forecasting future seasons, since all features are drawn from the same season as the injury label.
