# Project Task Breakdown

**Project:** NBA Performance & Injury Predictor  
**Group:** Tyler Bertrand, Yusuf Aminu, Kuthab Ibrahim

---

## Tyler — Database, Data Engineering & Backend Infrastructure

**Goal:** Architect the full data pipeline — from raw collection to a clean, query-ready database that the rest of the team builds on.

**Tasks:**
1. Design and create the full project folder structure
2. Write `database/schema.sql` — define normalized tables: `players`, `player_stats`, `injuries`, `player_season_features`, with proper primary keys, foreign keys, and indexes
3. Write `src/db_manager.py` — full database abstraction layer:
   - SQLite connection management (context manager pattern)
   - Bulk insert with conflict resolution (`INSERT OR IGNORE`)
   - Parameterized query runner
   - Schema migration support (versioned `ALTER TABLE` scripts)
4. Write `src/data_fetcher.py` — automated data collection pipeline using `nba_api`:
   - Fetch per-game stats for all players (2000–2024) using `nba_api` (`pip install nba_api`)
   - Implement rate limiting and retry logic to handle API timeouts
   - Normalize API responses into structured DataFrames
   - Export to `data/raw/player_stats.csv` as a checkpoint
5. Download and validate the Kaggle injury CSV (`data/raw/nba_injuries.csv`):
   - Write a validation script that checks column types, null rates, and value ranges before ingestion
6. Write `src/pipeline.py` — end-to-end ETL orchestrator that runs data_fetcher → validate → clean → insert in one command
7. Write `notebooks/01_data_collection.ipynb`:
   - Run the full pipeline, show row counts per table, verify referential integrity with JOIN queries
   - Profile data quality (null %, duplicate %, date range coverage)
8. Write `src/api.py` — FastAPI REST backend that serves the trained model:
   - `POST /predict` — accepts a player's season stats as JSON, loads the serialized model, and returns predicted injury probability + risk label
   - `GET /players` — returns all players in the DB with pagination support
   - `GET /players/{player_id}/stats` — returns full stat history for a given player
   - `GET /players/{player_id}/prediction` — queries DB for latest season features and runs prediction in one call
   - Input validation via Pydantic models (type-safe request/response schemas)
   - CORS middleware enabled so a frontend can connect later
9. Write `src/model_loader.py` — utility that deserializes Yusuf's trained model from `models/injury_model.pkl` (using `joblib`) and exposes a `predict()` function the API calls

**Deliverable:** A fully populated and validated `database/nba_predictor.db` + a running FastAPI server with a `/predict` endpoint that returns real injury risk scores.

---

## Kuthab — Data Cleaning, EDA & Reporting

**Goal:** Clean both datasets, produce EDA visualizations, and write up the project findings in a shareable report.

**Depends on:** Tyler's database being set up first.

**Tasks:**
1. Write `src/cleaner.py` — cleaning functions for:
   - Normalizing player names (lowercase, remove accents, strip Jr./III etc.)
   - Standardizing injury dates to season format (e.g. "2015-16")
   - Handling missing values (fill PER/usage_rate with season medians)
   - Deduplicating traded players (keep TOT row or most games played)
2. Write `src/visualizer.py` — reusable plotting module so all charts share consistent styling (colors, font sizes, axis labels); each function takes a DataFrame and saves a `.png` to `outputs/figures/`
3. Write `notebooks/02_data_cleaning.ipynb` — show before/after cleaning, validate no nulls in key columns, print a summary table of how many rows were dropped/fixed per rule
4. Write `notebooks/04_eda_visualization.ipynb` — produce the following plots using `matplotlib`/`seaborn`:
   - Workload score distribution (histogram)
   - Injury rate by workload quintile (bar chart)
   - Correlation heatmap (workload, age, PER, usage rate, injury flag)
   - Injury rate by age group and position (grouped bar)
   - Year-over-year PER change by workload category (boxplot)
   - Workload vs injury rate across NBA eras 2000–2024 (line chart)
5. Write `outputs/report.md` — a written summary of the project covering:
   - Dataset description and cleaning decisions
   - Key findings from EDA (reference the plots by filename)
   - Limitations and potential sources of bias in the data
6. Conduct two career case studies inside the EDA notebook:
   - **LeBron James** — plot workload score and injury flag across all seasons
   - **Kawhi Leonard** — same, with annotations for the 2017–18 load management season

**Deliverable:** Clean data written back to the DB + all visualizations saved as `.png` files + a completed `report.md`.

---

## Yusuf — Feature Engineering & Machine Learning

**Goal:** Compute derived features, build the injury prediction model, and serialize it for the API.

**Depends on:** Tyler's database + Kuthab's cleaning (can write the code now, run it later).

**Tasks:**
1. Write `src/features.py` — compute:
   - **Workload Score** = `(MPG / season_max_MPG) × (games_played / 82)`
   - **PER Change** = `PER(this season) − PER(previous season)`
   - **Injury Flag** = `1` if games missed ≥ 10, else `0`
   - **Age Risk Factor** = binned age categories (under 25, 25–29, 30–33, 34+)
   - **Games Missed Last Season** = rolling lag feature from the previous year's injury data
   - Write all computed features into the `player_season_features` table
2. Write `notebooks/03_feature_engineering.ipynb` — verify features look correct, check distributions, flag any players with suspicious values
3. Write `notebooks/05_machine_learning.ipynb`:
   - Load features from DB
   - Train/test split (80/20, stratified by injury flag to handle class imbalance)
   - Apply `SMOTE` or class weighting to address imbalanced labels
   - Logistic Regression (primary model)
   - Random Forest (comparison)
   - Gradient Boosting / XGBoost (stretch goal)
   - Evaluate: accuracy, precision, recall, F1, ROC-AUC
   - Plot: confusion matrix, ROC curve, feature importances
4. Write `src/model_trainer.py` — standalone training script (no notebook) that:
   - Pulls features from the DB
   - Trains the best-performing model
   - Serializes it to `models/injury_model.pkl` using `joblib`
   - Prints a training summary (date, model type, F1 score, feature list) to `models/model_metadata.json`
5. Write `notebooks/06_model_analysis.ipynb` — deeper post-hoc analysis:
   - Which positions are most/least accurately predicted?
   - False negative deep dive — who did the model miss and why?
   - Threshold tuning: what cutoff maximizes recall (we'd rather over-predict injury risk than miss one)

**Deliverable:** A trained model serialized to `models/injury_model.pkl` with metadata logged, and evaluation metrics printed in the notebook.

---

## Handoff Order

```
Tyler (DB + pipeline) → Kuthab (cleaning + viz) → Yusuf (features + ML) → Tyler (API wraps the model)
```

Yusuf can write and test all code using sample/fake data while waiting for the DB to be ready. Tyler builds the API last, once Yusuf's model is serialized to `models/injury_model.pkl`.
