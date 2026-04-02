# Project Task Breakdown

**Project:** NBA Performance & Injury Predictor  
**Group:** Tyler Bertrand, Yusuf Aminu, Kuthab Ibrahim

---

## Tyler — Database & Data Collection

**Goal:** Set up the database and load all raw data into it.

**Tasks:**
1. Create the folder structure for the project
2. Write `database/schema.sql` — define the 4 tables: `players`, `player_stats`, `injuries`, `player_season_features`
3. Write `src/db_manager.py` — functions to connect to SQLite, insert rows, and run queries
4. Write `src/scraper.py` — scrape per-game stats from Basketball Reference (2000–2024)
5. Download the Kaggle injury CSV and place it at `data/raw/nba_injuries.csv`
6. Write `notebooks/01_data_collection.ipynb` — run the scraper, load the injury CSV, insert both into the DB, verify row counts

**Deliverable:** A populated `database/nba_predictor.db` that Kuthab and Yusuf can query.

---

## Kuthab — Data Cleaning & Visualization

**Goal:** Clean both datasets and produce all EDA visualizations.

**Depends on:** Tyler's database being set up first.

**Tasks:**
1. Write `src/cleaner.py` — functions for:
   - Normalizing player names (lowercase, remove accents, strip Jr./III etc.)
   - Standardizing injury dates to season format (e.g. "2015-16")
   - Handling missing values (fill PER/usage_rate with season medians)
   - Deduplicating traded players (keep TOT row or most games played)
2. Write `src/visualizer.py` — reusable functions for all plots (see plots list below)
3. Write `notebooks/02_data_cleaning.ipynb` — show before/after cleaning, validate no nulls in key columns
4. Write `notebooks/04_eda_visualization.ipynb` — produce all 7 plots

**Plots to produce:**
- Workload score distribution (histogram)
- Injury rate by workload quintile (bar chart)
- Correlation heatmap (workload, age, PER, usage rate, injury flag)
- Year-over-year PER change by workload category (boxplot)
- Injury rate by age group and position (grouped bar)
- Workload vs injury rate across NBA eras (line chart)
- Career case studies: LeBron James & Kawhi Leonard

**Deliverable:** Clean data in the DB + all visualizations rendering in the notebook.

---

## Yusuf — Feature Engineering & Machine Learning

**Goal:** Compute derived features and build the injury prediction model.

**Depends on:** Tyler's database + Kuthab's cleaning (can write the code now, run it later).

**Tasks:**
1. Write `src/features.py` — compute:
   - **Workload Score** = `(MPG / season_max_MPG) × (games_played / 82)`
   - **PER Change** = `PER(this season) − PER(previous season)`
   - **Injury Flag** = `1` if games missed ≥ 10, else `0`
   - Write computed features into the `player_season_features` table
2. Write `notebooks/03_feature_engineering.ipynb` — verify features look correct, check distributions
3. Write `notebooks/05_machine_learning.ipynb`:
   - Load features from DB
   - Train/test split (80/20, stratified)
   - Logistic Regression (primary model)
   - Random Forest (comparison)
   - Evaluate: accuracy, precision, recall, F1, ROC-AUC
   - Plot: confusion matrix, ROC curve, feature importances

**Deliverable:** A trained model with evaluation metrics printed in the notebook.

---

## Handoff Order

```
Tyler (DB + data) → Kuthab (cleaning + viz) → Yusuf (features + ML)
```

Yusuf can write and test all code using sample/fake data while waiting for the DB to be ready.
