# NBA Performance & Injury Predictor — Project Report

**Authors:** Tyler Bertrand, Yusuf Aminu, Kuthab Ibrahim
**Course:** Data Management — Final Project
**Period covered:** NBA seasons 2000–01 through 2023–24

---

## 1. Project Goal

Predict whether an NBA player will lose **≥ 10 games to injury** in an upcoming season, using a single composite *Workload Score* alongside age, efficiency, and prior-injury features. The motivating question: **does heavy minutes load actually drive injury risk, or is the relationship more complicated?**

The system is split into three layers:

1. **Database & ETL** — SQLite schema and an `nba_api`-driven collection pipeline (Tyler).
2. **Cleaning, EDA & visualization** — this report's focus (Kuthab).
3. **Feature engineering & modeling** — workload-based features and a logistic / random-forest classifier (Yusuf).

---

## 2. Dataset Description

### 2.1 Player season stats — `data/raw/player_stats_<year>.csv`

- **Source:** `nba_api.stats.endpoints.LeagueDashPlayerStats` (the same feed that powers nba.com).
- **Coverage:** 24 separate CSVs, one per season from `2000-01` through `2023-24`.
- **Raw size:** roughly 12,400 player-season rows across the 24 files. After cleaning, **11,659 player-seasons** for **2,377 unique players** survive.
- **Key fields:** `PLAYER_ID`, `PLAYER_NAME`, `TEAM_ABBREVIATION`, `AGE`, `GP` (games played), `MIN` (total minutes), `PTS / AST / REB / BLK / STL` (totals), `FG_PCT`.

### 2.2 Player injuries — `data/raw/nba_injuries.csv`

- **Source:** Kaggle dataset `loganlauton/nba-injury-stats-1951-2023`, filtered to dates ≥ `2000-01-01`.
- **Format:** transactions, not events — each row says "Player X *placed on IL*" or "Player X *activated from IL*", with a free-text `Notes` column.
- **Raw size:** ~27,500 transaction rows; after cleaning, **18,950 injury events** with **91.9 % resolved to a `player_id`** via name matching.
- **Quirk:** the `Relinquished` column occasionally packs multiple players into one cell separated by ` / ` (a same-day team-wide IL move) — the cleaner explodes those into one row per player.

---

## 3. Data Cleaning Decisions

All rules live in `src/cleaner.py` and are demonstrated in `notebooks/02_data_cleaning.ipynb`.

| Rule | What it does | Why |
|------|--------------|-----|
| `normalize_player_name` | Lowercase, strip Unicode accents, drop apostrophes/periods, drop Jr./III/IV suffixes | The Kaggle file uses accented names ("Luka Dončić") while the nba.com feed often does not. A normalized key lets us join the two. |
| `standardize_injury_dates` | Parse ISO date → assign NBA season label (`2015-16` style) using Oct–Jun convention | The Kaggle file ships dates, but the analytical unit is the season, not the day. |
| `fill_missing_with_season_median` | Replace null `field_goal_percentage`, `minutes_per_game`, `points_per_game` with the season median | Low-minutes players have noisy/null efficiency stats; median-by-season avoids dropping them and prevents cross-era leakage that a global median would cause. |
| `dedupe_traded_players` | One row per `(player_id, season)` — prefer a `TOT` aggregate row, otherwise the row with the most games played | A traded player otherwise appears 2–3 times in the same season, which would double-count workload. |
| `clean_injuries` | Explode multi-player cells, classify free-text notes into 12 injury types, estimate games missed by pairing each "out" with the next "back" for the same player | The source is transactions; the schema requires events with `games_missed`. |

**Validation after cleaning** (from `02_data_cleaning.ipynb`):

- **0 nulls** in `player_id`, `season`, `team`, `games_played`, `minutes_per_game`, `field_goal_percentage`.
- **0 duplicate** `(player_id, season)` pairs.
- **91.9 %** of injury rows successfully matched to a player_id; the remaining 8.1 % are players outside the 2000–2024 stats window or one-off names that didn't match (e.g., 10-day contract guys).

---

## 4. Feature Engineering

Defined in `src/features.py`; previewed inline in `notebooks/04_eda_visualization.ipynb` so every chart in this report is reproducible from a single run.

- **Workload Score** = `(MPG / season_max_MPG) × (games_played / 82)` — bounded in `[0, 1]`. By construction the league leader in any given season sits near 1.0.
- **PER (proxy)** = `(PTS + AST + REB + BLK + STL) / MPG`. A simple efficiency stand-in for the licensed Hollinger PER.
- **Usage Rate (proxy)** = `PTS / MPG`.
- **PER Change** = year-over-year first difference of PER, grouped by `player_id`.
- **Age Bands** = `under 25`, `25–29`, `30–33`, `34+`.
- **Injury Flag** = `1` if season `total_games_missed ≥ 10`, else `0`. Threshold chosen because (a) it's roughly the cutoff between "healthy with a tweak" and "missed real time" and (b) it produces a workable class balance (≈ 30 / 70).

---

## 5. Key Findings from EDA

All numbers below come directly from `notebooks/04_eda_visualization.ipynb`. Figures referenced are saved in `outputs/figures/`.

### 5.1 Workload distribution — `workload_distribution.png`

The league-wide workload score has **mean 0.362** and **median 0.338**. The distribution is right-skewed: most players cluster in the 0.1–0.5 range (rotation players), with a thin upper tail of ironmen near 0.9–1.0.

### 5.2 Injury rate vs workload — the counterintuitive result — `injury_rate_by_workload_quintile.png`

The naive expectation is "more minutes → more injury." The data says the opposite:

| Workload Quintile | Injury Rate |
|---|---|
| Q1 (lowest)  | **39.3 %** |
| Q2           | 38.8 % |
| Q3           | 33.3 % |
| Q4           | 27.9 % |
| Q5 (highest) | **13.5 %** |

The Pearson correlation between `workload_score` and `injury_flag` is **−0.205**.

This is **selection / survivorship bias**, not a real causal effect. The mechanism is reverse causality: a player who tears an ACL in November 2018 logs few minutes for the rest of 2018–19, which puts him in Q1 *because of* the injury, not before it. Players in Q5 (the ironmen) are durable by definition — that's why they accumulated those minutes in the first place. Treating workload as a one-shot season-end statistic conflates "high availability" with "healthy."

**Implication for the model:** the workload score *as currently defined* will be a poor injury predictor in raw form. The right fix is a **lagged workload feature** (last season's workload predicts this season's injury), which is exactly what `games_missed_last_season` and `per_change` are intended to capture.

### 5.3 Correlation heatmap — `correlation_heatmap.png`

Among the analytical features:

| Feature | corr with `injury_flag` |
|---|---|
| `workload_score` | −0.205 |
| `age`            | −0.018 |
| `per`            |  0.000 |
| `usage_rate`     | −0.014 |

Same takeaway as above — once the contemporaneous workload is removed, almost no signal is left in the *current-season* features. This is a strong argument for the lagged / multi-season features that `features.py` builds (`per_change`, `games_missed_last_season`).

### 5.4 Injury rate by age band & position — `injury_rate_by_age_position.png`

Age does not show the sharp ramp the literature predicts. The `under 25` bucket has the highest injury rate, again because young end-of-bench players miss time for development assignments. The `34+` veterans actually skew low — survivorship again: anyone still in the league at 35 is a *known* iron man.

Position is sparse in the league-dash feed and shows up as `UNK` for the majority of player-seasons. Enriching with a roster file is on the limitations list.

### 5.5 Year-over-year PER change by workload bucket — `per_change_by_workload.png`

The boxplot is roughly symmetric around zero across all five workload buckets. **Workload alone does not predict whether a player gets better or worse year-to-year.** A heavy-minutes season is no more likely to precede a regression than a light-minutes one. This pushes against the popular media narrative that "the league is grinding stars into the ground."

### 5.6 Workload vs injury rate across NBA eras — `workload_injury_by_era.png`

- **Mean workload** drifted from **0.379 (2000–01)** down to **0.335 (2023–24)** — a measurable shift consistent with the load-management era.
- **League injury rate** is essentially flat across the era window when players are at full health; the 2023–24 dip in the chart is a **data artifact**: the Kaggle injury feed ends in 2023, so the second half of the 2023–24 season has no injury rows to match against.

---

## 6. Career Case Studies

### 6.1 LeBron James — `career_lebron_james.png`

LeBron is the cleanest example of "ironman with sustained extreme workload." Across **21 seasons** in the dataset:

- **Mean workload score: 0.82** — i.e., he typically operates near the league ceiling.
- **9 of 21 seasons (43 %)** crossed the 10-games-missed threshold, almost all of them clustered after age 33.
- The annotated turning points (move to Miami in 2010–11, return to Cleveland in 2014–15, first Lakers season groin injury in 2018–19) show the only real workload dips — none of them caused by traditional load-management.

### 6.2 Kawhi Leonard — `career_kawhi_leonard.png`

The opposite philosophy. Across **12 seasons**:

- **Mean workload score: 0.581** — substantially below LeBron despite peak-Kawhi being one of the league's best two-way players.
- **7 of 12 seasons (58 %)** are injury-flagged. Even with deliberate rest scheduling, he misses real time.
- The 2017–18 quad / mystery-injury season (annotated on the chart) is the canonical extreme — only **9 games played**, which the workload formula correctly drives toward zero.
- The 2018–19 Raptors title run (annotated) shows the modern playbook: 60 regular-season games of moderated minutes, leading to a championship.

These two careers neatly bracket the workload spectrum and reinforce the §5.2 finding: heavy workload is not what causes injury, prior injury is what causes light workload.

---

## 7. Modeling Notes

The full modeling work is Yusuf's section (`notebooks/05_machine_learning.ipynb` and `src/model_trainer.py`); the headline implication from the EDA above is:

- **Don't expect raw workload to carry the model.** It encodes "how much the player played" *after* injuries already happened.
- **Lean on lagged / longitudinal features:** `games_missed_last_season`, `per_change`, age band.
- **Class imbalance is roughly 30/70**, not severe, but worth handling with stratified splits (already in the ML notebook) and either class weights or SMOTE.

---

## 8. Limitations & Sources of Bias

1. **Reverse causality in workload (§5.2).** The strongest finding is also the strongest critique of the feature. A truly causal version of this analysis needs game-level minutes data and an injury *event* date so workload can be measured *up to* the moment of injury, not over the whole season.
2. **Position is sparse.** The league-dash feed doesn't ship a position column, and we backfilled `UNK`. The age-vs-position chart is therefore directional, not authoritative.
3. **Injury severity is collapsed into a binary flag.** A torn ACL and a sprained ankle are very different. The cleaner does extract a `injury_type` label and a `required_surgery` flag — those should be used in v2.
4. **Injury data ends in 2023.** The 2023–24 season is partially missing, which depresses recent injury rates and makes the era trend harder to read. A live-data follow-up would close this gap.
5. **PER is a proxy.** The Hollinger PER is a licensed metric; the proxy here `(PTS+AST+REB+BLK+STL)/MIN` is correlated with it but is not the same thing. Conclusions about "efficiency" should be interpreted accordingly.
6. **Name-matching for injuries leaves 8 % unmatched.** Most of these are short-term contract players. They're not random — they're disproportionately rookies and 10-day signings — so the cleaned dataset is slightly biased toward established players.
7. **No playoff data.** The CSVs are regular season only. Playoff workload, especially for stars, can dwarf the regular-season number.

---

## 9. What I'd Improve With More Time

- Replace the workload formula with a **rolling 10-game window** rather than a season aggregate, then re-run §5.2. I expect the relationship to *flip sign* once the leakage is fixed.
- Join a positions roster file (basketball-reference has one) and re-do the age-band-by-position chart.
- Use the `injury_type` and `required_surgery` columns the cleaner already produces — predict severity, not just incidence.
- Pull 2024–25 season injury data to close the era gap.

---

## 10. Reproducibility

Run from the repo root:

```bash
python src/cleaner.py                                  # clean both raw datasets
jupyter nbconvert --to notebook --execute notebooks/02_data_cleaning.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_eda_visualization.ipynb
```

Every figure referenced above is regenerated into `outputs/figures/`.
