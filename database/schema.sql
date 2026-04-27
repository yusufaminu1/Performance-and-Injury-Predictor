CREATE TABLE IF NOT EXISTS players (
  player_id INTEGER PRIMARY KEY NOT NULL,
  first_name TEXT NOT NULL,
  last_name TEXT NOT NULL,
  birth_date TEXT NOT NULL
);

--Player stats is by season of course--
CREATE TABLE IF NOT EXISTS player_stats (
  points_per_game REAL NOT NULL,
  assists_per_game REAL NOT NULL,
  rebounds_per_game REAL NOT NULL,
  blocks_per_game REAL NOT NULL,
  steals_per_game REAL NOT NULL,
  minutes_per_game REAL NOT NULL,
  field_goal_percentage REAL NOT NULL,
  games_played INTEGER NOT NULL,
  team TEXT NOT NULL,
  position TEXT NOT NULL,
  season TEXT NOT NULL,
  player_id INTEGER NOT NULL,
  FOREIGN KEY (player_id) REFERENCES players(player_id),
  PRIMARY KEY(player_id, season)
);

CREATE TABLE IF NOT EXISTS players_injuries (
  injury_id INTEGER PRIMARY KEY NOT NULL,
  player_id INTEGER NOT NULL,
  season TEXT NOT NULL,
  date_of_injury TEXT,
  injury_type TEXT NOT NULL,
  required_surgery INTEGER NOT NULL,
  games_missed INTEGER NOT NULL,
  FOREIGN KEY (player_id) REFERENCES players(player_id)
);

CREATE TABLE IF NOT EXISTS player_season_features (
  games_missed_last_season INTEGER,
  per_change REAL,
  workload_score REAL NOT NULL,
  age INTEGER NOT NULL,
  age_risk_factor TEXT NOT NULL,
  season TEXT NOT NULL,
  player_id INTEGER NOT NULL,
  FOREIGN KEY (player_id) REFERENCES players(player_id),
  PRIMARY KEY(player_id, season)
);
