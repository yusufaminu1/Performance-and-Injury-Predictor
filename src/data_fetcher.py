import time
import pandas as pd
import os
import kagglehub
from nba_api.stats.static import players
from nba_api.stats.static import teams
from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.endpoints import leaguedashplayerstats
from kagglehub import KaggleDatasetAdapter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
os.makedirs(RAW_DIR, exist_ok=True)

def get_players():
  nba_players_list = players.get_players()
  nba_players = {player['id'] : player for player in nba_players_list}
  df_players = pd.DataFrame(nba_players).sort_index(axis = 1)
  return df_players

def fetch_stats():
  seasons = []
  dfs = []
  for index, year in enumerate(range(2000, 2024)):
    seasons.append(f"{year}-{str(year+1)[2:]}")
    stats = leaguedashplayerstats.LeagueDashPlayerStats(season=seasons[index], timeout=120)
    df = stats.get_data_frames()[0]
    df['season'] = seasons[index]
    dfs.append(df)
    time.sleep(1)

  for i, df in enumerate(dfs):
    path = os.path.join(RAW_DIR, f'player_stats_{i+2000}.csv')
    if not os.path.exists(path):
      df.to_csv(path, index=False)

  return pd.concat(dfs, ignore_index=True)

def fetch_injuries():
  injuries_path = os.path.join(RAW_DIR, 'nba_injuries.csv')
  if not os.path.exists(injuries_path):
      path = kagglehub.dataset_download("loganlauton/nba-injury-stats-1951-2023")
      df_injuries = pd.read_csv(f"{path}/NBA Player Injury Stats(1951 - 2023).csv")
      df_injuries = df_injuries[pd.to_datetime(df_injuries['Date'], errors='coerce') >= '2000-01-01']
      df_injuries = df_injuries.dropna(subset=['Date'])
      df_injuries.to_csv(injuries_path, index=False, columns=['Date','Team','Acquired','Relinquished','Notes'])
  else:
      df_injuries = pd.read_csv(injuries_path)

  total_null_acquired = len(df_injuries[df_injuries['Acquired'].isna()])
  total_null_relinquished = len(df_injuries[df_injuries['Relinquished'].isna()])

  total_rows = len(df_injuries)
  null_rate_acquired = (total_null_acquired/total_rows) * 100
  null_rate_relinquished = (total_null_relinquished/total_rows) * 100

  print({'acquired null rate': null_rate_acquired, 'relinquished null rate': null_rate_relinquished})
  return df_injuries



