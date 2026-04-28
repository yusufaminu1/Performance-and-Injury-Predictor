import time
import pandas as pd
import os
import kagglehub
from nba_api.stats.static import players
from nba_api.stats.static import teams
from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.endpoints import leaguedashplayerstats
from kagglehub import KaggleDatasetAdapter

nba_players_list = players.get_players()
nba_players = {player['id'] : player for player in nba_players_list}
df_players = pd.DataFrame(nba_players).sort_index(axis = 1)
seasons = []
dfs = []
for index, year in enumerate(range(2000, 2024)):
  seasons.append(f"{year}-{str(year+1)[2:]}")
  stats = leaguedashplayerstats.LeagueDashPlayerStats(season=seasons[index], timeout=120)
  df = stats.get_data_frames()[0]
  df['season'] = seasons[index]
  dfs.append(stats.get_data_frames()[0])
  time.sleep(1)


for i, df in enumerate(dfs):
  if not os.path.exists(f'data/raw/player_stats_{i+2000}.csv'):
    df.to_csv(f'data/raw/player_stats_{i + 2000}.csv', index=False)

if not os.path.exists('data/raw/nba_injuries.csv'):
    path = kagglehub.dataset_download("loganlauton/nba-injury-stats-1951-2023")
    df_injuries = pd.read_csv(f"{path}/NBA Player Injury Stats(1951 - 2023).csv")
    df_injuries = df_injuries[df_injuries['Date'].str[:4].astype(int) >= 2000]
    df_injuries = df_injuries[df_injuries['Date'].str[4] == '-' & df_injuries['Date'].str[7] == '']
    df_injuries.to_csv('data/raw/nba_injuries.csv', index=False, columns=['Date','Team','Acquired','Relinquished','Notes'])
else:
    df_injuries = pd.read_csv('data/raw/nba_injuries.csv')

total_null_acquired = len(df_injuries[df_injuries['Acquired'].isna()])
total_null_relinquished = len(df_injuries[df_injuries['Relinquished'].isna()])

total_rows = len(df_injuries)
null_rate_acquired = (total_null_acquired/total_rows) * 100
null_rate_relinquished = (total_null_relinquished/total_rows) * 100

print({'acquired null rate': null_rate_acquired,  'relinquished null rate': null_rate_relinquished})



