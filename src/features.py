import sqlite3
import pandas as pd
import os

def get_connection():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    db_path = os.path.join(base_dir,'database','nba_predictor.db')
    return sqlite3.connect(db_path)

def compute_workload_score():
    conn = get_connection()
    df = pd.read_sql_query("""SELECT player_id, season, minutes_per_game, games_played FROM player_stats""",conn)
    conn.close()
    season_max_mpg = df.groupby('season')['minutes_per_game'].transform('max')
    df['workload_score'] = (df['minutes_per_game'] / season_max_mpg) * (df['games_played'] / 82)
    return df[['player_id','season','workload_score']]

def compute_per():
    conn = get_connection()
    df = pd.read_sql_query("""SELECT player_id, season, points_per_game, assists_per_game, rebounds_per_game, blocks_per_game, steals_per_game, minutes_per_game FROM player_stats""",conn)

    conn.close()
    df['per'] = (df['points_per_game']+df['assists_per_game'] +
                df['rebounds_per_game'] + df['blocks_per_game'] +
                df['steals_per_game'] ) / df['minutes_per_game']
    df = df.sort_values(['player_id','season'])
    df['per_change'] = df.groupby('player_id')['per'].diff()
    return df[['player_id','season', 'per', 'per_change']]

def compute_injury_flag():
    conn = get_connection()
    df = pd .read_sql_query("""SELECT player_id, season, SUM(games_missed) as total_games_missed FROM players_injuries GROUP BY player_id, season""",conn)
    conn.close()

    df['injury_flag'] = (df['total_games_missed']>= 10).astype(int)
    return df[['player_id','season','total_games_missed','injury_flag']]

def compute_age_risk_factor():
    conn = get_connection()
    df = pd.read_sql_query("""SELECT ps.player_id, ps.season, p.birth_date FROM player_stats ps JOIN players p on ps.player_id = p.player_id""",conn)
    conn.close()

    df['birth_date'] = pd.to_datetime(df['birth_date'])
    df['season_year'] = df['season'].str[:4].astype(int)
    df['age'] = df['season_year'] - df['birth_date'].dt.year

    bins = [0,24,29,33,100]
    labels = ['under 25','25-29','30-33','34+']
    df['age_risk_factor'] = pd.cut(df['age'],bins=bins,labels=labels)
    return df[['player_id', 'season', 'age', 'age_risk_factor']]
def compute_games_missed_last_season():                                                                
      conn = get_connection()
      df = pd.read_sql_query("""
          SELECT player_id, season, SUM(games_missed) as total_games_missed FROM players_injuries
          GROUP BY player_id, season
      """, conn)

      conn.close()

      df = df.sort_values(['player_id', 'season'])
      df['games_missed_last_season'] = df.groupby('player_id')['total_games_missed'].shift(1)

      return df[['player_id', 'season', 'games_missed_last_season']]

def build_features():
    workload = compute_workload_score()
    per = compute_per()
    injury = compute_injury_flag()
    age = compute_age_risk_factor()
    games_missed = compute_games_missed_last_season()
    df = workload.merge(per[['player_id','season','per_change']],on=['player_id','season'],how='left')
    df = df.merge(injury[['player_id','season','injury_flag']],on=['player_id','season'],how='left')
    df = df.merge(age[['player_id', 'season', 'age', 'age_risk_factor']], on=['player_id', 'season'],how='left')
    df = df.merge(games_missed, on=['player_id', 'season'], how='left')

    df['injury_flag']=df['injury_flag'].fillna(0).astype(int)
    df['games_missed_last_season'] = df['games_missed_last_season'].fillna(0).astype(int)
    df['age_risk_factor'] = df['age_risk_factor'].astype(str)

    conn = get_connection()
    df.to_sql('player_season_features', conn, if_exists='replace', index=False)
    conn.close()

    print(f'Done. {len(df)} rows written to player_season_features.')
    return df

if __name__ == '__main__':                                                                               
    df = build_features()
    print(df.head())  