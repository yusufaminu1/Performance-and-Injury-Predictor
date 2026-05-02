import os
import sqlite3

import data_fetcher
import cleaner
import db_manager

STATS_COLS = [
    'player_id', 'season', 'team', 'position', 'games_played',
    'minutes_per_game', 'points_per_game', 'assists_per_game',
    'rebounds_per_game', 'blocks_per_game', 'steals_per_game',
    'field_goal_percentage'
]

INJURY_COLS = [
    'player_id', 'season', 'date_of_injury', 'injury_type',
    'required_surgery', 'games_missed'
]

def run():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, 'database', 'nba_predictor.db')
    schema_path = os.path.join(base_dir, 'database', 'schema.sql')

    conn = sqlite3.connect(db_path)

    with open(schema_path, 'r') as f:
        conn.executescript(f.read())

    data_fetcher.fetch_stats()
    data_fetcher.fetch_injuries()

    result = cleaner.cleaner()
    stats = result['player_stats']
    injuries = result['injuries']

    players = stats[['player_id', 'player_name']].drop_duplicates(subset='player_id').copy()
    players[['first_name', 'last_name']] = players['player_name'].str.split(' ', n=1, expand=True)
    players['last_name'] = players['last_name'].fillna('')
    players['birth_date'] = ''
    players = players[['player_id', 'first_name', 'last_name', 'birth_date']]

    stats_insert = stats[[c for c in STATS_COLS if c in stats.columns]]
    injuries_insert = injuries.dropna(subset=['player_id'])[INJURY_COLS]

    db_manager.bulk_insert(conn, 'players', players.to_dict('records'))
    db_manager.bulk_insert(conn, 'player_stats', stats_insert.to_dict('records'))
    db_manager.bulk_insert(conn, 'players_injuries', injuries_insert.to_dict('records'))

    conn.close()
    print('Pipeline complete.')

if __name__ == '__main__':
    run()
