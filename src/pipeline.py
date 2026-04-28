import data_fetcher
import cleaner
import db_manager
import sqlite3

def run():
  if __name__ == '__main__':
    conn = sqlite3.connect('database/nba_predictor.db')

    players = data_fetcher.get_players()
    stats = data_fetcher.fetch_stats()
    injuries = data_fetcher.fetch_injuries()
    cleaner.cleaner()
    db_manager.bulk_insert(conn, 'players', players.to_dict('records'))
    db_manager.bulk_insert(conn, 'player_stats', stats.to_dict('records'))
    db_manager.bulk_insert(conn, 'players_injuries', injuries.to_dict('records'))

