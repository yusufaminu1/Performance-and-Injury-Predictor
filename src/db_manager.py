import sqlite3

conn = sqlite3.connect('database/nba_predictor.db')

def bulk_insert(conn, table, rows):
  columns = list(rows[0].keys())
  col_names = ', '.join(columns)
  placeholders_arr = ['?' for _ in columns]
  placeholders = ', '.join(placeholders_arr)
  conn.executemany(
      f"INSERT OR IGNORE INTO {table} ({col_names}) VALUES({placeholders})", [list(row.values()) for row in rows]
  )

with open('database/schema.sql', 'r') as file:
  conn.executescript(file.read())
  bulk_insert(conn, players, rows)





