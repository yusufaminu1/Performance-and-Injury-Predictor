import sqlite3

def bulk_insert(conn, table, rows):
  columns = list(rows[0].keys())
  col_names = ', '.join(columns)
  placeholders_arr = ['?' for _ in columns]
  placeholders = ', '.join(placeholders_arr)
  conn.executemany(
      f"INSERT OR IGNORE INTO {table} ({col_names}) VALUES({placeholders})", [list(row.values()) for row in rows]
  )
  conn.commit()

def query_runner(conn, sql, params=()):
  return conn.execute(sql, params)

def schema_migration(conn, table, column_name, data_type='INTEGER'):
  conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_name} {data_type}")
  conn.commit()

if (__name__ == '__main__'):
  conn = sqlite3.connect('database/nba_predictor.db')

  with open('database/schema.sql', 'r') as file:
    conn.executescript(file.read())
    rows = [{"player_id": 1, "first_name": "LeBron", "last_name": "James", "birth_date": "1984-12-30"}]
    bulk_insert(conn, 'players', rows)
    conn.commit()
    cursor = conn.execute("SELECT * FROM players")
    print(cursor.fetchall())


