import os
import sys
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import model_loader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'database', 'nba_predictor.db')

app = FastAPI(title='NBA Injury Predictor API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


def get_conn():
    return sqlite3.connect(DB_PATH)



class StatsInput(BaseModel):
    workload_score: float
    per_change: float
    age: int
    games_missed_last_season: int
    points_per_game: float
    minutes_per_game: float


class PredictionResponse(BaseModel):
    injury_probability: float
    injury_flag: int
    risk_label: str



@app.post('/predict', response_model=PredictionResponse)
def predict(stats: StatsInput):
    return model_loader.predict(stats.model_dump())


@app.get('/players')
def get_players(page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=200)):
    offset = (page - 1) * page_size
    conn = get_conn()
    rows = conn.execute(
        'SELECT player_id, first_name, last_name FROM players LIMIT ? OFFSET ?',
        (page_size, offset)
    ).fetchall()
    total = conn.execute('SELECT COUNT(*) FROM players').fetchone()[0]
    conn.close()
    return {
        'page': page,
        'page_size': page_size,
        'total': total,
        'players': [{'player_id': r[0], 'first_name': r[1], 'last_name': r[2]} for r in rows]
    }


@app.get('/players/{player_id}/stats')
def get_player_stats(player_id: int):
    conn = get_conn()
    player = conn.execute(
        'SELECT player_id, first_name, last_name FROM players WHERE player_id = ?',
        (player_id,)
    ).fetchone()
    if not player:
        conn.close()
        raise HTTPException(status_code=404, detail='Player not found')

    rows = conn.execute(
        'SELECT * FROM player_stats WHERE player_id = ? ORDER BY season',
        (player_id,)
    ).fetchall()
    cols = [d[0] for d in conn.execute('SELECT * FROM player_stats LIMIT 0').description]
    conn.close()

    return {
        'player_id': player[0],
        'first_name': player[1],
        'last_name': player[2],
        'stats': [dict(zip(cols, r)) for r in rows]
    }


@app.get('/players/{player_id}/prediction', response_model=PredictionResponse)
def get_player_prediction(player_id: int):
    conn = get_conn()
    row = conn.execute(
        """
        SELECT psf.workload_score, psf.per_change, psf.age, psf.games_missed_last_season,
               ps.points_per_game, ps.minutes_per_game
        FROM player_season_features psf
        JOIN player_stats ps ON psf.player_id = ps.player_id AND psf.season = ps.season
        WHERE psf.player_id = ?
        ORDER BY psf.season DESC
        LIMIT 1
        """,
        (player_id,)
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail='No features found for this player')

    features = {
        'workload_score': row[0],
        'per_change': row[1],
        'age': row[2],
        'games_missed_last_season': row[3],
        'points_per_game': row[4],
        'minutes_per_game': row[5],
    }
    return model_loader.predict(features)
