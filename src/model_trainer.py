import sys
import os
import json
import sqlite3
from datetime import datetime

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.features import build_features, get_connection

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'injury_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
METADATA_PATH = os.path.join(MODELS_DIR, 'model_metadata.json')

FEATURE_COLS = [
    'workload_score', 'per_change', 'age', 'games_missed_last_season',
    'points_per_game', 'minutes_per_game', 'games_played'
]


def load_training_data():
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT psf.player_id, psf.season, psf.workload_score, psf.per_change,
               psf.injury_flag, psf.age, psf.games_missed_last_season,
               ps.points_per_game, ps.minutes_per_game, ps.games_played
        FROM player_season_features psf
        JOIN player_stats ps ON psf.player_id = ps.player_id AND psf.season = ps.season
    """, conn)
    conn.close()
    df['per_change'] = df['per_change'].fillna(0)
    df = df.dropna()
    return df


def train():
    os.makedirs(os.path.abspath(MODELS_DIR), exist_ok=True)

    print("Loading data...")
    df = load_training_data()
    print(f"Rows available: {len(df)}")

    X = df[FEATURE_COLS]
    y = df['injury_flag']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train both models and pick the best by F1
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    }

    best_model = None
    best_name = None
    best_f1 = -1

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        f1 = f1_score(y_test, preds, zero_division=0)
        print(f"{name} F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

    # Save best model and scaler
    joblib.dump(best_model, os.path.abspath(MODEL_PATH))
    joblib.dump(scaler, os.path.abspath(SCALER_PATH))
    print(f"\nSaved best model ({best_name}) to {MODEL_PATH}")

    # Save metadata
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'model_type': best_name,
        'f1_score': round(best_f1, 4),
        'features': FEATURE_COLS,
        'training_rows': len(X_train),
        'test_rows': len(X_test)
    }
    with open(os.path.abspath(METADATA_PATH), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {METADATA_PATH}")
    print(f"\nTraining summary: {metadata}")


if __name__ == '__main__':
    train()
