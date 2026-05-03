import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'injury_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

FEATURE_COLS = [
    'workload_score', 'per_change', 'age', 'games_missed_last_season',
    'points_per_game', 'minutes_per_game'
]

_model = None
_scaler = None


def _load():
    global _model, _scaler
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)


def predict(features: dict) -> dict:
    _load()
    X = np.array([[features[col] for col in FEATURE_COLS]])
    X_scaled = _scaler.transform(X)
    prob = float(_model.predict_proba(X_scaled)[0, 1])
    flag = int(_model.predict(X_scaled)[0])
    label = 'high' if prob >= 0.6 else 'medium' if prob >= 0.4 else 'low'
    return {
        'injury_probability': round(prob, 4),
        'injury_flag': flag,
        'risk_label': label
    }
