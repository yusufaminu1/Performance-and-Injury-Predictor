

import os
import re
import glob
import unicodedata
from typing import Optional

import numpy as np
import pandas as pd


RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")



_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_player_name(name: Optional[str]) -> str:
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    text = str(name).strip()
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[\.\,\'\`]", "", text)
    text = re.sub(r"[-_/]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split(" ") if t and t not in _SUFFIXES]
    return " ".join(tokens)



def _date_to_season(date_str: Optional[str]) -> Optional[str]:
    if date_str is None or (isinstance(date_str, float) and np.isnan(date_str)):
        return None
    text = str(date_str).strip()
    if not text:
        return None
    try:
        dt = pd.to_datetime(text, errors="coerce")
    except Exception:
        return None
    if pd.isna(dt):
        return None
    year = dt.year
    if dt.month >= 7:
        start = year
    else:
        start = year - 1
    return f"{start}-{str(start + 1)[2:]}"


def standardize_injury_dates(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out["season"] = out[date_col].apply(lambda d: _date_to_season(d) if pd.notna(d) else None)
    return out




def fill_missing_with_season_median(df: pd.DataFrame, columns, season_col: str = "season") -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        season_median = out.groupby(season_col)[col].transform("median")
        out[col] = out[col].fillna(season_median)
        out[col] = out[col].fillna(out[col].median())
    return out




def dedupe_traded_players(df: pd.DataFrame, player_col: str = "PLAYER_ID", season_col: str = "season",
                          team_col: str = "TEAM_ABBREVIATION", games_col: str = "GP") -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if team_col in out.columns and (out[team_col].astype(str).str.upper() == "TOT").any():
        tot_mask = out[team_col].astype(str).str.upper() == "TOT"
        tot_rows = out[tot_mask]
        non_tot = out[~tot_mask]
        keys = set(map(tuple, tot_rows[[player_col, season_col]].itertuples(index=False, name=None)))
        non_tot = non_tot[~non_tot[[player_col, season_col]].apply(tuple, axis=1).isin(keys)]
        out = pd.concat([tot_rows, non_tot], ignore_index=True)

    out = out.sort_values([player_col, season_col, games_col], ascending=[True, True, False])
    out = out.drop_duplicates(subset=[player_col, season_col], keep="first").reset_index(drop=True)
    return out



def _season_label_from_filename(path: str) -> str:
    m = re.search(r"player_stats_(\d{4})", os.path.basename(path))
    if not m:
        return ""
    year = int(m.group(1))
    return f"{year}-{str(year + 1)[2:]}"


def load_raw_player_stats(raw_dir: str = RAW_DIR) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(raw_dir, "player_stats_*.csv")))
    frames = []
    for f in files:
        df = pd.read_csv(f)
        if "season" not in df.columns or df["season"].isna().all():
            df["season"] = _season_label_from_filename(f)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def clean_player_stats(df: pd.DataFrame) -> pd.DataFrame:
  
    if df.empty:
        return df.copy()
    out = df.copy()

    expected = {"PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "AGE", "GP",
                "PTS", "AST", "REB", "BLK", "STL", "MIN", "FG_PCT", "season"}
    missing = expected - set(out.columns)
    if missing:
        for col in missing:
            out[col] = np.nan

    gp_safe = out["GP"].replace(0, np.nan)
    out["points_per_game"] = out["PTS"] / gp_safe
    out["assists_per_game"] = out["AST"] / gp_safe
    out["rebounds_per_game"] = out["REB"] / gp_safe
    out["blocks_per_game"] = out["BLK"] / gp_safe
    out["steals_per_game"] = out["STL"] / gp_safe
    out["minutes_per_game"] = out["MIN"] / gp_safe
    out["field_goal_percentage"] = out["FG_PCT"]
    out["games_played"] = out["GP"].fillna(0).astype(int)
    out["team"] = out["TEAM_ABBREVIATION"].astype(str)
    out["position"] = out.get("POSITION", pd.Series(["UNK"] * len(out))).fillna("UNK")
    out["player_id"] = out["PLAYER_ID"].astype("Int64")
    out["player_name"] = out["PLAYER_NAME"].astype(str)
    out["player_name_norm"] = out["player_name"].apply(normalize_player_name)
    out["age"] = pd.to_numeric(out["AGE"], errors="coerce")

    out = out.dropna(subset=["player_id"])
    out = out[out["games_played"] > 0]

    out = fill_missing_with_season_median(
        out,
        columns=["field_goal_percentage", "minutes_per_game", "points_per_game"],
        season_col="season",
    )

    # Dedupe traded players.
    out = dedupe_traded_players(out, player_col="player_id", season_col="season",
                                team_col="team", games_col="games_played")

    keep = ["player_id", "player_name", "player_name_norm", "season", "team",
            "position", "age", "games_played", "minutes_per_game",
            "points_per_game", "assists_per_game", "rebounds_per_game",
            "blocks_per_game", "steals_per_game", "field_goal_percentage"]
    return out[keep].reset_index(drop=True)



def _split_relinquished(cell: Optional[str]) -> list:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    text = str(cell).strip().lstrip("•").strip()
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"\s*/\s*", text) if p.strip()]
    return parts


_INJURY_PATTERNS = [
    (r"\b(acl|mcl|pcl|meniscus|knee)\b", "knee"),
    (r"\b(ankle|achilles)\b", "ankle"),
    (r"\b(hamstring|quad(ricep)?|groin|calf|thigh)\b", "leg-soft-tissue"),
    (r"\b(shoulder|rotator cuff)\b", "shoulder"),
    (r"\b(back|spine|lumbar)\b", "back"),
    (r"\b(concussion|head)\b", "head"),
    (r"\b(foot|toe|plantar)\b", "foot"),
    (r"\b(wrist|hand|finger|thumb)\b", "hand"),
    (r"\b(hip)\b", "hip"),
    (r"\b(illness|flu|covid|virus)\b", "illness"),
    (r"\b(rest|load management|coach)\b", "rest"),
    (r"\b(surgery|surgical|operate)\b", "surgery"),
]


def classify_injury(notes: Optional[str]) -> str:
    if notes is None or (isinstance(notes, float) and np.isnan(notes)):
        return "unspecified"
    text = str(notes).lower()
    for pattern, label in _INJURY_PATTERNS:
        if re.search(pattern, text):
            return label
    return "other"


def required_surgery_flag(notes: Optional[str]) -> int:
    if notes is None or (isinstance(notes, float) and np.isnan(notes)):
        return 0
    return int(bool(re.search(r"\bsurg(ery|ical|eries)\b", str(notes).lower())))


def clean_injuries(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["player_id", "player_name_norm", "season",
                                     "date_of_injury", "injury_type",
                                     "required_surgery", "games_missed", "notes"])

    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"])
    out = standardize_injury_dates(out, date_col="Date")

    rel_records = []
    acq_records = []
    for _, row in out.iterrows():
        for name in _split_relinquished(row.get("Relinquished")):
            rel_records.append({
                "date": row["Date"],
                "season": row["season"],
                "player_name_norm": normalize_player_name(name),
                "player_name": name.strip(),
                "team": str(row.get("Team", "")).strip(),
                "notes": row.get("Notes"),
            })
        for name in _split_relinquished(row.get("Acquired")):
            acq_records.append({
                "date": row["Date"],
                "player_name_norm": normalize_player_name(name),
            })

    rel_df = pd.DataFrame(rel_records)
    acq_df = pd.DataFrame(acq_records)
    if rel_df.empty:
        return pd.DataFrame(columns=["player_id", "player_name_norm", "season",
                                     "date_of_injury", "injury_type",
                                     "required_surgery", "games_missed", "notes"])

    rel_df = rel_df.sort_values(["player_name_norm", "date"]).reset_index(drop=True)
    acq_df = acq_df.sort_values(["player_name_norm", "date"]).reset_index(drop=True)

    games_missed = []
    if not acq_df.empty:
        acq_lookup = acq_df.groupby("player_name_norm")["date"].apply(list).to_dict()
    else:
        acq_lookup = {}
    for _, r in rel_df.iterrows():
        returns = acq_lookup.get(r["player_name_norm"], [])
        next_return = next((d for d in returns if d > r["date"]), None)
        if next_return is None:
            games_missed.append(0)
        else:
            days_out = (next_return - r["date"]).days
            games_missed.append(min(82, max(0, int(round(days_out / 2.5)))))
    rel_df["games_missed"] = games_missed

    rel_df["injury_type"] = rel_df["notes"].apply(classify_injury)
    rel_df["required_surgery"] = rel_df["notes"].apply(required_surgery_flag)
    rel_df = rel_df.rename(columns={"date": "date_of_injury"})
    rel_df["date_of_injury"] = rel_df["date_of_injury"].dt.strftime("%Y-%m-%d")
    rel_df["player_id"] = pd.NA 

    keep = ["player_id", "player_name", "player_name_norm", "team", "season",
            "date_of_injury", "injury_type", "required_surgery", "games_missed", "notes"]
    return rel_df[keep].reset_index(drop=True)



def attach_player_ids(injuries_df: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:
    if injuries_df.empty or stats_df.empty:
        return injuries_df.copy()
    name_to_id = (
        stats_df.dropna(subset=["player_id"])
        .groupby("player_name_norm")["player_id"]
        .first()
        .to_dict()
    )
    out = injuries_df.copy()
    out["player_id"] = out["player_name_norm"].map(name_to_id).astype("Int64")
    return out



def cleaner(raw_dir: str = RAW_DIR, processed_dir: str = PROCESSED_DIR,
            write_files: bool = True) -> dict:
    os.makedirs(processed_dir, exist_ok=True)

    raw_stats = load_raw_player_stats(raw_dir)
    cleaned_stats = clean_player_stats(raw_stats)

    injuries_path = os.path.join(raw_dir, "nba_injuries.csv")
    if os.path.exists(injuries_path):
        raw_injuries = pd.read_csv(injuries_path)
        cleaned_injuries = clean_injuries(raw_injuries)
        cleaned_injuries = attach_player_ids(cleaned_injuries, cleaned_stats)
    else:
        cleaned_injuries = pd.DataFrame()

    if write_files:
        cleaned_stats.to_csv(os.path.join(processed_dir, "player_stats_clean.csv"), index=False)
        if not cleaned_injuries.empty:
            cleaned_injuries.to_csv(os.path.join(processed_dir, "injuries_clean.csv"), index=False)

    return {"player_stats": cleaned_stats, "injuries": cleaned_injuries}


if __name__ == "__main__":
    result = cleaner()
    print(f"player_stats: {len(result['player_stats'])} rows")
    print(f"injuries:     {len(result['injuries'])} rows")
