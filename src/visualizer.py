
import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


FIGURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs",
    "figures",
)

PALETTE = {
    "primary": "#1f4e79",
    "accent": "#e07a1f",
    "muted": "#7f8c8d",
    "good": "#2e7d32",
    "bad": "#c62828",
    "sequence": ["#1f4e79", "#e07a1f", "#2e7d32", "#7d3c98", "#c62828", "#f1c40f"],
}

DEFAULT_FIGSIZE = (10, 6)


def setup_style() -> None:
    """Apply project-wide matplotlib/seaborn styling. Idempotent."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.figsize": DEFAULT_FIGSIZE,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "font.family": "sans-serif",
    })


def _save(fig: plt.Figure, name: str, output_dir: Optional[str] = None) -> str:
    """Save ``fig`` as ``<name>.png`` into the figures dir; return the path."""
    target = output_dir or FIGURES_DIR
    os.makedirs(target, exist_ok=True)
    if not name.lower().endswith(".png"):
        name = f"{name}.png"
    path = os.path.join(target, name)
    fig.savefig(path)
    return path



def plot_workload_distribution(df: pd.DataFrame, workload_col: str = "workload_score",
                               name: str = "workload_distribution",
                               output_dir: Optional[str] = None) -> str:
    setup_style()
    fig, ax = plt.subplots()
    data = df[workload_col].dropna()
    ax.hist(data, bins=40, color=PALETTE["primary"], edgecolor="white", alpha=0.9)
    ax.axvline(data.mean(), color=PALETTE["accent"], linestyle="--", linewidth=2,
               label=f"Mean = {data.mean():.2f}")
    ax.axvline(data.median(), color=PALETTE["good"], linestyle=":", linewidth=2,
               label=f"Median = {data.median():.2f}")
    ax.set_xlabel("Workload Score")
    ax.set_ylabel("Number of Player-Seasons")
    ax.set_title("Distribution of NBA Workload Scores (2000–2024)")
    ax.legend()
    return _save(fig, name, output_dir)


def plot_injury_rate_by_workload_quintile(df: pd.DataFrame,
                                          workload_col: str = "workload_score",
                                          injury_col: str = "injury_flag",
                                          name: str = "injury_rate_by_workload_quintile",
                                          output_dir: Optional[str] = None) -> str:
    setup_style()
    work = df[[workload_col, injury_col]].dropna().copy()
    work["quintile"] = pd.qcut(work[workload_col], q=5,
                               labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"])
    rates = work.groupby("quintile", observed=True)[injury_col].mean() * 100

    fig, ax = plt.subplots()
    bars = ax.bar(rates.index.astype(str), rates.values,
                  color=PALETTE["sequence"][: len(rates)], edgecolor="white")
    ax.set_xlabel("Workload Quintile")
    ax.set_ylabel("Injury Rate (%)")
    ax.set_title("Injury Rate by Workload Quintile")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    for bar, value in zip(bars, rates.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.5, f"{value:.1f}%",
                ha="center", va="bottom", fontsize=10)
    return _save(fig, name, output_dir)


def plot_correlation_heatmap(df: pd.DataFrame,
                             columns: Optional[Iterable[str]] = None,
                             name: str = "correlation_heatmap",
                             output_dir: Optional[str] = None) -> str:
    setup_style()
    default_cols = ["workload_score", "age", "per", "usage_rate", "injury_flag"]
    cols = list(columns) if columns is not None else [c for c in default_cols if c in df.columns]
    data = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    corr = data.corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix")
    return _save(fig, name, output_dir)


def plot_injury_rate_by_age_position(df: pd.DataFrame,
                                     age_col: str = "age_risk_factor",
                                     position_col: str = "position",
                                     injury_col: str = "injury_flag",
                                     name: str = "injury_rate_by_age_position",
                                     output_dir: Optional[str] = None) -> str:
    setup_style()
    work = df[[age_col, position_col, injury_col]].dropna().copy()
    rates = (
        work.groupby([age_col, position_col], observed=True)[injury_col]
        .mean()
        .reset_index()
    )
    rates[injury_col] = rates[injury_col] * 100

    fig, ax = plt.subplots(figsize=(11, 6))
    n_positions = rates[position_col].nunique()
    sns.barplot(
        data=rates, x=age_col, y=injury_col, hue=position_col,
        palette=PALETTE["sequence"][:max(n_positions, 1)], ax=ax,
    )
    ax.set_xlabel("Age Band")
    ax.set_ylabel("Injury Rate (%)")
    ax.set_title("Injury Rate by Age Band and Position")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.legend(title="Position", bbox_to_anchor=(1.02, 1), loc="upper left")
    return _save(fig, name, output_dir)


def plot_per_change_by_workload(df: pd.DataFrame,
                                workload_col: str = "workload_score",
                                per_change_col: str = "per_change",
                                name: str = "per_change_by_workload",
                                output_dir: Optional[str] = None) -> str:
    setup_style()
    work = df[[workload_col, per_change_col]].dropna().copy()
    work["workload_bucket"] = pd.cut(
        work[workload_col],
        bins=[-0.01, 0.2, 0.4, 0.6, 0.8, 1.01],
        labels=["very low", "low", "medium", "high", "very high"],
    )

    fig, ax = plt.subplots()
    n_buckets = work["workload_bucket"].nunique()
    sns.boxplot(
        data=work, x="workload_bucket", y=per_change_col, hue="workload_bucket",
        palette=PALETTE["sequence"][:n_buckets], showfliers=False,
        legend=False, ax=ax,
    )
    ax.axhline(0, color=PALETTE["muted"], linewidth=1, linestyle="--")
    ax.set_xlabel("Workload Category")
    ax.set_ylabel("PER Change vs Previous Season")
    ax.set_title("Year-over-Year PER Change by Workload Category")
    return _save(fig, name, output_dir)


def plot_workload_vs_injury_by_era(df: pd.DataFrame,
                                   workload_col: str = "workload_score",
                                   injury_col: str = "injury_flag",
                                   season_col: str = "season",
                                   name: str = "workload_injury_by_era",
                                   output_dir: Optional[str] = None) -> str:
    setup_style()
    work = df[[season_col, workload_col, injury_col]].dropna().copy()
    work["season_year"] = work[season_col].astype(str).str[:4].astype(int)
    yearly = work.groupby("season_year").agg(
        mean_workload=(workload_col, "mean"),
        injury_rate=(injury_col, "mean"),
    ).reset_index()
    yearly["injury_rate"] *= 100

    fig, ax1 = plt.subplots(figsize=(11, 6))
    color1 = PALETTE["primary"]
    ax1.plot(yearly["season_year"], yearly["mean_workload"], marker="o",
             color=color1, linewidth=2, label="Mean Workload")
    ax1.set_xlabel("Season Start Year")
    ax1.set_ylabel("Mean Workload Score", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    color2 = PALETTE["accent"]
    ax2.plot(yearly["season_year"], yearly["injury_rate"], marker="s",
             color=color2, linewidth=2, label="Injury Rate (%)")
    ax2.set_ylabel("League Injury Rate (%)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))

    fig.suptitle("Workload vs. Injury Rate Across NBA Eras (2000–2024)")
    fig.tight_layout()
    return _save(fig, name, output_dir)



def plot_player_career(df: pd.DataFrame, player_name: str,
                       workload_col: str = "workload_score",
                       injury_col: str = "injury_flag",
                       season_col: str = "season",
                       annotations: Optional[dict] = None,
                       name: Optional[str] = None,
                       output_dir: Optional[str] = None) -> str:
    setup_style()
    work = df[df["player_name_norm"].str.contains(player_name.lower(), na=False)].copy()
    if work.empty:
        raise ValueError(f"No rows found for player matching '{player_name}'")
    work = work.sort_values(season_col)
    work["season_year"] = work[season_col].astype(str).str[:4].astype(int)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(work["season_year"], work[workload_col], marker="o",
            color=PALETTE["primary"], linewidth=2, label="Workload Score")
    ax.fill_between(work["season_year"], 0, work[workload_col],
                    color=PALETTE["primary"], alpha=0.12)

    injured = work[work[injury_col] == 1]
    if not injured.empty:
        ax.scatter(injured["season_year"], injured[workload_col],
                   color=PALETTE["bad"], s=120, zorder=5, label="Injury-flagged season",
                   edgecolor="white", linewidth=1.5)

    if annotations:
        for season_label, text in annotations.items():
            match = work[work[season_col] == season_label]
            if match.empty:
                continue
            x = int(match["season_year"].iloc[0])
            y = float(match[workload_col].iloc[0])
            ax.annotate(
                text,
                xy=(x, y),
                xytext=(x, y + 0.18),
                ha="center",
                fontsize=10,
                color=PALETTE["accent"],
                arrowprops=dict(arrowstyle="->", color=PALETTE["accent"], linewidth=1.5),
            )

    ax.set_ylim(0, max(1.0, work[workload_col].max() * 1.25))
    ax.set_xlabel("Season Start Year")
    ax.set_ylabel("Workload Score")
    ax.set_title(f"Career Arc: {player_name.title()} — Workload & Injury Flags")
    ax.legend(loc="upper right")
    fname = name or f"career_{player_name.lower().replace(' ', '_')}"
    return _save(fig, fname, output_dir)
