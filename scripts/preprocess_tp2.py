#!/usr/bin/env python3
"""
TP2 Preprocessing Pipeline for UCDP GEDEvent 25.1

This script prepares a training-ready dataset for the task described in `README.md`:
- Predict a heatmap of deaths by armed actors on a 100x200 global grid, monthly.

Pipeline steps implemented:
1) Load + basic cleaning
2) Missing value handling
3) Outlier handling (winsorization)
4) Feature engineering (monthly grid aggregation, lags, rolling stats)
5) Encoding + scaling
6) Class balance analysis and optional balancing (SMOTE/undersample)
7) Dimensionality reduction (PCA, UMAP)
8) Train/test temporal split and artifact export

Outputs (under `datasets/processed/` by default):
- train.csv, test.csv (feature matrix + label)
- X_train_scaled.parquet, X_test_scaled.parquet (post-scaling)
- y_train.csv, y_test.csv
- metadata.json (config & provenance)
- pca_components.npy, pca_explained_variance.npy, umap_embedding_train.csv, umap_embedding_test.csv

Run:
    python scripts/preprocess_tp2.py \
        --input datasets/GEDEvent_v25_1.csv \
        --outdir datasets/processed \
        --test_start 2022-01-01 \
        --winsor_quantile 0.99 \
        --apply_smote false

Note: By default, this script does NOT perform balancing; set --apply_smote true to create a balanced variant.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# UMAP is optional; guarded import
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

# imbalanced-learn optional
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    HAS_IMBLEARN = True
except Exception:
    HAS_IMBLEARN = False


@dataclass
class Config:
    input: str
    outdir: str
    grid_rows: int = 100
    grid_cols: int = 200
    cell_deg: float = 1.8  # 180 / 100 = 1.8 degrees; 360 / 200 = 1.8
    winsor_quantile: float = 0.99
    test_start: str = "2022-01-01"  # temporal split start date (inclusive) for test
    label_bins: List[int] = None  # boundaries for class labels
    apply_smote: bool = False
    random_state: int = 42

    def __post_init__(self):
        if self.label_bins is None:
            # 4 classes: [0], (0-5], (5-20], (20+]
            # You can modify per your rubric
            self.label_bins = [0, 5, 20]


# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["date_start", "date_end"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Drop exact duplicates
    before = len(df)
    df = df.drop_duplicates()

    # Keep valid lat/lon
    if {"latitude", "longitude"}.issubset(df.columns):
        df = df[(df["latitude"].between(-90, 90)) & (df["longitude"].between(-180, 180))]

    # Parse dates and ensure date_start exists
    df = parse_dates(df)
    if "date_start" not in df.columns:
        raise ValueError("Expected 'date_start' in dataset")

    # Coerce negative deaths to NaN for later handling
    for col in ["best", "deaths_civilians"]:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    # Trim actor/categorical text fields
    for col in ["country", "region", "adm_1", "side_a", "side_b"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    after = len(df)
    print(f"Basic cleaning: dropped {before - after} duplicates/invalid rows")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    # If date_end missing, set to date_start
    if "date_end" in df.columns and "date_start" in df.columns:
        mask = df["date_end"].isna()
        df.loc[mask, "date_end"] = df.loc[mask, "date_start"]

    # Fill missing deaths with 0 (conservative for aggregation)
    for col in ["best", "deaths_civilians"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Drop rows missing critical geolocation or date_start
    required = ["latitude", "longitude", "date_start"]
    missing_required = df[required].isna().any(axis=1)
    dropped = int(missing_required.sum())
    df = df[~missing_required].copy()
    if dropped:
        print(f"Missing handling: dropped {dropped} rows lacking critical fields")

    return df


def winsorize_series(s: pd.Series, q: float) -> pd.Series:
    hi = s.quantile(q)
    return np.minimum(s, hi)


def handle_outliers(df: pd.DataFrame, winsor_q: float) -> pd.DataFrame:
    # Apply winsorization to deaths estimate 'best'
    if "best" in df.columns:
        df["best_w"] = winsorize_series(df["best"], winsor_q)
    else:
        raise ValueError("Expected 'best' column for deaths estimate")
    return df


def month_floor(dt: pd.Series) -> pd.Series:
    return pd.to_datetime(dt.values).to_period("M").dt.to_timestamp()


def grid_index(lat: float, lon: float, rows: int, cols: int) -> Tuple[int, int]:
    # Map lat[-90,90] to rows[0, rows-1] top-to-bottom; lon[-180,180] to cols[0, cols-1]
    r = int(np.floor((lat + 90) / (180 / rows)))
    c = int(np.floor((lon + 180) / (360 / cols)))
    r = max(0, min(rows - 1, r))
    c = max(0, min(cols - 1, c))
    return r, c


def add_grid_cells(df: pd.DataFrame, rows: int, cols: int) -> pd.DataFrame:
    lat_step = 180 / rows
    lon_step = 360 / cols
    # Compute zero-based indices
    r = np.floor((df["latitude"] + 90) / lat_step).astype(int).clip(0, rows - 1)
    c = np.floor((df["longitude"] + 180) / lon_step).astype(int).clip(0, cols - 1)
    df["grid_r"] = r
    df["grid_c"] = c
    df["cell_id"] = df["grid_r"] * cols + df["grid_c"]
    return df


def aggregate_monthly_grid(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    # Month bucket
    df["month"] = month_floor(df["date_start"]) + MonthEnd(0)

    # Assign grid cell
    df = add_grid_cells(df, cfg.grid_rows, cfg.grid_cols)

    agg = (
        df.groupby(["month", "cell_id"])  # aggregation at month-cell level
        .agg(
            deaths=("best_w", "sum"),
            events=("best_w", "count"),
            deaths_civilians=("deaths_civilians", "sum"),
            lat_mean=("latitude", "mean"),
            lon_mean=("longitude", "mean"),
        )
        .reset_index()
    )

    # Fill in missing month-cell combos with zeros (sparse grid)
    months = pd.period_range(
        df["month"].min().to_period("M"), df["month"].max().to_period("M"), freq="M"
    ).to_timestamp() + MonthEnd(0)
    all_cells = pd.Index(np.arange(cfg.grid_rows * cfg.grid_cols), name="cell_id")
    full_idx = pd.MultiIndex.from_product([months, all_cells], names=["month", "cell_id"])

    agg = agg.set_index(["month", "cell_id"]).reindex(full_idx).fillna(0).reset_index()

    return agg


def make_labels(deaths: pd.Series, bins: List[int]) -> pd.Series:
    # 4 classes by default
    # 0: 0
    # 1: 1..bins[0]
    # 2: bins[0]+1 .. bins[1]
    # 3: > bins[1]
    labels = pd.cut(
        deaths,
        bins=[-0.1, 0.0, bins[0], bins[1], np.inf],
        labels=[0, 1, 2, 3],
    ).astype(int)
    return labels


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal features by delegating to lag/rolling feature generator."""
    return add_lags_and_rollups(df)


def add_lags_and_rollups(agg: pd.DataFrame, lags: List[int] = [1, 3, 6]) -> pd.DataFrame:
    # Create lags and rolling features per cell
    agg = agg.sort_values(["cell_id", "month"]).copy()
    grp = agg.groupby("cell_id", group_keys=False)

    for L in lags:
        agg[f"deaths_lag{L}"] = grp["deaths"].shift(L)
        agg[f"events_lag{L}"] = grp["events"].shift(L)

    # Rolling means and sums (3-month window)
    agg["deaths_roll3_mean"] = grp["deaths"].apply(lambda s: s.rolling(3, min_periods=1).mean())
    agg["deaths_roll3_sum"] = grp["deaths"].apply(lambda s: s.rolling(3, min_periods=1).sum())
    agg["events_roll3_mean"] = grp["events"].apply(lambda s: s.rolling(3, min_periods=1).mean())

    # Fill NaNs from lags with 0 (no previous info)
    lag_cols = [c for c in agg.columns if c.startswith("deaths_lag") or c.startswith("events_lag")]
    agg[lag_cols] = agg[lag_cols].fillna(0)

    return agg


def temporal_train_test_split(df: pd.DataFrame, test_start: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_dt = pd.to_datetime(test_start) + MonthEnd(0)
    train = df[df["month"] < split_dt].copy()
    test = df[df["month"] >= split_dt].copy()
    return train, test


def scale_features(train: pd.DataFrame, test: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols])
    X_test = scaler.transform(test[feature_cols])

    X_train = pd.DataFrame(X_train, index=train.index, columns=feature_cols)
    X_test = pd.DataFrame(X_test, index=test.index, columns=feature_cols)

    return X_train, X_test


def run_pca(X_train: pd.DataFrame, X_test: pd.DataFrame, n_components: int = 20, random_state: int = 42):
    pca = PCA(n_components=n_components, random_state=random_state)
    Xtr_p = pca.fit_transform(X_train)
    Xte_p = pca.transform(X_test)
    return pca, Xtr_p, Xte_p


def run_umap(X_train: pd.DataFrame, X_test: pd.DataFrame, n_components: int = 2, random_state: int = 42):
    if not HAS_UMAP:
        print("UMAP not installed; skipping.")
        return None, None, None
    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    Xtr_u = reducer.fit_transform(X_train)
    Xte_u = reducer.transform(X_test)
    return reducer, Xtr_u, Xte_u


def analyze_class_balance(y: pd.Series) -> pd.Series:
    counts = y.value_counts().sort_index()
    ratios = counts / counts.sum()
    print("Class distribution:")
    print(pd.DataFrame({"count": counts, "ratio": ratios}).to_string())
    return counts


def maybe_balance(X: pd.DataFrame, y: pd.Series, method: str = "smote", random_state: int = 42):
    if not HAS_IMBLEARN:
        print("imbalanced-learn not installed; skipping balancing.")
        return X, y
    if method == "smote":
        sm = SMOTE(random_state=random_state)
        Xb, yb = sm.fit_resample(X, y)
        return Xb, yb
    elif method == "undersample":
        rus = RandomUnderSampler(random_state=random_state)
        Xb, yb = rus.fit_resample(X, y)
        return Xb, yb
    else:
        print(f"Unknown balancing method '{method}', skipping.")
        return X, y


def build_features(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    # Aggregate
    agg = aggregate_monthly_grid(df, cfg)

    # Create labels
    agg["label"] = make_labels(agg["deaths"], cfg.label_bins)

    # Add temporal features
    agg = add_lags_and_rollups(agg)

    # Calendar features
    agg["month_num"] = agg["month"].dt.month
    agg["year"] = agg["month"].dt.year

    # Drop any potential infinities
    agg = agg.replace([np.inf, -np.inf], np.nan).fillna(0)

    return agg


def export_artifacts(
    outdir: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    Xtr: pd.DataFrame,
    Xte: pd.DataFrame,
    ytr: pd.Series,
    yte: pd.Series,
    pca_obj: Optional[PCA],
    Xtr_p: Optional[np.ndarray],
    Xte_p: Optional[np.ndarray],
    Xtr_u: Optional[np.ndarray],
    Xte_u: Optional[np.ndarray],
    cfg: Config,
):
    ensure_dir(outdir)

    # Raw splits with labels
    train_df.to_csv(os.path.join(outdir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(outdir, "test.csv"), index=False)

    # Scaled features
    try:
        import pyarrow as pa  # noqa: F401
        Xtr.to_parquet(os.path.join(outdir, "X_train_scaled.parquet"))
        Xte.to_parquet(os.path.join(outdir, "X_test_scaled.parquet"))
    except Exception:
        # Fallback to CSV
        Xtr.to_csv(os.path.join(outdir, "X_train_scaled.csv"), index=False)
        Xte.to_csv(os.path.join(outdir, "X_test_scaled.csv"), index=False)

    ytr.to_csv(os.path.join(outdir, "y_train.csv"), index=False, header=["label"])
    yte.to_csv(os.path.join(outdir, "y_test.csv"), index=False, header=["label"])

    # PCA artifacts
    if pca_obj is not None and Xtr_p is not None:
        np.save(os.path.join(outdir, "pca_components.npy"), pca_obj.components_)
        np.save(os.path.join(outdir, "pca_explained_variance.npy"), pca_obj.explained_variance_ratio_)
        pd.DataFrame(Xtr_p).to_csv(os.path.join(outdir, "pca_train.csv"), index=False)
        pd.DataFrame(Xte_p).to_csv(os.path.join(outdir, "pca_test.csv"), index=False)

    # UMAP embeddings
    if Xtr_u is not None:
        pd.DataFrame(Xtr_u, columns=["umap1", "umap2"]).to_csv(
            os.path.join(outdir, "umap_embedding_train.csv"), index=False
        )
        pd.DataFrame(Xte_u, columns=["umap1", "umap2"]).to_csv(
            os.path.join(outdir, "umap_embedding_test.csv"), index=False
        )

    # Metadata
    with open(os.path.join(outdir, "metadata.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description="TP2 Preprocessing Pipeline")
    parser.add_argument("--input", default="datasets/GEDEvent_v25_1.csv")
    parser.add_argument("--outdir", default="datasets/processed")
    parser.add_argument("--winsor_quantile", type=float, default=0.99)
    parser.add_argument("--test_start", type=str, default="2022-01-01")
    parser.add_argument("--apply_smote", type=str, default="false", help="true|false")
    parser.add_argument("--pca_components", type=int, default=20)
    parser.add_argument("--umap_components", type=int, default=2)
    args = parser.parse_args()

    cfg = Config(
        input=args.input,
        outdir=args.outdir,
        winsor_quantile=args.winsor_quantile,
        test_start=args.test_start,
        apply_smote=args.apply_smote.strip().lower() == "true",
    )

    # Load
    print(f"Loading {cfg.input} ...")
    df = pd.read_csv(cfg.input)

    # Cleaning
    df = basic_cleaning(df)
    df = handle_missing(df)
    df = handle_outliers(df, cfg.winsor_quantile)

    # Feature building
    feat = build_features(cfg, df)

    # Temporal split
    train_df, test_df = temporal_train_test_split(feat, cfg.test_start)

    # Features and target
    feature_cols = [
        c for c in train_df.columns
        if c not in {"label", "month"} and not c.startswith("lat_") and not c.startswith("lon_")
    ]

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    y_train = train_df["label"].astype(int).copy()
    y_test = test_df["label"].astype(int).copy()

    # Analyze class balance
    analyze_class_balance(y_train)

    # Scaling
    X_train_scaled, X_test_scaled = scale_features(train_df, test_df, feature_cols)

    # Optional balancing on training set only
    if cfg.apply_smote:
        X_train_bal, y_train_bal = maybe_balance(X_train_scaled, y_train, method="smote", random_state=cfg.random_state)
        # replace for export
        X_train_scaled = X_train_bal
        y_train = y_train_bal

    # PCA
    pca_obj, Xtr_p, Xte_p = run_pca(X_train_scaled, X_test_scaled, n_components=args.pca_components, random_state=cfg.random_state)
    print("PCA explained variance (first 10):", getattr(pca_obj, "explained_variance_ratio_", [None])[:10])

    # UMAP (2D for visualization)
    reducer, Xtr_u, Xte_u = run_umap(X_train_scaled, X_test_scaled, n_components=args.umap_components, random_state=cfg.random_state)

    # Export
    export_artifacts(
        cfg.outdir,
        train_df,
        test_df,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        pca_obj,
        Xtr_p,
        Xte_p,
        Xtr_u,
        Xte_u,
        cfg,
    )

    print("Done. Artifacts saved to:", cfg.outdir)


if __name__ == "__main__":
    main()
