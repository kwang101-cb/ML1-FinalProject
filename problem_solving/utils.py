"""utils.py – shared helpers for the problem_solving notebooks."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
DATA_PATH = Path(__file__).parent.parent / "nyc_taxi_ml_dataset_2024_2025.parquet"
VOLUME_TABLE_PATH = Path(__file__).parent / "data" / "volume_table.parquet"

HOUR_BUCKET_ORDER = ["Early Morning", "Morning Rush", "Midday", "Evening Rush", "Night"]
HOUR_BUCKET_MAP: dict[int, str] = {
    **{h: "Early Morning" for h in range(0, 7)},
    **{h: "Morning Rush" for h in range(7, 10)},
    **{h: "Midday" for h in range(10, 16)},
    **{h: "Evening Rush" for h in range(16, 20)},
    **{h: "Night" for h in range(20, 24)},
}


def assign_hour_bucket(hour: int) -> str:
    return HOUR_BUCKET_MAP.get(int(hour), "Night")



def load_and_validate(
    path: Path | str = DATA_PATH,
    required_cols: list[str] | None = None,
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_parquet(path)
    df = _add_datetime_parts(df, "pickup_datetime", day_col="pickup_day_of_week", hour_col="pickup_hour_of_day")
    df = _add_datetime_parts(df, "dropoff_datetime", day_col="dropoff_day_of_week", hour_col="dropoff_hour_of_day")
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    print(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates derived columns used across all notebooks.
    Assumes the parquet stores numeric types throughout (ints for flags/IDs, floats for amounts).
    No type coercions are performed on existing columns.
    """

    # in_cbd_zone: stored as int in parquet → convert to bool for filtering,
    # keep int version for regression formulas
    df["in_cbd_zone"] = df["in_cbd_zone"].astype(bool)
    df["in_cbd_zone_int"] = df["in_cbd_zone"].astype(int)

    # Derived: hour bucket from pickup_hour_of_day (int 0–23)
    df["hour_bucket"] = df["pickup_hour_of_day"].map(HOUR_BUCKET_MAP)
    df["hour_bucket"] = pd.Categorical(
        df["hour_bucket"], categories=HOUR_BUCKET_ORDER, ordered=True
    )

    # Derived: weekend flag and day bucket from pickup_day_of_week (int 0=Mon … 6=Sun)
    df["is_weekend"] = df["pickup_day_of_week"].isin([5, 6])
    df["day_bucket"] = df["is_weekend"].map({True: "Weekend", False: "Weekday"})

    # Derived: post flag from dataset_split string ('pre' / 'post')
    df["post"] = (df["pickup_datetime"] >= "2025-01-09").astype(int)
    df["period_label"] = df["post"].map({0: "Pre-Policy", 1: "Post-Policy"})

    # Derived: airport trip indicator
    df["is_airport"] = df["airport_fee"] > 0

    return df


def aggregate_volume(
    df: pd.DataFrame,
    groupby_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Aggregate trip counts and mean weather to a bucket-level volume table.
    Default groupby: (post, in_cbd_zone, pickup_hour_of_day, pickup_day_of_week).
    """
    if groupby_cols is None:
        groupby_cols = ["post", "in_cbd_zone", "pickup_hour_of_day", "pickup_day_of_week",
                        "hour_bucket", "day_bucket"]
    agg = (
        df.groupby(groupby_cols, observed=True)
        .agg(
            trip_count=("fare_amount", "count"),
            avg_cbd_fee=("cbd_congestion_fee", "mean"),
            avg_fare=("fare_amount", "mean"),
            avg_temp=("temperature", "mean"),
            avg_precip=("precipitation", "mean"),
            avg_wind=("windspeed", "mean"),
        )
        .reset_index()
    )
    agg["log_trip_count"] = np.log1p(agg["trip_count"])
    agg["in_cbd_zone_int"] = agg["in_cbd_zone"].astype(int)
    agg["period_label"] = agg["post"].map({0: "Pre-Policy", 1: "Post-Policy"})
    agg["zone_label"] = agg["in_cbd_zone"].map({True: "CBD Zone", False: "Non-CBD Zone"})
    return agg


def did_regression(
    df: pd.DataFrame,
    outcome: str,
    extra_controls: list[str] | None = None,
) -> smf.ols:
    """
    OLS DiD: outcome ~ post * in_cbd_zone_int + hour/day FEs + weather + extras.
    df must have 'post' (0/1), 'in_cbd_zone_int' (0/1), 'avg_temp', 'avg_precip', 'avg_wind'.
    Hour and day are included as categoricals if present.
    """
    base_controls = ["avg_temp", "avg_precip", "avg_wind"]
    if "pickup_hour_of_day" in df.columns:
        base_controls.append("C(pickup_hour_of_day)")
    elif "hour_bucket" in df.columns:
        base_controls.append("C(hour_bucket)")
    if "pickup_day_of_week" in df.columns:
        base_controls.append("C(pickup_day_of_week)")
    elif "day_bucket" in df.columns:
        base_controls.append("C(day_bucket)")
    controls = base_controls + (extra_controls or [])
    formula = f"{outcome} ~ post * in_cbd_zone_int + " + " + ".join(controls)
    return smf.ols(formula, data=df).fit(cov_type="HC3")


def pca_od_matrix(
    df: pd.DataFrame,
    n_components: int = 15,
) -> tuple[Pipeline, pd.DataFrame]:
    """
    Build a PULocationID × DOLocationID trip-count OD matrix and fit PCA.

    Corridors are standardized (zero mean, unit variance) before PCA so that
    high-volume routes do not dominate the decomposition purely due to scale.

    Returns (fitted Pipeline[StandardScaler → PCA], full OD matrix).
    Callers can access PCA attributes via pipeline.named_steps['pca'].
    """
    od = (
        df.groupby(["PULocationID", "DOLocationID"])
        .size()
        .unstack(fill_value=0)
    )
    n_comp = min(n_components, min(od.shape) - 1)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_comp, random_state=RANDOM_STATE)),
    ])
    pipeline.fit(od)
    return pipeline, od


def plot_bucket_bars(
    agg_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    title: str,
    figsize: tuple[int, int] = (12, 5),
) -> tuple:
    fig, ax = plt.subplots(figsize=figsize)
    order = HOUR_BUCKET_ORDER if x_col == "hour_bucket" else sorted(agg_df[x_col].unique())
    sns.barplot(data=agg_df, x=x_col, y=y_col, hue=hue_col, ax=ax, order=order)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    return fig, ax
    
def _add_datetime_parts(df, datetime_col, *, day_col="day_of_week", hour_col="hour_of_day"):
    """Add day-of-week and hour-of-day columns from a datetime column."""
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df[day_col] = df[datetime_col].dt.dayofweek
    df[hour_col] = df[datetime_col].dt.hour
    return df
