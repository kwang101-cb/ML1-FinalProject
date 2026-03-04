from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


DATASET_PATH = Path("final_dataset/nyc_taxi_ml_dataset_2024_2025.parquet")
EXCLUDED_YEARS = {2002, 2009, 2023}


def _detect_pickup_datetime_column(df: pd.DataFrame) -> str:
	candidates = [
		"pickup_datetime",
		"tpep_pickup_datetime",
		"lpep_pickup_datetime",
		"request_datetime",
	]
	for column in candidates:
		if column in df.columns:
			return column
	raise ValueError(
		"No pickup datetime column was found. "
		"Expected one of: pickup_datetime, tpep_pickup_datetime, "
		"lpep_pickup_datetime, request_datetime."
	)


def _detect_pickup_location_column(df: pd.DataFrame) -> str:
	candidates = ["PULocationID", "pu_location_id", "pickup_location_id"]
	for column in candidates:
		if column in df.columns:
			return column
	raise ValueError(
		"No pickup location column was found. "
		"Expected one of: PULocationID, pu_location_id, pickup_location_id."
	)


def load_dataset(dataset_path: Path | str = DATASET_PATH) -> pd.DataFrame:
	dataset_path = Path(dataset_path)
	if not dataset_path.exists():
		raise FileNotFoundError(f"Dataset not found: {dataset_path}")

	df = pd.read_parquet(dataset_path)

	pickup_datetime_col = _detect_pickup_datetime_column(df)
	pickup_location_col = _detect_pickup_location_column(df)

	df[pickup_datetime_col] = pd.to_datetime(df[pickup_datetime_col], errors="coerce")
	df = df.dropna(subset=[pickup_datetime_col, pickup_location_col]).copy()

	df = df.rename(
		columns={
			pickup_datetime_col: "pickup_datetime",
			pickup_location_col: "PULocationID",
		}
	)

	df["PULocationID"] = pd.to_numeric(df["PULocationID"], errors="coerce")
	df = df.dropna(subset=["PULocationID"]).copy()
	df["PULocationID"] = df["PULocationID"].astype("int64")

	return df


def build_pickup_location_monthly_matrix(df: pd.DataFrame) -> pd.DataFrame:
	matrix = (
		df.assign(
			year=df["pickup_datetime"].dt.year,
			month=df["pickup_datetime"].dt.month,
		)
		.groupby(["PULocationID", "year", "month"], observed=True)
		.size()
		.rename("trip_count")
		.reset_index()
	)

	matrix = matrix.pivot_table(
		index="PULocationID",
		columns=["year", "month"],
		values="trip_count",
		fill_value=0,
	)

	matrix.columns = [f"y{year}_m{month:02d}" for year, month in matrix.columns]
	matrix = matrix.sort_index(axis=1)

	return matrix


def run_initial_clustering(
	monthly_matrix: pd.DataFrame,
	n_clusters: int = 5,
	random_state: int = 42,
) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
	if monthly_matrix.empty:
		raise ValueError("The monthly matrix is empty; clustering cannot be computed.")

	scaler = StandardScaler()
	x_scaled = scaler.fit_transform(monthly_matrix)

	model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
	clusters = model.fit_predict(x_scaled)

	cluster_df = pd.DataFrame(
		{
			"PULocationID": monthly_matrix.index.astype("int64"),
			"cluster": clusters.astype("int64"),
		}
	)

	return cluster_df, model, scaler


def build_cluster_monthly_summary(
	df: pd.DataFrame,
	cluster_df: pd.DataFrame,
) -> pd.DataFrame:
	merged = df.merge(cluster_df, on="PULocationID", how="inner")

	summary = (
		merged.assign(
			year=merged["pickup_datetime"].dt.year,
			month=merged["pickup_datetime"].dt.month,
		)
		.groupby(["year", "month", "cluster"], observed=True)
		.size()
		.rename("trip_count")
		.reset_index()
		.sort_values(["year", "month", "cluster"])
	)

	summary = summary[~summary["year"].isin(EXCLUDED_YEARS)].copy()

	return summary


def plot_monthly_cluster_trends(summary_df: pd.DataFrame) -> None:
	if summary_df.empty:
		raise ValueError("Summary dataframe is empty; nothing to plot.")

	sns.set_theme(style="whitegrid")

	clusters = sorted(summary_df["cluster"].unique())
	plot_df = summary_df.copy()
	plot_df["year_month"] = pd.to_datetime(
		dict(year=plot_df["year"], month=plot_df["month"], day=1)
	)

	plt.figure(figsize=(15, 6))
	for cluster in clusters:
		cluster_data = (
			plot_df[plot_df["cluster"] == cluster]
			.sort_values("year_month")
		)
		plt.plot(
			cluster_data["year_month"],
			cluster_data["trip_count"],
			marker="o",
			linewidth=2,
			label=f"Cluster {cluster}",
		)

	plt.title("Monthly Pickup Trips by Cluster (All Years Combined)")
	plt.xlabel("Year-Month")
	plt.ylabel("Trip count")
	plt.legend(title="Traveler type")

	plt.tight_layout()
	plt.show()


def plot_monthly_cluster_shares(summary_df: pd.DataFrame) -> None:
	if summary_df.empty:
		raise ValueError("Summary dataframe is empty; nothing to plot.")

	sns.set_theme(style="whitegrid")

	shares = summary_df.copy()
	shares["monthly_total"] = shares.groupby(["year", "month"])["trip_count"].transform("sum")
	shares["share"] = shares["trip_count"] / shares["monthly_total"]

	shares["year_month"] = pd.to_datetime(
		dict(year=shares["year"], month=shares["month"], day=1)
	)

	pivot_shares = (
		shares.pivot(index="year_month", columns="cluster", values="share")
		.fillna(0)
		.sort_index()
	)
	pivot_shares.index = pivot_shares.index.strftime("%Y-%m")

	fig, axis = plt.subplots(figsize=(15, 6))
	pivot_shares.plot(kind="bar", stacked=True, width=0.9, ax=axis)
	plt.title("Monthly Cluster Share of Pickup Trips (All Years Combined)")
	plt.xlabel("Year-Month")
	plt.ylabel("Share of monthly trips")
	plt.ylim(0, 1)
	plt.legend(title="Traveler type", bbox_to_anchor=(1.01, 1), loc="upper left")

	plt.tight_layout()
	plt.show()


def run_pickup_clustering_analysis(
	dataset_path: Path | str = DATASET_PATH,
	n_clusters: int = 5,
	random_state: int = 42,
) -> dict[str, pd.DataFrame | KMeans | StandardScaler]:
	df = load_dataset(dataset_path)
	monthly_matrix = build_pickup_location_monthly_matrix(df)
	cluster_df, model, scaler = run_initial_clustering(
		monthly_matrix=monthly_matrix,
		n_clusters=n_clusters,
		random_state=random_state,
	)
	summary_df = build_cluster_monthly_summary(df=df, cluster_df=cluster_df)

	print(f"Loaded rows: {len(df):,}")
	print(f"Unique pickup zones: {df['PULocationID'].nunique():,}")
	print(f"Years in data: {sorted(df['pickup_datetime'].dt.year.unique())}")
	print(f"Clusters: {n_clusters}")

	plot_monthly_cluster_trends(summary_df)
	plot_monthly_cluster_shares(summary_df)

	return {
		"dataset": df,
		"monthly_matrix": monthly_matrix,
		"zone_clusters": cluster_df,
		"monthly_cluster_summary": summary_df,
		"kmeans_model": model,
		"scaler": scaler,
	}


if __name__ == "__main__":
	results = run_pickup_clustering_analysis(
		dataset_path=DATASET_PATH,
		n_clusters=5,
		random_state=42,
	)
