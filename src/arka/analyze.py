import argparse
import sys
from pathlib import Path
import polars as pl
import os
import json
import logging
import numpy as np
import umap
import plotly.express as px
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalyzeCommand:
    def __init__(self, run_id: str, project_root: Path = Path(".")):
        self.run_id = run_id
        self.project_root = project_root
        self.run_dir = project_root / "runs" / self.run_id
        self.stages_dir = self.run_dir / "stages"
        self.report_dir = self.run_dir / "report"

    def _read_data(self) -> tuple[pl.DataFrame | None, pl.DataFrame | None, pl.DataFrame | None]:
        if not self.stages_dir.exists():
            logger.warning(f"Stages directory not found: {self.stages_dir}")
            return None, None, None

        data_frames = []
        dropped_frames = []
        cluster_frames = []

        for stage_dir in sorted(self.stages_dir.iterdir()):
            if not stage_dir.is_dir():
                continue

            stage_name = stage_dir.name

            data_path = stage_dir / "data.parquet"
            if data_path.exists():
                df = pl.read_parquet(data_path)
                if "stage" not in df.columns:
                    df = df.with_columns(pl.lit(stage_name).alias("stage"))
                data_frames.append(df)

            dropped_path = stage_dir / "dropped.parquet"
            if dropped_path.exists():
                df = pl.read_parquet(dropped_path)
                if "stage" not in df.columns:
                    df = df.with_columns(pl.lit(stage_name).alias("stage"))
                dropped_frames.append(df)

            clusters_path = stage_dir / "clusters.parquet"
            if clusters_path.exists():
                df = pl.read_parquet(clusters_path)
                cluster_frames.append(df)

        all_data = pl.concat(data_frames, how="diagonal_relaxed") if data_frames else None
        all_dropped = pl.concat(dropped_frames, how="diagonal_relaxed") if dropped_frames else None
        all_clusters = pl.concat(cluster_frames, how="diagonal_relaxed") if cluster_frames else None

        return all_data, all_dropped, all_clusters

    def run(self):
        logger.info(f"Analyzing run: {self.run_id}")
        data, dropped, clusters = self._read_data()

        if data is None or len(data) == 0:
            logger.error("No data found for analysis.")
            return

        if "embedding" not in data.columns:
            logger.error("No embeddings found in data. Please ensure diversity scoring is run.")
            return

        logger.info(f"Loaded {len(data)} records across stages.")
        if dropped is not None:
            logger.info(f"Loaded {len(dropped)} dropped records.")
        if clusters is not None:
            logger.info(f"Loaded {len(clusters)} clusters.")

        # Prepare embeddings for UMAP
        embeddings = np.stack(data["embedding"].to_numpy())
        logger.info(f"Running UMAP projection on {embeddings.shape[0]} embeddings...")

        reducer = umap.UMAP(
            n_components=2,
            metric="cosine",
            random_state=42, # fixed random_state for reproducibility
            n_neighbors=15,
            min_dist=0.1
        )

        umap_embeddings = reducer.fit_transform(embeddings)
        data = data.with_columns([
            pl.Series("umap_x", umap_embeddings[:, 0]),
            pl.Series("umap_y", umap_embeddings[:, 1])
        ])
        logger.info("UMAP projection completed.")

        self.report_dir.mkdir(parents=True, exist_ok=True)

        # Prepare merged dataframe for plotting
        plot_df = data.to_pandas()

        # Deduplicate records: they persist across stages, so we want only their latest occurrence
        id_col = "id" if "id" in plot_df.columns else "record_id"
        if id_col in plot_df.columns:
            # Drop duplicates by ID, keeping the latest stage entry
            plot_df = plot_df.drop_duplicates(subset=[id_col], keep="last")

        # Merge drop reason if dropped df is provided
        if dropped is not None and "drop_reason" in dropped.columns:
             # Find corresponding IDs and add drop_reason
             if id_col in dropped.columns and "drop_reason" in dropped.columns:
                 drop_pandas = dropped.to_pandas()[[id_col, "drop_reason"]].drop_duplicates(subset=[id_col], keep="last")
                 plot_df = plot_df.merge(drop_pandas, on=id_col, how="left")
                 plot_df["drop_reason"] = plot_df["drop_reason"].fillna("kept")

        # Merge cluster_id if cluster df is provided
        if clusters is not None and "cluster_id" in clusters.columns and "member_ids_json" in clusters.columns:
            try:
                cluster_pandas = clusters.to_pandas()
                # Expand the JSON list of member_ids into individual rows
                member_records = []
                for _, row in cluster_pandas.iterrows():
                    cluster_id = row["cluster_id"]
                    member_ids = json.loads(row["member_ids_json"])
                    for mid in member_ids:
                        member_records.append({id_col: mid, "cluster_id": cluster_id})
                if member_records:
                    cluster_members_df = pl.DataFrame(member_records).to_pandas()
                    plot_df = plot_df.merge(cluster_members_df, on=id_col, how="left")
                    plot_df["cluster_id"] = plot_df["cluster_id"].fillna("unclustered")
            except Exception as e:
                logger.warning(f"Failed to merge cluster data: {e}")

        # Extract quality score if it's a struct/dict or string in scores column
        if "scores" in plot_df.columns:
            def extract_quality(score_val):
                if isinstance(score_val, dict):
                    return score_val.get("quality", None)
                if isinstance(score_val, str):
                    try:
                        scores_dict = json.loads(score_val)
                        return scores_dict.get("quality", None)
                    except json.JSONDecodeError:
                        return None
                return None
            plot_df["quality_score"] = plot_df["scores"].apply(extract_quality)

        # Set default color column
        color_col = "stage" if "stage" in plot_df.columns else None

        # Determine all available hover data and color options
        hover_data = ["stage"]
        if id_col in plot_df.columns:
            hover_data.append(id_col)
        if "drop_reason" in plot_df.columns:
            hover_data.append("drop_reason")
        if "cluster_id" in plot_df.columns:
            hover_data.append("cluster_id")
        if "quality_score" in plot_df.columns:
            hover_data.append("quality_score")

        # Interactive Plotly Map
        fig = px.scatter(
            plot_df,
            x="umap_x",
            y="umap_y",
            color=color_col,
            hover_data=hover_data,
            title=f"UMAP Data Map - Run: {self.run_id}"
        )

        # Add buttons to switch color dimensions
        buttons = []
        color_dimensions = ["stage"]
        if "drop_reason" in plot_df.columns:
            color_dimensions.append("drop_reason")
        if "cluster_id" in plot_df.columns:
            color_dimensions.append("cluster_id")
        if "quality_score" in plot_df.columns:
            color_dimensions.append("quality_score")

        for dim in color_dimensions:
            # We recreate the plot data trace coloring by the selected dim
            dim_fig = px.scatter(plot_df, x="umap_x", y="umap_y", color=dim, hover_data=hover_data)

            buttons.append(dict(
                label=f"Color by {dim}",
                method="restyle",
                args=[
                    {"marker.color": [trace.marker.color for trace in dim_fig.data],
                     "name": [trace.name for trace in dim_fig.data],
                     "marker.colorscale": [trace.marker.colorscale for trace in dim_fig.data if hasattr(trace.marker, "colorscale")] or [None],
                     "marker.showscale": [trace.marker.showscale for trace in dim_fig.data if hasattr(trace.marker, "showscale")] or [None]
                    }
                ]
            ))

        if buttons:
            fig.update_layout(
                updatemenus=[dict(
                    type="buttons",
                    direction="down",
                    buttons=buttons,
                    showactive=True,
                    x=1.1,
                    y=1.0,
                )]
            )
        html_path = self.report_dir / "data_map.html"
        fig.write_html(str(html_path))
        logger.info(f"Saved interactive map to {html_path}")

        # Static Matplotlib Map
        plt.figure(figsize=(10, 8))
        if color_col:
            categories = plot_df[color_col].unique()
            for cat in categories:
                subset = plot_df[plot_df[color_col] == cat]
                plt.scatter(subset["umap_x"], subset["umap_y"], label=str(cat), alpha=0.6, s=10)
            plt.legend()
        else:
            plt.scatter(plot_df["umap_x"], plot_df["umap_y"], alpha=0.6, s=10)

        plt.title(f"UMAP Data Map - Run: {self.run_id}")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        png_path = self.report_dir / "data_map.png"
        plt.savefig(str(png_path), bbox_inches="tight", dpi=300)
        plt.close()
        logger.info(f"Saved static map to {png_path}")

        # Coverage metric calculation
        grid_resolution = 50
        x_min, x_max = plot_df["umap_x"].min(), plot_df["umap_x"].max()
        y_min, y_max = plot_df["umap_y"].min(), plot_df["umap_y"].max()

        # Compute grid coverage
        x_bins = np.linspace(x_min, x_max, grid_resolution + 1)
        y_bins = np.linspace(y_min, y_max, grid_resolution + 1)

        H, _, _ = np.histogram2d(plot_df["umap_x"], plot_df["umap_y"], bins=[x_bins, y_bins])
        covered_cells = np.sum(H > 0)
        coverage_score = float(covered_cells / (grid_resolution ** 2))

        coverage_stats = {
            "run_id": self.run_id,
            "grid_resolution": grid_resolution,
            "covered_cells": int(covered_cells),
            "total_cells": grid_resolution ** 2,
            "coverage_score": coverage_score
        }

        stats_path = self.report_dir / "coverage_stats.json"
        stats_path.write_text(json.dumps(coverage_stats, indent=2))
        logger.info(f"Saved coverage stats to {stats_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="arka.analyze", description="Generate visual data map for a run.")
    parser.add_argument("--run-id", required=True, help="The run ID to analyze.")
    args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])

    cmd = AnalyzeCommand(run_id=args.run_id)
    cmd.run()

if __name__ == "__main__":
    main()
