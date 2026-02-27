import sys
from pathlib import Path

import mlflow

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))


def get_experiment_results(experiment_name: str = "wine_quality_multimodel_v1") -> None:
    mlflow.set_tracking_uri("file://" + str(Path.cwd() / "mlruns"))
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        print(f"Experiment '{experiment_name}' not found.")
        return

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    # Sort by f1_score descending
    runs = runs.sort_values("metrics.f1_score", ascending=False)

    # Select interesting columns
    cols = [
        "tags.mlflow.runName",
        "params.model_type",
        "metrics.accuracy",
        "metrics.f1_score",
        "metrics.precision",
        "metrics.recall",
    ]

    print("\nTop 5 Runs:")
    print(runs[cols].head(5).to_markdown(index=False))

    print("\nBest Run Details:")
    best_run = runs.iloc[0]
    for col in runs.columns:
        if col.startswith("params.") or col.startswith("metrics."):
            print(f"{col}: {best_run[col]}")


if __name__ == "__main__":
    get_experiment_results()
