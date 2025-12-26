"""
Model evaluation and comparison script.

Collects metrics from all trained models and generates comparison reports.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Model names to evaluate (directory names are lowercase without separators)
MODEL_NAMES = [
    "randomforest",
    "gradientboosting",
    "logisticregression",
    "svm",
    "decisiontree",
    "knn",
]


def load_model_metrics(model_name: str, base_path: Path) -> dict[str, Any] | None:
    """
    Load metrics for a single model.

    Args:
        model_name: Name of the model
        base_path: Base outputs directory

    Returns:
        Metrics dictionary or None if not found
    """
    metrics_path = base_path / model_name / "metrics.json"

    if not metrics_path.exists():
        logger.warning(f"Metrics not found for {model_name}: {metrics_path}")
        return None

    with open(metrics_path) as f:
        data = json.load(f)

    return {
        "model": model_name,
        **data.get("metrics", {}),
        "run_id": data.get("run_id", ""),
        "timestamp": data.get("timestamp", ""),
    }


def collect_all_metrics(base_path: Path) -> list[dict[str, Any]]:
    """
    Collect metrics from all models.

    Args:
        base_path: Base outputs directory

    Returns:
        List of metrics dictionaries
    """
    all_metrics = []

    for model_name in MODEL_NAMES:
        metrics = load_model_metrics(model_name, base_path)
        if metrics:
            all_metrics.append(metrics)

    logger.info(f"Collected metrics from {len(all_metrics)} models")
    return all_metrics


def find_best_model(
    metrics_list: list[dict[str, Any]], metric: str = "f1_score"
) -> dict[str, Any]:
    """
    Find the best model based on specified metric.

    Args:
        metrics_list: List of metrics dictionaries
        metric: Metric to use for comparison

    Returns:
        Best model information
    """
    if not metrics_list:
        return {"error": "No metrics available"}

    best = max(metrics_list, key=lambda x: x.get(metric, 0))

    return {
        "best_model": best["model"],
        "best_metric_name": metric,
        "best_metric_value": best.get(metric, 0),
        "run_id": best.get("run_id", ""),
        "all_metrics": {m["model"]: m.get(metric, 0) for m in metrics_list},
    }


def create_comparison_table(metrics_list: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Create comparison DataFrame.

    Args:
        metrics_list: List of metrics dictionaries

    Returns:
        Comparison DataFrame
    """
    if not metrics_list:
        return pd.DataFrame()

    df = pd.DataFrame(metrics_list)

    # Reorder columns
    cols = ["model", "accuracy", "precision", "recall", "f1_score"]
    existing_cols = [c for c in cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in cols]
    df = df[existing_cols + other_cols]

    # Sort by f1_score descending
    if "f1_score" in df.columns:
        df = df.sort_values("f1_score", ascending=False)

    return df


def generate_report(
    metrics_list: list[dict[str, Any]], best_model: dict[str, Any]
) -> str:
    """
    Generate text comparison report.

    Args:
        metrics_list: List of metrics dictionaries
        best_model: Best model information

    Returns:
        Report text
    """
    lines = [
        "=" * 70,
        "MODEL COMPARISON REPORT",
        "=" * 70,
        f"Generated: {datetime.now().isoformat()}",
        f"Models evaluated: {len(metrics_list)}",
        "",
        "-" * 70,
        "RESULTS (sorted by F1 Score)",
        "-" * 70,
        "",
        f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}",
        "-" * 70,
    ]

    # Sort by f1_score
    sorted_metrics = sorted(
        metrics_list, key=lambda x: x.get("f1_score", 0), reverse=True
    )

    for m in sorted_metrics:
        lines.append(
            f"{m['model']:<25} "
            f"{m.get('accuracy', 0):>10.4f} "
            f"{m.get('precision', 0):>10.4f} "
            f"{m.get('recall', 0):>10.4f} "
            f"{m.get('f1_score', 0):>10.4f}"
        )

    lines.extend(
        [
            "",
            "-" * 70,
            "BEST MODEL",
            "-" * 70,
            f"Model: {best_model.get('best_model', 'N/A')}",
            f"F1 Score: {best_model.get('best_metric_value', 0):.4f}",
            f"MLflow Run ID: {best_model.get('run_id', 'N/A')}",
            "",
            "=" * 70,
        ]
    )

    return "\n".join(lines)


def save_results(
    metrics_list: list[dict[str, Any]],
    best_model: dict[str, Any],
    comparison_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save evaluation results.

    Args:
        metrics_list: List of metrics dictionaries
        best_model: Best model information
        comparison_df: Comparison DataFrame
        output_path: Output directory
    """
    output_path.mkdir(parents=True, exist_ok=True)

    # Save best model info
    best_model_path = output_path / "best_model.json"
    with open(best_model_path, "w") as f:
        json.dump(
            {
                **best_model,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )
    logger.info(f"Best model info saved to: {best_model_path}")

    # Save comparison CSV
    comparison_csv_path = output_path / "metrics_comparison.csv"
    comparison_df.to_csv(comparison_csv_path, index=False)
    logger.info(f"Comparison CSV saved to: {comparison_csv_path}")

    # Save full metrics
    full_metrics_path = output_path / "all_metrics.json"
    with open(full_metrics_path, "w") as f:
        json.dump(metrics_list, f, indent=2)
    logger.info(f"All metrics saved to: {full_metrics_path}")

    # Save report
    report = generate_report(metrics_list, best_model)
    report_path = output_path / "comparison_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Report saved to: {report_path}")

    # Print report
    print(report)


def main() -> None:
    """Main evaluation function."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting model evaluation")

    # Paths
    base_path = Path("outputs")
    output_path = base_path / "comparison"

    # Collect metrics
    metrics_list = collect_all_metrics(base_path)

    if not metrics_list:
        logger.error("No metrics found. Run training pipelines first.")
        return

    # Find best model
    best_model = find_best_model(metrics_list)

    # Create comparison table
    comparison_df = create_comparison_table(metrics_list)

    # Save results
    save_results(metrics_list, best_model, comparison_df, output_path)

    logger.info("Model evaluation completed")


if __name__ == "__main__":
    main()
