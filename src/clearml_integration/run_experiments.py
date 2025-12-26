#!/usr/bin/env python3
"""
Run ML experiments with ClearML tracking.

This script provides a unified interface for running experiments
with automatic ClearML tracking, model registration, and comparison.

Usage:
    python -m src.clearml_integration.run_experiments --model RandomForest
    python -m src.clearml_integration.run_experiments --all
    python -m src.clearml_integration.run_experiments --compare
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.clearml_integration.experiment_tracker import (  # noqa: E402
    ClearMLExperiment,
    ExperimentComparison,
)
from src.clearml_integration.model_manager import ClearMLModelManager  # noqa: E402

logger = logging.getLogger(__name__)

# Model configurations
MODELS_CONFIG: dict[str, dict[str, Any]] = {
    "RandomForest": {
        "class": RandomForestClassifier,
        "params": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "random_state": 42,
        },
    },
    "GradientBoosting": {
        "class": GradientBoostingClassifier,
        "params": {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": 42,
        },
    },
    "LogisticRegression": {
        "class": LogisticRegression,
        "params": {
            "max_iter": 1000,
            "random_state": 42,
        },
    },
    "SVM": {
        "class": SVC,
        "params": {
            "kernel": "rbf",
            "C": 1.0,
            "random_state": 42,
        },
    },
    "DecisionTree": {
        "class": DecisionTreeClassifier,
        "params": {
            "max_depth": 10,
            "min_samples_split": 2,
            "random_state": 42,
        },
    },
    "KNN": {
        "class": KNeighborsClassifier,
        "params": {
            "n_neighbors": 5,
            "weights": "uniform",
        },
    },
}


def load_data(
    train_path: str = "data/processed/train.csv",
    test_path: str = "data/processed/test.csv",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load training and test data."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]

    return X_train, y_train, X_test, y_test


def run_single_experiment(
    model_name: str,
    offline_mode: bool = False,
    register_model: bool = True,
) -> dict[str, Any]:
    """
    Run a single model experiment with ClearML tracking.

    Args:
        model_name: Name of the model to train
        offline_mode: Run without ClearML server
        register_model: Whether to register the model

    Returns:
        Experiment results
    """
    if model_name not in MODELS_CONFIG:
        raise ValueError(f"Unknown model: {model_name}")

    config = MODELS_CONFIG[model_name]

    # Create experiment
    with ClearMLExperiment(
        experiment_name=f"{model_name}_Experiment",
        project_name="EPML-ITMO/Wine-Quality/Experiments",
        task_type="training",
        tags=[model_name, "wine-quality", "classification"],
        offline_mode=offline_mode,
    ) as exp:
        # Log experiment description
        exp.set_comment(
            f"""
Wine Quality Classification Experiment
======================================
Model: {model_name}
Dataset: Wine Quality (Red Wine)

This experiment trains a {model_name} model on the wine quality dataset
and logs all metrics, parameters, and the trained model to ClearML.
        """
        )

        # Load data
        logger.info("Loading data...")
        X_train, y_train, X_test, y_test = load_data()

        # Log data info
        exp.log_parameters(
            {
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "features": X_train.shape[1],
                "classes": len(y_train.unique()),
            },
            prefix="data",
        )

        # Log model parameters
        exp.log_parameters(config["params"], prefix="model")

        # Create and train model
        logger.info(f"Training {model_name}...")
        model_class = config["class"]
        model = model_class(**config["params"])
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Log classification metrics
        metrics = exp.log_classification_report(
            y_true=y_test,
            y_pred=y_pred,
            target_names=[str(i) for i in sorted(y_test.unique())],
        )

        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1 Score: {metrics['f1_weighted']:.4f}")

        # Log model
        if exp.task:
            exp.log_model(
                model=model,
                model_name=model_name,
                metadata={"metrics": metrics, "params": config["params"]},
            )

        # Register model with model manager
        if register_model:
            manager = ClearMLModelManager()
            model_id = manager.register_model(
                model=model,
                model_name=model_name,
                metrics=metrics,
                parameters=config["params"],
                tags=["wine-quality", "classification"],
                task_id=exp.get_task_id(),
            )
            logger.info(f"Registered model: {model_id}")

        return {
            "model_name": model_name,
            "task_id": exp.get_task_id(),
            "metrics": metrics,
            "success": True,
        }


def run_all_experiments(
    offline_mode: bool = False,
    register_models: bool = True,
) -> dict[str, Any]:
    """
    Run experiments for all configured models.

    Args:
        offline_mode: Run without ClearML server
        register_models: Whether to register models

    Returns:
        Dictionary of all experiment results
    """
    logger.info("=" * 60)
    logger.info("Running All Model Experiments")
    logger.info("=" * 60)

    results: dict[str, Any] = {}

    for model_name in MODELS_CONFIG:
        logger.info(f"\n{'=' * 40}")
        logger.info(f"Running: {model_name}")
        logger.info("=" * 40)

        try:
            result = run_single_experiment(
                model_name=model_name,
                offline_mode=offline_mode,
                register_model=register_models,
            )
            results[model_name] = result
            logger.info(f"✓ {model_name} completed")

        except Exception as e:
            logger.error(f"✗ {model_name} failed: {e}")
            results[model_name] = {
                "model_name": model_name,
                "success": False,
                "error": str(e),
            }

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)

    for model_name, result in results.items():
        if result.get("success"):
            metrics = result.get("metrics", {})
            acc = metrics.get("accuracy", 0)
            f1 = metrics.get("f1_weighted", 0)
            logger.info(f"  ✓ {model_name}: accuracy={acc:.4f}, f1={f1:.4f}")
        else:
            logger.info(f"  ✗ {model_name}: FAILED - {result.get('error', 'Unknown')}")

    # Find best model
    successful = {
        k: v for k, v in results.items() if v.get("success") and v.get("metrics")
    }
    if successful:
        best_model = max(
            successful.items(),
            key=lambda x: x[1]["metrics"].get("accuracy", 0),
        )
        logger.info(f"\n  Best Model: {best_model[0]}")
        logger.info(f"  Best Accuracy: {best_model[1]['metrics']['accuracy']:.4f}")

    return results


def compare_experiments() -> pd.DataFrame:
    """
    Compare all experiments in the project.

    Returns:
        DataFrame with comparison results
    """
    comparison = ExperimentComparison(project_name="EPML-ITMO/Wine-Quality/Experiments")

    # Get and display comparison
    df = comparison.compare_metrics()

    if not df.empty:
        logger.info("\nExperiment Comparison:")
        logger.info(df.to_string())

        # Generate report
        comparison.generate_report()

    return df


def compare_models() -> pd.DataFrame:
    """
    Compare all registered models.

    Returns:
        DataFrame with comparison results
    """
    manager = ClearMLModelManager()

    # Get comparison
    df = manager.compare_models()

    if not df.empty:
        logger.info("\nModel Comparison:")
        logger.info(df.to_string())

        # Generate report
        manager.generate_model_report()

        # Find best model
        best = manager.get_best_model(metric="accuracy")
        if best:
            logger.info(f"\nBest Model: {best[0]}")
            logger.info(f"Accuracy: {best[1]['metrics'].get('accuracy', 0):.4f}")

    return df


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run ML experiments with ClearML tracking"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODELS_CONFIG.keys()),
        help="Model to train",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all models",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare experiments",
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Compare registered models",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run in offline mode (no ClearML server)",
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Don't register models",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.compare:
        compare_experiments()
    elif args.compare_models:
        compare_models()
    elif args.all:
        run_all_experiments(
            offline_mode=args.offline,
            register_models=not args.no_register,
        )
    elif args.model:
        run_single_experiment(
            model_name=args.model,
            offline_mode=args.offline,
            register_model=not args.no_register,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
