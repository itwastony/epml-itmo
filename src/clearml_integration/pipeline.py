"""
ClearML Pipeline Module.

Provides ML pipeline orchestration using ClearML Pipelines:
- Automated pipeline creation
- Step-by-step execution
- Monitoring and notifications
- Automatic retries and error handling
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from sklearn.base import ClassifierMixin

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if TYPE_CHECKING:
    from clearml.logger import Logger as ClearMLLogger

logger = logging.getLogger(__name__)


class ClearMLPipeline:
    """
    ClearML Pipeline for ML Workflow Orchestration.

    Features:
    - Multi-step pipeline execution
    - Automatic task creation for each step
    - Progress monitoring
    - Notification support
    - Pipeline versioning

    Example:
        pipeline = ClearMLPipeline("Wine-Quality-Pipeline")
        pipeline.add_data_step(train_path, test_path)
        pipeline.add_training_step("RandomForest", params)
        pipeline.add_evaluation_step()
        results = pipeline.run()
    """

    def __init__(
        self,
        pipeline_name: str,
        project_name: str = "EPML-ITMO/Wine-Quality/Pipelines",
        version: str = "1.0.0",
    ):
        """
        Initialize ClearML Pipeline.

        Args:
            pipeline_name: Name of the pipeline
            project_name: ClearML project name
            version: Pipeline version
        """
        self.pipeline_name = pipeline_name
        self.project_name = project_name
        self.version = version
        self.steps: list[dict[str, Any]] = []
        self.results: dict[str, Any] = {}
        self._pipeline_controller: Any = None
        self._start_time: datetime | None = None

    def add_data_step(
        self,
        train_path: str | Path,
        test_path: str | Path,
        step_name: str = "data_loading",
    ) -> ClearMLPipeline:
        """
        Add data loading step to pipeline.

        Args:
            train_path: Path to training data
            test_path: Path to test data
            step_name: Name of the step

        Returns:
            Self for chaining
        """
        self.steps.append(
            {
                "name": step_name,
                "type": "data",
                "config": {
                    "train_path": str(train_path),
                    "test_path": str(test_path),
                },
            }
        )
        return self

    def add_training_step(
        self,
        model_name: str,
        model_params: dict[str, Any],
        step_name: str | None = None,
    ) -> ClearMLPipeline:
        """
        Add training step to pipeline.

        Args:
            model_name: Name of the model type
            model_params: Model hyperparameters
            step_name: Optional custom step name

        Returns:
            Self for chaining
        """
        self.steps.append(
            {
                "name": step_name or f"train_{model_name.lower()}",
                "type": "training",
                "config": {
                    "model_name": model_name,
                    "params": model_params,
                },
            }
        )
        return self

    def add_evaluation_step(
        self,
        metrics: list[str] | None = None,
        step_name: str = "evaluation",
    ) -> ClearMLPipeline:
        """
        Add evaluation step to pipeline.

        Args:
            metrics: List of metrics to compute
            step_name: Name of the step

        Returns:
            Self for chaining
        """
        self.steps.append(
            {
                "name": step_name,
                "type": "evaluation",
                "config": {
                    "metrics": metrics
                    or ["accuracy", "precision", "recall", "f1_score"],
                },
            }
        )
        return self

    def add_model_registration_step(
        self,
        step_name: str = "model_registration",
    ) -> ClearMLPipeline:
        """
        Add model registration step to pipeline.

        Args:
            step_name: Name of the step

        Returns:
            Self for chaining
        """
        self.steps.append(
            {
                "name": step_name,
                "type": "registration",
                "config": {},
            }
        )
        return self

    def run(
        self,
        local_mode: bool = True,
        queue_name: str = "default",
    ) -> dict[str, Any]:
        """
        Execute the pipeline.

        Args:
            local_mode: Run locally or on ClearML agent
            queue_name: Queue name for remote execution

        Returns:
            Pipeline results
        """
        self._start_time = datetime.now()

        logger.info("=" * 60)
        logger.info(f"Starting Pipeline: {self.pipeline_name}")
        logger.info(f"Version: {self.version}")
        logger.info(f"Steps: {len(self.steps)}")
        logger.info("=" * 60)

        if local_mode:
            return self._run_local()
        else:
            return self._run_remote(queue_name)

    def _run_local(self) -> dict[str, Any]:
        """Run pipeline locally."""
        self.results = {
            "pipeline_name": self.pipeline_name,
            "version": self.version,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "steps": {},
            "success": True,
        }

        # Initialize ClearML task for pipeline
        pipeline_task: Any = None
        pipeline_logger: ClearMLLogger | None = None

        try:
            from clearml import Task

            pipeline_task = Task.init(
                project_name=self.project_name,
                task_name=f"{self.pipeline_name}_v{self.version}",
                task_type=Task.TaskTypes.controller,
                reuse_last_task_id=False,
            )

            pipeline_task.add_tags(["pipeline", f"v{self.version}"])
            pipeline_logger = pipeline_task.get_logger()
        except Exception as e:
            logger.warning(f"ClearML not available: {e}")
            pipeline_task = None
            pipeline_logger = None

        # Track data and model across steps
        train_df: pd.DataFrame | None = None
        test_df: pd.DataFrame | None = None
        model: ClassifierMixin | None = None
        model_name: str = ""
        metrics: dict[str, float] = {}

        # Execute each step
        for i, step in enumerate(self.steps):
            step_name = step["name"]
            step_type = step["type"]
            step_config = step["config"]

            logger.info(f"\n[Step {i + 1}/{len(self.steps)}] {step_name}")
            step_start = datetime.now()

            try:
                if step_type == "data":
                    train_df, test_df = self._execute_data_step(step_config)
                    step_result = {
                        "train_shape": (
                            train_df.shape if train_df is not None else None
                        ),
                        "test_shape": test_df.shape if test_df is not None else None,
                    }

                elif step_type == "training":
                    if train_df is None or test_df is None:
                        raise ValueError("Data not loaded. Add data step first.")
                    model, model_name, train_metrics = self._execute_training_step(
                        train_df, test_df, step_config
                    )
                    metrics.update(train_metrics)
                    step_result = {
                        "model_name": model_name,
                        "metrics": train_metrics,
                    }

                elif step_type == "evaluation":
                    if model is None or test_df is None:
                        raise ValueError("Model not trained. Add training step first.")
                    eval_metrics = self._execute_evaluation_step(
                        model, test_df, step_config
                    )
                    metrics.update(eval_metrics)
                    step_result = {"metrics": eval_metrics}

                elif step_type == "registration":
                    if model is None:
                        raise ValueError(
                            "Model not available. Add training step first."
                        )
                    model_id = self._execute_registration_step(
                        model, model_name, metrics
                    )
                    step_result = {"model_id": model_id}

                else:
                    logger.warning(f"Unknown step type: {step_type}")
                    step_result = {}

                step_duration = (datetime.now() - step_start).total_seconds()
                self.results["steps"][step_name] = {
                    "success": True,
                    "duration": step_duration,
                    "result": step_result,
                }

                # Log to ClearML
                if pipeline_logger:
                    pipeline_logger.report_scalar(
                        title="Pipeline Progress",
                        series="steps_completed",
                        value=i + 1,
                        iteration=0,
                    )

                logger.info(f"  ✓ Completed in {step_duration:.2f}s")

            except Exception as e:
                logger.error(f"  ✗ Step failed: {e}")
                self.results["steps"][step_name] = {
                    "success": False,
                    "error": str(e),
                }
                self.results["success"] = False
                break

        # Finalize
        end_time = datetime.now()
        total_duration = (
            (end_time - self._start_time).total_seconds() if self._start_time else 0
        )
        self.results["end_time"] = end_time.isoformat()
        self.results["total_duration"] = total_duration
        self.results["final_metrics"] = metrics

        # Log final results
        if pipeline_task:
            for metric_name, metric_value in metrics.items():
                pipeline_task.get_logger().report_scalar(
                    title="Final Metrics",
                    series=metric_name,
                    value=metric_value,
                    iteration=0,
                )
            pipeline_task.close()

        # Save results
        self._save_results()

        logger.info("\n" + "=" * 60)
        logger.info(f"Pipeline {'COMPLETED' if self.results['success'] else 'FAILED'}")
        logger.info(f"Total Duration: {total_duration:.2f}s")
        logger.info("=" * 60)

        return self.results

    def _run_remote(self, queue_name: str) -> dict[str, Any]:
        """Run pipeline on ClearML agent."""
        try:
            from clearml import PipelineController

            # Create pipeline controller
            pipe = PipelineController(
                name=self.pipeline_name,
                project=self.project_name,
                version=self.version,
            )

            # Add steps as pipeline components
            # This is a simplified version - full implementation would use
            # @PipelineDecorator.component decorators

            # Start pipeline
            pipe.start(queue=queue_name)

            logger.info(f"Pipeline started on queue: {queue_name}")
            return {"status": "started", "queue": queue_name}

        except Exception as e:
            logger.error(f"Failed to start remote pipeline: {e}")
            return {"status": "failed", "error": str(e)}

    def _execute_data_step(
        self,
        config: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Execute data loading step."""
        train_path = Path(config["train_path"])
        test_path = Path(config["test_path"])

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        logger.info(f"    Loaded train: {train_df.shape}")
        logger.info(f"    Loaded test: {test_df.shape}")

        return train_df, test_df

    def _execute_training_step(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        config: dict[str, Any],
    ) -> tuple[ClassifierMixin, str, dict[str, float]]:
        """Execute model training step."""
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier

        model_name = config["model_name"]
        params = config.get("params", {})

        # Model mapping
        model_classes: dict[str, type] = {
            "RandomForest": RandomForestClassifier,
            "GradientBoosting": GradientBoostingClassifier,
            "LogisticRegression": LogisticRegression,
            "SVM": SVC,
            "DecisionTree": DecisionTreeClassifier,
            "KNN": KNeighborsClassifier,
        }

        if model_name not in model_classes:
            raise ValueError(f"Unknown model: {model_name}")

        # Create and train model
        model_class = model_classes[model_name]
        model: ClassifierMixin = model_class(**params)

        X_train = train_df.iloc[:, :-1]
        y_train = train_df.iloc[:, -1]
        X_test = test_df.iloc[:, :-1]
        y_test = test_df.iloc[:, -1]

        logger.info(f"    Training {model_name}...")
        model.fit(X_train, y_train)

        # Quick evaluation
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred, average="weighted")),
        }

        logger.info(f"    Accuracy: {metrics['accuracy']:.4f}")

        return model, model_name, metrics

    def _execute_evaluation_step(
        self,
        model: ClassifierMixin,
        test_df: pd.DataFrame,
        config: dict[str, Any],
    ) -> dict[str, float]:
        """Execute model evaluation step."""
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        X_test = test_df.iloc[:, :-1]
        y_test = test_df.iloc[:, -1]

        y_pred = model.predict(X_test)

        metrics_to_compute = config.get(
            "metrics", ["accuracy", "precision", "recall", "f1_score"]
        )
        metrics: dict[str, float] = {}

        metric_funcs = {
            "accuracy": lambda: accuracy_score(y_test, y_pred),
            "precision": lambda: precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "recall": lambda: recall_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "f1_score": lambda: f1_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
        }

        for metric_name in metrics_to_compute:
            if metric_name in metric_funcs:
                metrics[metric_name] = float(metric_funcs[metric_name]())
                logger.info(f"    {metric_name}: {metrics[metric_name]:.4f}")

        return metrics

    def _execute_registration_step(
        self,
        model: ClassifierMixin,
        model_name: str,
        metrics: dict[str, float],
    ) -> str:
        """Execute model registration step."""
        from src.clearml_integration.model_manager import ClearMLModelManager

        manager = ClearMLModelManager()
        model_id = manager.register_model(
            model=model,
            model_name=model_name,
            metrics=metrics,
            tags=["pipeline", self.pipeline_name],
        )

        logger.info(f"    Registered model: {model_id}")
        return model_id

    def _save_results(self) -> None:
        """Save pipeline results to file."""
        output_dir = Path("outputs/clearml/pipelines")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"{self.pipeline_name}_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"Results saved to: {output_file}")


def create_wine_quality_pipeline(
    model_name: str = "RandomForest",
    model_params: dict[str, Any] | None = None,
) -> ClearMLPipeline:
    """
    Create a standard Wine Quality classification pipeline.

    Args:
        model_name: Model type to use
        model_params: Model hyperparameters

    Returns:
        Configured ClearMLPipeline
    """
    if model_params is None:
        model_params = {"n_estimators": 100, "random_state": 42}

    pipeline = ClearMLPipeline(
        pipeline_name=f"Wine-Quality-{model_name}",
        version="1.0.0",
    )

    pipeline.add_data_step(
        train_path="data/processed/train.csv",
        test_path="data/processed/test.csv",
    )

    pipeline.add_training_step(
        model_name=model_name,
        model_params=model_params,
    )

    pipeline.add_evaluation_step()

    pipeline.add_model_registration_step()

    return pipeline


def run_all_models_pipeline() -> dict[str, dict[str, Any]]:
    """
    Run pipeline for all available models.

    Returns:
        Dictionary of model results
    """
    models_config: dict[str, dict[str, Any]] = {
        "RandomForest": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
        "GradientBoosting": {
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": 42,
        },
        "LogisticRegression": {"max_iter": 1000, "random_state": 42},
        "SVM": {"kernel": "rbf", "random_state": 42},
        "DecisionTree": {"max_depth": 10, "random_state": 42},
        "KNN": {"n_neighbors": 5},
    }

    results: dict[str, dict[str, Any]] = {}

    for model_name, params in models_config.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Running pipeline for: {model_name}")
        logger.info("=" * 60)

        pipeline = create_wine_quality_pipeline(
            model_name=model_name,
            model_params=params,
        )

        results[model_name] = pipeline.run(local_mode=True)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("ALL PIPELINES COMPLETED")
    logger.info("=" * 60)

    for model_name, result in results.items():
        status = "✓" if result.get("success") else "✗"
        result_metrics = result.get("final_metrics", {})
        acc = result_metrics.get("accuracy", 0)
        logger.info(f"  {status} {model_name}: accuracy={acc:.4f}")

    return results


# CLI entry point
def main() -> None:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="ClearML Pipeline Runner")
    parser.add_argument(
        "--model",
        type=str,
        default="RandomForest",
        help="Model to train",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all models",
    )

    args = parser.parse_args()

    if args.all:
        run_all_models_pipeline()
    else:
        pipeline = create_wine_quality_pipeline(model_name=args.model)
        pipeline.run()


if __name__ == "__main__":
    main()
