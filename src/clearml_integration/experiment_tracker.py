"""
ClearML Experiment Tracker Module.

Provides automatic experiment tracking with ClearML including:
- Parameter logging
- Metric logging with plots
- Artifact management
- Comparison dashboards
- Automatic scikit-learn integration
"""

from __future__ import annotations

import functools
import json
import logging
import pickle  # nosec B403
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

if TYPE_CHECKING:
    from clearml import Task as ClearMLTask
    from clearml.logger import Logger as ClearMLLogger

logger = logging.getLogger(__name__)


class ClearMLExperiment:
    """
    ClearML Experiment Tracker with automatic logging support.

    Features:
    - Automatic parameter and metric logging
    - Sklearn model auto-logging
    - Artifact and model versioning
    - Comparison dashboards
    - Offline mode support

    Example:
        with ClearMLExperiment("my_experiment") as exp:
            exp.log_parameters({"n_estimators": 100})
            model.fit(X_train, y_train)
            exp.log_model(model, "random_forest")
            exp.log_metrics({"accuracy": 0.95})
    """

    def __init__(
        self,
        experiment_name: str,
        project_name: str = "EPML-ITMO/Wine-Quality",
        task_type: str = "training",
        tags: list[str] | None = None,
        auto_connect_frameworks: bool = True,
        offline_mode: bool = False,
    ):
        """
        Initialize ClearML experiment.

        Args:
            experiment_name: Name of the experiment
            project_name: ClearML project name
            task_type: Type of task (training, testing, inference, etc.)
            tags: List of tags for the experiment
            auto_connect_frameworks: Enable automatic framework logging
            offline_mode: Run in offline mode (no server connection required)
        """
        self.experiment_name = experiment_name
        self.project_name = project_name
        self.task_type = task_type
        self.tags = tags or []
        self.auto_connect_frameworks = auto_connect_frameworks
        self.offline_mode = offline_mode
        self.task: ClearMLTask | None = None
        self._logger: ClearMLLogger | None = None
        self._start_time: float | None = None

    def __enter__(self) -> ClearMLExperiment:
        """Start the experiment context."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """End the experiment context."""
        success = exc_type is None
        self.end(success=success)

    def start(self) -> None:
        """Start the ClearML experiment."""
        try:
            from clearml import Task

            # Set offline mode if specified
            if self.offline_mode:
                Task.set_offline(offline_mode=True)

            # Map task type string to Task.TaskTypes
            task_types = {
                "training": Task.TaskTypes.training,
                "testing": Task.TaskTypes.testing,
                "inference": Task.TaskTypes.inference,
                "data_processing": Task.TaskTypes.data_processing,
                "qc": Task.TaskTypes.qc,
                "service": Task.TaskTypes.service,
                "optimizer": Task.TaskTypes.optimizer,
                "monitor": Task.TaskTypes.monitor,
                "controller": Task.TaskTypes.controller,
                "application": Task.TaskTypes.application,
                "custom": Task.TaskTypes.custom,
            }

            task_type_enum = task_types.get(self.task_type, Task.TaskTypes.training)

            # Create task
            self.task = Task.init(
                project_name=self.project_name,
                task_name=self.experiment_name,
                task_type=task_type_enum,
                auto_connect_frameworks=self.auto_connect_frameworks,
                reuse_last_task_id=False,
            )

            # Add tags
            if self.tags and self.task:
                self.task.add_tags(self.tags)

            if self.task:
                self._logger = self.task.get_logger()

            self._start_time = time.time()

            logger.info(f"Started ClearML experiment: {self.experiment_name}")
            if self.task:
                logger.info(f"Task ID: {self.task.id}")

        except Exception as e:
            logger.warning(f"Failed to start ClearML experiment: {e}")
            logger.info("Running in local mode without ClearML tracking")
            self.task = None
            self._start_time = time.time()

    def end(self, success: bool = True) -> None:
        """
        End the ClearML experiment.

        Args:
            success: Whether the experiment completed successfully
        """
        if self._start_time:
            duration = time.time() - self._start_time
            self.log_metric("total_duration_seconds", duration)

        if self.task:
            try:
                if not success:
                    self.task.mark_failed()
                self.task.close()
                logger.info(f"Closed ClearML experiment: {self.experiment_name}")
            except Exception as e:
                logger.warning(f"Error closing ClearML task: {e}")

    def log_parameters(self, params: dict[str, Any], prefix: str = "") -> None:
        """
        Log parameters to the experiment.

        Args:
            params: Dictionary of parameters
            prefix: Optional prefix for parameter names
        """
        if self.task:
            try:
                # Add prefix if specified
                if prefix:
                    params = {f"{prefix}/{k}": v for k, v in params.items()}
                self.task.connect(params)
                logger.debug(f"Logged parameters: {list(params.keys())}")
            except Exception as e:
                logger.warning(f"Failed to log parameters: {e}")

    def log_metric(
        self, name: str, value: float, series: str = "metrics", iteration: int = 0
    ) -> None:
        """
        Log a single metric.

        Args:
            name: Metric name
            value: Metric value
            series: Series name for grouping
            iteration: Iteration number
        """
        if self._logger:
            try:
                self._logger.report_scalar(
                    title=series, series=name, value=value, iteration=iteration
                )
                logger.debug(f"Logged metric {name}: {value}")
            except Exception as e:
                logger.warning(f"Failed to log metric {name}: {e}")

    def log_metrics(
        self,
        metrics: dict[str, float],
        series: str = "metrics",
        iteration: int = 0,
    ) -> None:
        """
        Log multiple metrics.

        Args:
            metrics: Dictionary of metrics
            series: Series name for grouping
            iteration: Iteration number
        """
        for name, value in metrics.items():
            self.log_metric(name, value, series=series, iteration=iteration)

    def log_plot(
        self,
        title: str,
        series: str,
        x: list[float] | np.ndarray,
        y: list[float] | np.ndarray,
        xlabel: str = "x",
        ylabel: str = "y",
    ) -> None:
        """
        Log a 2D plot.

        Args:
            title: Plot title
            series: Series name
            x: X values
            y: Y values
            xlabel: X-axis label
            ylabel: Y-axis label
        """
        if self._logger:
            try:
                self._logger.report_line_plot(
                    title=title,
                    series=[series],
                    iteration=0,
                    xaxis=xlabel,
                    yaxis=ylabel,
                    mode="lines+markers",
                )
                logger.debug(f"Logged plot: {title}")
            except Exception as e:
                logger.warning(f"Failed to log plot: {e}")

    def log_confusion_matrix(
        self,
        y_true: np.ndarray | list[Any],
        y_pred: np.ndarray | list[Any],
        labels: list[str] | None = None,
        title: str = "Confusion Matrix",
    ) -> None:
        """
        Log a confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Optional class labels
            title: Plot title
        """
        if self._logger:
            try:
                cm = confusion_matrix(y_true, y_pred)
                self._logger.report_confusion_matrix(
                    title=title,
                    series="Confusion Matrix",
                    matrix=cm,
                    xlabels=labels,
                    ylabels=labels,
                    iteration=0,
                )
                logger.debug("Logged confusion matrix")
            except Exception as e:
                logger.warning(f"Failed to log confusion matrix: {e}")

    def log_classification_report(
        self,
        y_true: np.ndarray | list[Any],
        y_pred: np.ndarray | list[Any],
        target_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Log classification metrics and report.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Optional class names

        Returns:
            Dictionary with classification metrics
        """
        # Calculate metrics
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_weighted": float(
                precision_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "recall_weighted": float(
                recall_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "f1_weighted": float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "precision_macro": float(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "recall_macro": float(
                recall_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "f1_macro": float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            ),
        }

        # Log metrics
        self.log_metrics(metrics, series="classification")

        # Log confusion matrix
        self.log_confusion_matrix(y_true, y_pred, labels=target_names)

        # Log classification report as text
        report_text = classification_report(
            y_true, y_pred, target_names=target_names, zero_division=0
        )
        if self._logger:
            self._logger.report_text(report_text)

        return metrics

    def log_model(
        self,
        model: BaseEstimator,
        model_name: str,
        framework: str = "sklearn",
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """
        Log a trained model.

        Args:
            model: Trained model object
            model_name: Name for the model
            framework: ML framework (sklearn, pytorch, etc.)
            metadata: Optional metadata dictionary

        Returns:
            Model ID if successful, None otherwise
        """
        if self.task:
            try:
                import joblib

                from clearml import OutputModel

                # Create output model
                output_model = OutputModel(task=self.task, framework=framework)

                # Save model to temporary file
                output_dir = Path("outputs/clearml/models")
                output_dir.mkdir(parents=True, exist_ok=True)

                model_path = output_dir / f"{model_name}_{datetime.now():%Y%m%d_%H%M%S}"

                # Use joblib for sklearn models
                if framework == "sklearn":
                    model_file = str(model_path) + ".joblib"
                    joblib.dump(model, model_file)
                else:
                    # Generic pickle
                    model_file = str(model_path) + ".pkl"
                    with open(model_file, "wb") as f:
                        pickle.dump(model, f)  # nosec B301

                # Update model with file
                output_model.update_weights(model_file)

                # Add metadata as labels
                if metadata:
                    output_model.update_labels(metadata)

                logger.info(f"Logged model: {model_name}")
                return str(output_model.id)

            except Exception as e:
                logger.warning(f"Failed to log model: {e}")

        return None

    def log_artifact(
        self,
        name: str,
        artifact: Any,
        artifact_type: str = "data",
    ) -> None:
        """
        Log an artifact (data, file, etc.).

        Args:
            name: Artifact name
            artifact: Artifact object (DataFrame, dict, path, etc.)
            artifact_type: Type hint for the artifact
        """
        if self.task:
            try:
                if isinstance(artifact, pd.DataFrame):
                    self.task.upload_artifact(name=name, artifact_object=artifact)
                elif isinstance(artifact, dict):
                    self.task.upload_artifact(
                        name=name, artifact_object=json.dumps(artifact, indent=2)
                    )
                elif isinstance(artifact, str | Path):
                    self.task.upload_artifact(name=name, artifact_object=str(artifact))
                else:
                    self.task.upload_artifact(name=name, artifact_object=artifact)

                logger.debug(f"Logged artifact: {name}")
            except Exception as e:
                logger.warning(f"Failed to log artifact {name}: {e}")

    def log_dataset(
        self,
        train_df: pd.DataFrame | None = None,
        test_df: pd.DataFrame | None = None,
        name: str = "dataset",
    ) -> None:
        """
        Log training and test datasets.

        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            name: Dataset name prefix
        """
        if train_df is not None:
            self.log_artifact(f"{name}_train", train_df)
            self.log_parameters(
                {
                    f"{name}_train_shape": str(train_df.shape),
                    f"{name}_train_columns": list(train_df.columns),
                },
                prefix="data",
            )

        if test_df is not None:
            self.log_artifact(f"{name}_test", test_df)
            self.log_parameters(
                {
                    f"{name}_test_shape": str(test_df.shape),
                },
                prefix="data",
            )

    def set_comment(self, comment: str) -> None:
        """
        Set experiment comment/description.

        Args:
            comment: Comment text
        """
        if self.task:
            try:
                self.task.set_comment(comment)
            except Exception as e:
                logger.warning(f"Failed to set comment: {e}")

    def get_task_id(self) -> str | None:
        """Get the ClearML task ID."""
        return str(self.task.id) if self.task else None


def clearml_experiment(
    experiment_name: str | None = None,
    project_name: str = "EPML-ITMO/Wine-Quality",
    task_type: str = "training",
    tags: list[str] | None = None,
) -> Callable[..., Any]:
    """
    Decorator to wrap a function in a ClearML experiment.

    Args:
        experiment_name: Name of the experiment (defaults to function name)
        project_name: ClearML project name
        task_type: Type of task
        tags: List of tags

    Returns:
        Decorated function
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            exp_name = experiment_name or func.__name__

            with ClearMLExperiment(
                experiment_name=exp_name,
                project_name=project_name,
                task_type=task_type,
                tags=tags,
            ) as exp:
                # Inject experiment into kwargs if function accepts it
                import inspect

                sig = inspect.signature(func)
                if "clearml_experiment" in sig.parameters:
                    kwargs["clearml_experiment"] = exp

                return func(*args, **kwargs)

        return wrapper

    return decorator


class ExperimentComparison:
    """
    Utility for comparing multiple ClearML experiments.

    Features:
    - Compare metrics across experiments
    - Generate comparison reports
    - Create comparison plots
    """

    def __init__(self, project_name: str = "EPML-ITMO/Wine-Quality"):
        """
        Initialize experiment comparison utility.

        Args:
            project_name: ClearML project name to search
        """
        self.project_name = project_name

    def get_experiments(
        self,
        tags: list[str] | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Get experiments from ClearML.

        Args:
            tags: Filter by tags
            status: Filter by status
            limit: Maximum number of experiments

        Returns:
            List of experiment dictionaries
        """
        try:
            from clearml import Task

            # Get tasks from project
            tasks: list[Any] = Task.get_tasks(
                project_name=self.project_name,
                tags=tags,
                task_filter={"status": [status]} if status else None,
            )

            experiments: list[dict[str, Any]] = []
            for task in tasks[:limit]:
                exp_data: dict[str, Any] = {
                    "id": task.id,
                    "name": task.name,
                    "status": task.status,
                    "created": str(task.data.created),
                    "tags": list(task.get_tags()),
                    "parameters": task.get_parameters(),
                    "metrics": {},
                }

                # Get last metrics
                try:
                    scalars = task.get_last_scalar_metrics()
                    for title, series_dict in scalars.items():
                        for series, value in series_dict.items():
                            exp_data["metrics"][f"{title}/{series}"] = value
                except Exception:  # nosec B110
                    pass

                experiments.append(exp_data)

            return experiments

        except Exception as e:
            logger.warning(f"Failed to get experiments: {e}")
            return []

    def compare_metrics(
        self,
        experiment_ids: list[str] | None = None,
        metric_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Compare metrics across experiments.

        Args:
            experiment_ids: List of experiment IDs to compare
            metric_names: Specific metrics to compare

        Returns:
            DataFrame with comparison results
        """
        experiments = self.get_experiments()

        if experiment_ids:
            experiments = [e for e in experiments if e["id"] in experiment_ids]

        # Build comparison DataFrame
        rows = []
        for exp in experiments:
            row: dict[str, Any] = {
                "experiment_id": exp["id"],
                "experiment_name": exp["name"],
                "status": exp["status"],
            }
            row.update(exp["metrics"])
            rows.append(row)

        df = pd.DataFrame(rows)

        # Filter columns if specific metrics requested
        if metric_names:
            cols = ["experiment_id", "experiment_name", "status"] + [
                c for c in df.columns if any(m in c for m in metric_names)
            ]
            df = df[cols]

        return df

    def generate_report(
        self,
        output_path: str | Path = "outputs/clearml/comparison_report.json",
    ) -> dict[str, Any]:
        """
        Generate a comparison report for all experiments.

        Args:
            output_path: Path to save the report

        Returns:
            Report dictionary
        """
        experiments = self.get_experiments()

        report: dict[str, Any] = {
            "generated_at": datetime.now().isoformat(),
            "project": self.project_name,
            "total_experiments": len(experiments),
            "experiments": experiments,
        }

        # Save report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Comparison report saved to: {output_path}")
        return report
