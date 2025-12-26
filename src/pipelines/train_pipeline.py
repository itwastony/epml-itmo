"""
Main training pipeline with Hydra configuration management.

This module provides an automated ML training pipeline that integrates:
- Hydra for configuration management
- MLflow for experiment tracking
- DVC for data versioning and pipeline orchestration
"""

import importlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import mlflow
import mlflow.sklearn
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipelines.monitoring import PipelineMonitor  # noqa: E402

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    ML Training Pipeline with Hydra configuration support.

    Handles data loading, model training, evaluation, and MLflow logging.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize training pipeline.

        Args:
            cfg: Hydra configuration object
        """
        self.cfg = cfg
        self.monitor = PipelineMonitor(cfg)
        self.model: ClassifierMixin | None = None
        self.metrics: dict[str, float] = {}

        # Setup logging
        log_level = getattr(logging, cfg.logging.level, logging.INFO)
        logging.basicConfig(level=log_level, format=cfg.logging.format)

        logger.info("Training pipeline initialized")
        logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    def load_data(self) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Load training and test data.

        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        self.monitor.start_stage("data_loading")

        try:
            # Get original working directory (before Hydra changes it)
            original_cwd = hydra.utils.get_original_cwd()
            processed_path = self.cfg.data.get("processed_path", "data/processed")
            data_path = Path(original_cwd) / processed_path

            train_file = self.cfg.data.get("train_file", "train.csv")
            test_file = self.cfg.data.get("test_file", "test.csv")
            train_path = data_path / train_file
            test_path = data_path / test_file

            if not train_path.exists() or not test_path.exists():
                raise FileNotFoundError(
                    f"Data not found. Run 'dvc repro prepare' first. "
                    f"Looking in: {data_path}"
                )

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            X_train = train_df.iloc[:, :-1]
            y_train = train_df.iloc[:, -1]
            X_test = test_df.iloc[:, :-1]
            y_test = test_df.iloc[:, -1]

            logger.info(f"Loaded training data: {X_train.shape}")
            logger.info(f"Loaded test data: {X_test.shape}")

            self.monitor.end_stage("data_loading", success=True)
            return X_train, y_train, X_test, y_test

        except Exception as e:
            self.monitor.end_stage("data_loading", success=False, error=str(e))
            raise

    def create_model(self) -> ClassifierMixin:
        """
        Create model instance from configuration.

        Returns:
            Initialized model instance
        """
        self.monitor.start_stage("model_creation")

        try:
            # Parse model class path
            target = self.cfg.model._target_
            module_path, class_name = target.rsplit(".", 1)

            # Import and instantiate model
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)

            # Get model parameters
            params = OmegaConf.to_container(self.cfg.model.params, resolve=True)

            # Create model instance
            self.model = model_class(**params)

            logger.info(f"Created model: {self.cfg.model.name}")
            logger.info(f"Parameters: {params}")

            self.monitor.end_stage("model_creation", success=True)
            return self.model

        except Exception as e:
            self.monitor.end_stage("model_creation", success=False, error=str(e))
            raise

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, float]:
        """
        Train model and evaluate metrics.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of evaluation metrics
        """
        self.monitor.start_stage("training")

        try:
            if self.model is None:
                raise ValueError("Model not created. Call create_model() first.")

            # Train model
            logger.info("Training model...")
            self.model.fit(X_train, y_train)

            # Cross-validation
            if self.cfg.training.cv_folds > 1:
                cv_scores = cross_val_score(
                    self.model,
                    X_train,
                    y_train,
                    cv=self.cfg.training.cv_folds,
                    scoring="accuracy",
                )
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std() * 2
                logger.info(f"CV Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")

            # Predict on test set
            y_pred = self.model.predict(X_test)

            # Calculate metrics
            self.metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(
                    precision_score(y_test, y_pred, average="weighted", zero_division=0)
                ),
                "recall": float(
                    recall_score(y_test, y_pred, average="weighted", zero_division=0)
                ),
                "f1_score": float(
                    f1_score(y_test, y_pred, average="weighted", zero_division=0)
                ),
            }

            if self.cfg.training.cv_folds > 1:
                self.metrics["cv_accuracy_mean"] = float(cv_scores.mean())
                self.metrics["cv_accuracy_std"] = float(cv_scores.std())

            logger.info(f"Test metrics: {self.metrics}")

            self.monitor.end_stage("training", success=True)
            return self.metrics

        except Exception as e:
            self.monitor.end_stage("training", success=False, error=str(e))
            raise

    def log_to_mlflow(self) -> str:
        """
        Log experiment to MLflow.

        Returns:
            MLflow run ID
        """
        self.monitor.start_stage("mlflow_logging")

        try:
            # Get original working directory
            original_cwd = hydra.utils.get_original_cwd()
            tracking_uri = f"file://{original_cwd}/mlruns"

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(self.cfg.mlflow.experiment_name)

            with mlflow.start_run(
                run_name=f"{self.cfg.model.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ) as run:
                # Log parameters
                params = OmegaConf.to_container(self.cfg.model.params, resolve=True)
                mlflow.log_params(params)
                mlflow.log_param("model_type", self.cfg.model.name)
                mlflow.log_param("cv_folds", self.cfg.training.cv_folds)
                mlflow.log_param("seed", self.cfg.seed)

                # Log metrics
                mlflow.log_metrics(self.metrics)

                # Log configuration as artifact
                config_path = Path("config.yaml")
                with open(config_path, "w") as f:
                    OmegaConf.save(self.cfg, f)
                mlflow.log_artifact(str(config_path))

                # Log model
                if self.cfg.training.register_model and self.model is not None:
                    mlflow.sklearn.log_model(
                        self.model,
                        "model",
                        registered_model_name=self.cfg.model.name,
                    )

                run_id: str = run.info.run_id
                logger.info(f"Logged to MLflow. Run ID: {run_id}")

                self.monitor.end_stage("mlflow_logging", success=True)
                return run_id

        except Exception as e:
            self.monitor.end_stage("mlflow_logging", success=False, error=str(e))
            raise

    def save_results(self, run_id: str) -> None:
        """
        Save pipeline results to output directory.

        Args:
            run_id: MLflow run ID
        """
        self.monitor.start_stage("save_results")

        try:
            # Get original working directory
            original_cwd = hydra.utils.get_original_cwd()

            # Create model-specific output directory
            model_name = self.cfg.model.name.lower().replace(" ", "_")
            output_dir = Path(original_cwd) / "outputs" / model_name
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save metrics
            metrics_file = output_dir / "metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(
                    {
                        "metrics": self.metrics,
                        "model": self.cfg.model.name,
                        "run_id": run_id,
                        "timestamp": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )

            logger.info(f"Results saved to {output_dir}")
            self.monitor.end_stage("save_results", success=True)

            # Update config output_dir for monitor
            self.cfg.output_dir = str(output_dir)

        except Exception as e:
            self.monitor.end_stage("save_results", success=False, error=str(e))
            raise

    def run(self) -> dict[str, Any]:
        """
        Execute full training pipeline.

        Returns:
            Dictionary with pipeline results
        """
        self.monitor.start_pipeline()

        try:
            # Execute pipeline stages
            X_train, y_train, X_test, y_test = self.load_data()
            self.create_model()
            metrics = self.train(X_train, y_train, X_test, y_test)
            run_id = self.log_to_mlflow()
            self.save_results(run_id)

            # Generate final report
            report = self.monitor.end_pipeline(success=True)
            self.monitor.save_report()

            return {
                "success": True,
                "metrics": metrics,
                "run_id": run_id,
                "model": self.cfg.model.name,
                "report": report,
            }

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            report = self.monitor.end_pipeline(success=False, error=str(e))
            self.monitor.save_report()

            return {
                "success": False,
                "error": str(e),
                "report": report,
            }


@hydra.main(config_path="../../conf", config_name="config", version_base=None)  # type: ignore[misc]
def main(cfg: DictConfig) -> dict[str, Any]:
    """
    Main entry point for training pipeline.

    Args:
        cfg: Hydra configuration

    Returns:
        Pipeline results
    """
    pipeline = TrainingPipeline(cfg)
    return pipeline.run()


if __name__ == "__main__":
    main()
