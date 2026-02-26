import logging
import sys
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Add project root to path to ensure imports work
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.mlflow_decorators import (  # noqa: E402
    log_experiment,
    log_metrics,
    log_params,
)

logger = logging.getLogger(__name__)


class ExperimentRunner:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.X_train: pd.DataFrame
        self.y_train: pd.Series
        self.X_test: pd.DataFrame
        self.y_test: pd.Series
        self._load_data()

    def _load_data(self) -> None:
        train_path = self.data_path / "train.csv"
        test_path = self.data_path / "test.csv"

        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(f"Data not found at {self.data_path}")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        self.X_train = train_df.iloc[:, :-1]
        self.y_train = train_df.iloc[:, -1]
        self.X_test = test_df.iloc[:, :-1]
        self.y_test = test_df.iloc[:, -1]

    @log_experiment(experiment_name="wine_quality_multimodel_v1")
    def run_experiment(
        self,
        model_name: str,
        model_class: type[ClassifierMixin],
        params: dict[str, Any],
    ) -> dict[str, float]:
        """
        Runs a single experiment with the given model and parameters.
        """
        logger.info(f"Running experiment with {model_name} and params {params}")

        # Log params
        log_params(params)
        mlflow.log_param("model_type", model_name)

        # Initialize and train model
        model = model_class(**params)
        model.fit(self.X_train, self.y_train)

        # Predict
        y_pred = model.predict(self.X_test)

        # Calculate metrics
        metrics = {
            "accuracy": float(accuracy_score(self.y_test, y_pred)),
            "precision": float(
                precision_score(self.y_test, y_pred, average="weighted")
            ),
            "recall": float(recall_score(self.y_test, y_pred, average="weighted")),
            "f1_score": float(f1_score(self.y_test, y_pred, average="weighted")),
        }

        # Log metrics
        log_metrics(metrics)
        logger.info(f"Metrics: {metrics}")

        # Log model
        mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)

        return metrics


def main() -> None:
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Path to processed data
    data_path = Path("data/processed")
    runner = ExperimentRunner(data_path)

    # Define experiments
    # (name, class, params)
    experiments: list[tuple[str, type[ClassifierMixin], dict[str, Any]]] = [
        # Random Forest Experiments
        (
            "RandomForest",
            RandomForestClassifier,
            {"n_estimators": 50, "max_depth": 5, "random_state": 42},
        ),
        (
            "RandomForest",
            RandomForestClassifier,
            {"n_estimators": 100, "max_depth": 10, "random_state": 42},
        ),
        (
            "RandomForest",
            RandomForestClassifier,
            {"n_estimators": 200, "max_depth": None, "random_state": 42},
        ),
        (
            "RandomForest",
            RandomForestClassifier,
            {"n_estimators": 50, "min_samples_split": 5, "random_state": 42},
        ),
        # Gradient Boosting Experiments
        (
            "GradientBoosting",
            GradientBoostingClassifier,
            {"n_estimators": 50, "learning_rate": 0.1, "random_state": 42},
        ),
        (
            "GradientBoosting",
            GradientBoostingClassifier,
            {
                "n_estimators": 100,
                "learning_rate": 0.05,
                "max_depth": 3,
                "random_state": 42,
            },
        ),
        (
            "GradientBoosting",
            GradientBoostingClassifier,
            {"n_estimators": 100, "learning_rate": 0.2, "random_state": 42},
        ),
        # Logistic Regression Experiments
        (
            "LogisticRegression",
            LogisticRegression,
            {"C": 0.1, "max_iter": 1000, "random_state": 42},
        ),
        (
            "LogisticRegression",
            LogisticRegression,
            {"C": 1.0, "max_iter": 1000, "random_state": 42},
        ),
        (
            "LogisticRegression",
            LogisticRegression,
            {"C": 10.0, "max_iter": 1000, "random_state": 42},
        ),
        # SVM Experiments
        ("SVM", SVC, {"kernel": "rbf", "C": 1.0, "random_state": 42}),
        ("SVM", SVC, {"kernel": "linear", "C": 1.0, "random_state": 42}),
        ("SVM", SVC, {"kernel": "poly", "degree": 3, "random_state": 42}),
        # Decision Tree Experiments
        (
            "DecisionTree",
            DecisionTreeClassifier,
            {"max_depth": 5, "random_state": 42},
        ),
        (
            "DecisionTree",
            DecisionTreeClassifier,
            {"max_depth": 10, "min_samples_split": 5, "random_state": 42},
        ),
        # KNN Experiments
        ("KNN", KNeighborsClassifier, {"n_neighbors": 3}),
        ("KNN", KNeighborsClassifier, {"n_neighbors": 5}),
        ("KNN", KNeighborsClassifier, {"n_neighbors": 7}),
    ]

    logger.info(f"Starting {len(experiments)} experiments...")

    mlflow.set_tracking_uri("file://" + str(Path.cwd() / "mlruns"))

    for model_name, model_class, params in experiments:
        try:
            runner.run_experiment(model_name, model_class, params)
        except Exception as e:
            logger.error(f"Experiment failed: {e}")


if __name__ == "__main__":
    main()
