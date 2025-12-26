import logging
import sys
import warnings
from pathlib import Path

import click
import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

warnings.filterwarnings("ignore")


@click.command()  # type: ignore[misc]
@click.argument("input_filepath", type=click.Path(exists=True))  # type: ignore[misc]
@click.option(  # type: ignore[misc]
    "--n_estimators", default=100, type=int, help="Number of trees in the forest"
)
@click.option(  # type: ignore[misc]
    "--max_depth", default=None, type=int, help="The maximum depth of the tree"
)
def main(input_filepath: str, n_estimators: int, max_depth: int) -> None:
    """Trains a model on processed data."""
    logger = logging.getLogger(__name__)
    logger.info("Training model...")

    # Load data
    train_path = Path(input_filepath) / "train.csv"
    test_path = Path(input_filepath) / "test.csv"

    if not train_path.exists() or not test_path.exists():
        logger.error("Processed data not found. Run make_dataset.py first.")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]

    # Set up MLflow
    mlflow.set_tracking_uri("file://" + str(Path.cwd() / "mlruns"))
    mlflow.set_experiment("wine_quality_experiment")

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Train model
        clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        clf.fit(X_train, y_train)

        # Predict
        y_pred = clf.predict(X_test)

        # metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"F1 Score: {f1}")

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log model
        mlflow.sklearn.log_model(
            clf, "model", registered_model_name="WineQualityRandomForest"
        )
        logger.info("Model logged to MLflow")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())
    main()
