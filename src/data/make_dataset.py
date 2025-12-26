import logging
import urllib.request
from pathlib import Path

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split


@click.command()  # type: ignore[misc]
@click.argument("input_filepath", type=click.Path())  # type: ignore[misc]
@click.argument("output_filepath", type=click.Path())  # type: ignore[misc]
def main(input_filepath: str, output_filepath: str) -> None:
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("Downloading and processing dataset...")

    # Download Wine Quality Dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    raw_path = Path(input_filepath) / "winequality-red.csv"

    if not raw_path.exists():
        logger.info(f"Downloading data from {url} to {raw_path}")
        urllib.request.urlretrieve(url, raw_path)  # nosec
    else:
        logger.info(f"File {raw_path} already exists")

    # Process data
    df = pd.read_csv(raw_path, sep=";")
    logger.info(f"Dataset shape: {df.shape}")

    # Simple processing: split into train/test
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    output_path = Path(output_filepath)
    output_path.mkdir(parents=True, exist_ok=True)

    train.to_csv(output_path / "train.csv", index=False)
    test.to_csv(output_path / "test.csv", index=False)

    logger.info(f"Saved processed data to {output_path}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())

    main()
