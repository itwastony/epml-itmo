"""
Data preparation pipeline with Hydra configuration.

Handles data downloading, preprocessing, and splitting.
"""

import logging
import sys
import urllib.request
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logger = logging.getLogger(__name__)

# Wine Quality dataset URL
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"


def download_data(raw_path: Path, force: bool = False) -> Path:
    """
    Download Wine Quality dataset.

    Args:
        raw_path: Directory to save raw data
        force: Force re-download even if file exists

    Returns:
        Path to downloaded file
    """
    raw_path.mkdir(parents=True, exist_ok=True)
    file_path = raw_path / "winequality-red.csv"

    if file_path.exists() and not force:
        logger.info(f"File already exists: {file_path}")
        return file_path

    logger.info(f"Downloading data from {DATASET_URL}")
    urllib.request.urlretrieve(DATASET_URL, file_path)  # nosec
    logger.info(f"Downloaded to: {file_path}")

    return file_path


def preprocess_data(
    df: pd.DataFrame,
    normalize: bool = False,
    handle_missing: str = "drop",
) -> pd.DataFrame:
    """
    Preprocess data according to configuration.

    Args:
        df: Input DataFrame
        normalize: Whether to normalize features
        handle_missing: Missing value handling strategy

    Returns:
        Preprocessed DataFrame
    """
    # Handle missing values
    if handle_missing == "drop":
        df = df.dropna()
    elif handle_missing == "mean":
        df = df.fillna(df.mean(numeric_only=True))
    elif handle_missing == "median":
        df = df.fillna(df.median(numeric_only=True))

    logger.info(f"After handling missing values: {df.shape}")

    # Normalize features (except target)
    if normalize:
        feature_cols = df.columns[:-1]
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        logger.info("Features normalized")

    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets.

    Args:
        df: Input DataFrame
        test_size: Test set proportion
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, test_df)
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
    )

    logger.info(f"Train set: {train_df.shape}")
    logger.info(f"Test set: {test_df.shape}")

    return train_df, test_df


def save_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_path: Path,
    train_file: str = "train.csv",
    test_file: str = "test.csv",
) -> None:
    """
    Save processed data to files.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        output_path: Output directory
        train_file: Training file name
        test_file: Test file name
    """
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = output_path / train_file
    test_path = output_path / test_file

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Saved train data to: {train_path}")
    logger.info(f"Saved test data to: {test_path}")


@hydra.main(config_path="../../conf", config_name="config", version_base=None)  # type: ignore[misc]
def main(cfg: DictConfig) -> None:
    """
    Main data preparation pipeline.

    Args:
        cfg: Hydra configuration
    """
    # Setup logging
    log_level = getattr(logging, cfg.logging.level, logging.INFO)
    logging.basicConfig(level=log_level, format=cfg.logging.format)

    logger.info("Starting data preparation pipeline")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg.data)}")

    # Get original working directory
    original_cwd = hydra.utils.get_original_cwd()

    # Paths - access directly from cfg.data (Hydra merges defaults)
    raw_path = Path(original_cwd) / cfg.data.raw_path
    processed_path = Path(original_cwd) / cfg.data.processed_path

    # Download data
    data_file = download_data(raw_path)

    # Load data
    df = pd.read_csv(data_file, sep=";")
    logger.info(f"Loaded dataset: {df.shape}")

    # Get preprocessing config if available
    normalize = False
    handle_missing = "drop"

    if "preprocessing" in cfg.data:
        normalize = cfg.data.preprocessing.get("normalize", False)
        handle_missing = cfg.data.preprocessing.get("handle_missing", "drop")

    # Preprocess data
    df = preprocess_data(df, normalize=normalize, handle_missing=handle_missing)

    # Split data
    train_df, test_df = split_data(
        df,
        test_size=cfg.data.test_size,
        random_state=cfg.data.random_state,
    )

    # Save data
    save_data(
        train_df,
        test_df,
        processed_path,
        cfg.data.train_file,
        cfg.data.test_file,
    )

    logger.info("Data preparation completed successfully")


if __name__ == "__main__":
    main()
