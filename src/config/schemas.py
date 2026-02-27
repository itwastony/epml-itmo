"""
Configuration validation schemas using Pydantic.
Provides type-safe configuration validation for the ML pipeline.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):  # type: ignore[misc]
    """Data configuration schema."""

    raw_path: str = Field(default="data/raw", description="Path to raw data")
    processed_path: str = Field(
        default="data/processed", description="Path to processed data"
    )
    train_file: str = Field(default="train.csv", description="Training data filename")
    test_file: str = Field(default="test.csv", description="Test data filename")
    target_column: str = Field(default="quality", description="Target column name")
    test_size: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Test set size ratio"
    )
    random_state: int = Field(default=42, description="Random state for splitting")


class PreprocessingConfig(BaseModel):  # type: ignore[misc]
    """Preprocessing configuration schema."""

    normalize: bool = Field(default=False, description="Whether to normalize features")
    handle_missing: Literal["drop", "mean", "median"] = Field(
        default="drop", description="Missing value handling strategy"
    )


class DataFullConfig(BaseModel):  # type: ignore[misc]
    """Full data configuration including preprocessing."""

    data: DataConfig = Field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)


class ModelParamsConfig(BaseModel):  # type: ignore[misc]
    """Base model parameters configuration."""

    random_state: int = Field(default=42, description="Random state for model")

    class Config:
        extra = "allow"  # Allow additional model-specific parameters


class ModelConfig(BaseModel):  # type: ignore[misc]
    """Model configuration schema."""

    name: str = Field(..., description="Model name")
    _target_: str = Field(..., description="Full path to model class")
    params: dict[str, Any] = Field(
        default_factory=dict, description="Model hyperparameters"
    )

    @field_validator("name")  # type: ignore[misc]
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name is not empty."""
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v


class TrainingConfig(BaseModel):  # type: ignore[misc]
    """Training configuration schema."""

    cv_folds: int = Field(default=5, ge=2, le=20, description="Number of CV folds")
    shuffle: bool = Field(default=True, description="Shuffle data before CV")
    metrics: list[str] = Field(
        default=["accuracy", "precision", "recall", "f1_score"],
        description="Metrics to track",
    )
    register_model: bool = Field(default=True, description="Register model to MLflow")
    model_name: str = Field(default="", description="Registered model name")


class EarlyStoppingConfig(BaseModel):  # type: ignore[misc]
    """Early stopping configuration schema."""

    enabled: bool = Field(default=False, description="Enable early stopping")
    patience: int = Field(default=10, ge=1, description="Patience for early stopping")
    min_delta: float = Field(
        default=0.001, ge=0.0, description="Minimum improvement delta"
    )


class MLflowConfig(BaseModel):  # type: ignore[misc]
    """MLflow configuration schema."""

    tracking_uri: str = Field(..., description="MLflow tracking URI")
    experiment_name: str = Field(..., description="Experiment name")


class LoggingConfig(BaseModel):  # type: ignore[misc]
    """Logging configuration schema."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string",
    )


class PipelineConfig(BaseModel):  # type: ignore[misc]
    """Full pipeline configuration schema."""

    model: ModelConfig
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    mlflow: MLflowConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    seed: int = Field(default=42, description="Global random seed")
    output_dir: str = Field(default="outputs", description="Output directory")


def validate_config(config: dict[str, Any]) -> PipelineConfig:
    """
    Validate configuration dictionary against schema.

    Args:
        config: Configuration dictionary from Hydra

    Returns:
        Validated PipelineConfig instance

    Raises:
        ValidationError: If configuration is invalid
    """
    return PipelineConfig(**config)
