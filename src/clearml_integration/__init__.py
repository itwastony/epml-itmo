"""
ClearML Integration Module for EPML-ITMO Project.

This module provides comprehensive ClearML integration including:
- Experiment tracking with automatic logging
- Model management and versioning
- Pipeline orchestration
- Dashboard and comparison utilities

Example usage:
    from src.clearml_integration import ClearMLExperiment

    with ClearMLExperiment("my_experiment") as exp:
        exp.log_parameters({"lr": 0.01})
        exp.log_metrics({"accuracy": 0.95})
        exp.log_model(model, "best_model")
"""

from src.clearml_integration.experiment_tracker import (
    ClearMLExperiment,
    clearml_experiment,
)
from src.clearml_integration.model_manager import ClearMLModelManager
from src.clearml_integration.pipeline import ClearMLPipeline

__all__ = [
    "ClearMLExperiment",
    "clearml_experiment",
    "ClearMLModelManager",
    "ClearMLPipeline",
]
