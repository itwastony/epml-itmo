"""
Script to run training pipeline for all configured models.

Uses Hydra's multirun feature for parallel execution.
"""

import logging
import subprocess  # nosec B404
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Available model configurations
MODELS = [
    "random_forest",
    "gradient_boosting",
    "logistic_regression",
    "svm",
    "decision_tree",
    "knn",
]


def run_single_model(model: str) -> tuple[str, bool, str]:
    """
    Run training pipeline for a single model.

    Args:
        model: Model configuration name

    Returns:
        Tuple of (model_name, success, output)
    """
    logger.info(f"Running training for model: {model}")

    cmd = [
        sys.executable,
        "-m",
        "src.pipelines.train_pipeline",
        f"model={model}",
    ]

    try:
        result = subprocess.run(  # nosec B603
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[2],
            timeout=300,  # 5 minute timeout
        )

        success = result.returncode == 0
        output = result.stdout if success else result.stderr

        if success:
            logger.info(f"✓ {model} completed successfully")
        else:
            logger.error(f"✗ {model} failed: {result.stderr}")

        return model, success, output

    except subprocess.TimeoutExpired:
        logger.error(f"✗ {model} timed out")
        return model, False, "Execution timed out"
    except Exception as e:
        logger.error(f"✗ {model} error: {e}")
        return model, False, str(e)


def run_all_models(models: list[str] | None = None) -> dict[str, bool]:
    """
    Run training pipeline for all specified models.

    Args:
        models: List of model names to run. Defaults to all models.

    Returns:
        Dictionary mapping model names to success status
    """
    if models is None:
        models = MODELS

    logger.info(f"Running {len(models)} model(s): {models}")
    logger.info("=" * 60)

    results = {}

    for model in models:
        model_name, success, output = run_single_model(model)
        results[model_name] = success
        logger.info("-" * 40)

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    successful = sum(1 for v in results.values() if v)
    failed = len(results) - successful

    for model, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"  {model}: {status}")

    logger.info("-" * 40)
    logger.info(f"Total: {successful} succeeded, {failed} failed")

    return results


def run_multirun() -> None:
    """
    Run all models using Hydra's multirun feature.

    This enables parallel execution when configured.
    """
    models_str = ",".join(MODELS)

    cmd = [
        sys.executable,
        "-m",
        "src.pipelines.train_pipeline",
        "--multirun",
        f"model={models_str}",
    ]

    logger.info(f"Running multirun with models: {models_str}")

    result = subprocess.run(  # nosec B603
        cmd,
        cwd=Path(__file__).resolve().parents[2],
    )

    if result.returncode == 0:
        logger.info("Multirun completed successfully")
    else:
        logger.error("Multirun failed")


def main() -> None:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Parse arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--multirun":
            run_multirun()
        else:
            # Run specific models
            models = sys.argv[1:]
            run_all_models(models)
    else:
        # Run all models sequentially
        run_all_models()


if __name__ == "__main__":
    main()
