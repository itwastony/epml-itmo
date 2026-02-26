import functools
import logging
import time
from collections.abc import Callable
from typing import Any

import mlflow

logger = logging.getLogger(__name__)


def log_experiment(experiment_name: str = "default_experiment") -> Callable[..., Any]:
    """
    Decorator to wrap a function execution in an MLflow run.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            mlflow.set_experiment(experiment_name)

            # Start run
            with mlflow.start_run(run_name=func.__name__) as run:
                logger.info(f"Started MLflow run: {run.info.run_id}")

                # Log execution time
                start_time = time.time()

                try:
                    # Execute function
                    result = func(*args, **kwargs)

                    # Log duration
                    duration = time.time() - start_time
                    mlflow.log_metric("execution_time", duration)

                    return result

                except Exception as e:
                    # Log error
                    logger.error(f"Error in {func.__name__}: {str(e)}")
                    mlflow.set_tag("status", "failed")
                    mlflow.log_param("error", str(e))
                    raise e

        return wrapper

    return decorator


def log_metrics(metrics: dict[str, float]) -> None:
    """Helper to log multiple metrics."""
    for name, value in metrics.items():
        mlflow.log_metric(name, value)


def log_params(params: dict[str, Any]) -> None:
    """Helper to log multiple parameters."""
    for name, value in params.items():
        mlflow.log_param(name, value)
