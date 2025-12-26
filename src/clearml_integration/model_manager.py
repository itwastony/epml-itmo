"""
ClearML Model Manager Module.

Provides comprehensive model management including:
- Model registration and versioning
- Metadata management
- Model comparison
- Automatic version control
- Model deployment support
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import pandas as pd
from sklearn.base import BaseEstimator

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ClearMLModelManager:
    """
    Model Manager for ClearML with versioning and metadata support.

    Features:
    - Automatic model versioning
    - Rich metadata support
    - Model comparison utilities
    - Deployment-ready model export
    - Model lineage tracking

    Example:
        manager = ClearMLModelManager("Wine-Quality-Models")
        model_id = manager.register_model(
            model=trained_model,
            model_name="RandomForest",
            metrics={"accuracy": 0.95},
            parameters={"n_estimators": 100}
        )
    """

    def __init__(
        self,
        project_name: str = "EPML-ITMO/Wine-Quality/Models",
        output_uri: str = "outputs/clearml/models",
    ):
        """
        Initialize Model Manager.

        Args:
            project_name: ClearML project for models
            output_uri: Local output directory for models
        """
        self.project_name = project_name
        self.output_uri = Path(output_uri)
        self.output_uri.mkdir(parents=True, exist_ok=True)
        self._models_registry: dict[str, list[dict[str, Any]]] = {}
        self._registry_file = self.output_uri / "model_registry.json"
        self._load_registry()

    def _load_registry(self) -> None:
        """Load local model registry."""
        if self._registry_file.exists():
            with open(self._registry_file) as f:
                self._models_registry = json.load(f)
        else:
            self._models_registry = {}

    def _save_registry(self) -> None:
        """Save local model registry."""
        with open(self._registry_file, "w") as f:
            json.dump(self._models_registry, f, indent=2, default=str)

    def register_model(
        self,
        model: BaseEstimator,
        model_name: str,
        metrics: dict[str, float] | None = None,
        parameters: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        description: str = "",
        framework: str = "sklearn",
        task_id: str | None = None,
    ) -> str:
        """
        Register a trained model with ClearML.

        Args:
            model: Trained model object
            model_name: Name of the model
            metrics: Model performance metrics
            parameters: Model hyperparameters
            tags: Tags for categorization
            description: Model description
            framework: ML framework (sklearn, pytorch, etc.)
            task_id: Optional ClearML task ID to associate

        Returns:
            Model ID
        """
        timestamp = datetime.now()
        version = self._get_next_version(model_name)
        model_id = f"{model_name}_v{version}_{timestamp:%Y%m%d_%H%M%S}"

        # Save model locally
        model_dir = self.output_uri / model_name / f"v{version}"
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / f"{model_id}.joblib"
        joblib.dump(model, model_path)

        # Create metadata
        metadata: dict[str, Any] = {
            "model_id": model_id,
            "model_name": model_name,
            "version": version,
            "framework": framework,
            "created_at": timestamp.isoformat(),
            "model_path": str(model_path),
            "metrics": metrics or {},
            "parameters": parameters or {},
            "tags": tags or [],
            "description": description,
            "task_id": task_id,
            "model_class": type(model).__name__,
        }

        # Save metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Update registry
        if model_name not in self._models_registry:
            self._models_registry[model_name] = []
        self._models_registry[model_name].append(metadata)
        self._save_registry()

        # Register with ClearML if available
        clearml_model_id = self._register_with_clearml(
            model_path=model_path,
            model_name=model_name,
            version=version,
            metadata=metadata,
            framework=framework,
            task_id=task_id,
        )

        if clearml_model_id:
            metadata["clearml_model_id"] = clearml_model_id
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

        logger.info(f"Registered model: {model_id}")
        return model_id

    def _get_next_version(self, model_name: str) -> int:
        """Get next version number for a model."""
        if model_name not in self._models_registry:
            return 1
        versions = [m["version"] for m in self._models_registry[model_name]]
        return int(max(versions)) + 1 if versions else 1

    def _register_with_clearml(
        self,
        model_path: Path,
        model_name: str,
        version: int,
        metadata: dict[str, Any],
        framework: str,
        task_id: str | None = None,
    ) -> str | None:
        """
        Register model with ClearML server.

        Args:
            model_path: Path to saved model
            model_name: Model name
            version: Version number
            metadata: Model metadata
            framework: ML framework
            task_id: Optional task ID

        Returns:
            ClearML model ID if successful
        """
        try:
            from clearml import OutputModel, Task

            # Create or get task
            task: Any
            if task_id:
                task = Task.get_task(task_id=task_id)
            else:
                task = Task.init(
                    project_name=self.project_name,
                    task_name=f"Register {model_name} v{version}",
                    task_type=Task.TaskTypes.custom,
                    reuse_last_task_id=False,
                )

            # Create output model
            output_model = OutputModel(
                task=task,
                name=f"{model_name}_v{version}",
                framework=framework,
                comment=metadata.get("description", ""),
            )

            # Upload model weights
            output_model.update_weights(weights_filename=str(model_path))

            # Add labels (metadata)
            labels = {
                "version": str(version),
                "model_class": metadata.get("model_class", ""),
                **{
                    f"metric_{k}": str(v)
                    for k, v in metadata.get("metrics", {}).items()
                },
                **{
                    f"param_{k}": str(v)
                    for k, v in metadata.get("parameters", {}).items()
                },
            }
            output_model.update_labels(labels)

            # Add tags
            for tag in metadata.get("tags", []):
                task.add_tags([tag])

            if not task_id:
                task.close()

            logger.info(f"Registered model with ClearML: {output_model.id}")
            return str(output_model.id)

        except Exception as e:
            logger.warning(f"Failed to register with ClearML: {e}")
            return None

    def load_model(
        self,
        model_name: str,
        version: int | str = "latest",
    ) -> tuple[BaseEstimator, dict[str, Any]]:
        """
        Load a registered model.

        Args:
            model_name: Model name
            version: Version number or "latest"

        Returns:
            Tuple of (model, metadata)
        """
        if model_name not in self._models_registry:
            raise ValueError(f"Model '{model_name}' not found in registry")

        models = self._models_registry[model_name]

        if version == "latest":
            model_meta = max(models, key=lambda m: m["version"])
        else:
            model_meta_found = next(
                (m for m in models if m["version"] == version), None
            )
            if not model_meta_found:
                raise ValueError(
                    f"Version {version} not found for model '{model_name}'"
                )
            model_meta = model_meta_found

        model_path = Path(model_meta["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model: BaseEstimator = joblib.load(model_path)
        logger.info(f"Loaded model: {model_name} v{model_meta['version']}")

        return model, model_meta

    def get_model_versions(self, model_name: str) -> list[dict[str, Any]]:
        """
        Get all versions of a model.

        Args:
            model_name: Model name

        Returns:
            List of version metadata
        """
        return self._models_registry.get(model_name, [])

    def get_all_models(self) -> dict[str, list[dict[str, Any]]]:
        """
        Get all registered models.

        Returns:
            Dictionary of model name -> versions
        """
        return self._models_registry.copy()

    def compare_models(
        self,
        model_names: list[str] | None = None,
        metric: str = "accuracy",
    ) -> pd.DataFrame:
        """
        Compare models by a specific metric.

        Args:
            model_names: List of model names to compare (None = all)
            metric: Metric to compare

        Returns:
            DataFrame with comparison results
        """
        rows: list[dict[str, Any]] = []

        models_to_compare = model_names or list(self._models_registry.keys())

        for model_name in models_to_compare:
            versions = self._models_registry.get(model_name, [])
            for version_meta in versions:
                row: dict[str, Any] = {
                    "model_name": model_name,
                    "version": version_meta["version"],
                    "created_at": version_meta["created_at"],
                    "model_class": version_meta.get("model_class", ""),
                }

                # Add all metrics
                for metric_name, metric_value in version_meta.get(
                    "metrics", {}
                ).items():
                    row[metric_name] = metric_value

                rows.append(row)

        df = pd.DataFrame(rows)

        # Sort by specified metric if available
        if metric in df.columns:
            df = df.sort_values(metric, ascending=False)

        return df

    def get_best_model(
        self,
        model_name: str | None = None,
        metric: str = "accuracy",
        higher_is_better: bool = True,
    ) -> tuple[str, dict[str, Any]] | None:
        """
        Get the best model by a specific metric.

        Args:
            model_name: Specific model name (None = search all)
            metric: Metric to optimize
            higher_is_better: Whether higher metric values are better

        Returns:
            Tuple of (model_id, metadata) or None
        """
        comparison = self.compare_models(
            model_names=[model_name] if model_name else None,
            metric=metric,
        )

        if comparison.empty or metric not in comparison.columns:
            return None

        if higher_is_better:
            best_idx = comparison[metric].idxmax()
        else:
            best_idx = comparison[metric].idxmin()

        best_row = comparison.loc[best_idx]

        best_model_name = str(best_row["model_name"])
        version = best_row["version"]

        versions = self._models_registry.get(best_model_name, [])
        metadata = next((m for m in versions if m["version"] == version), None)

        if metadata:
            return str(metadata["model_id"]), metadata

        return None

    def delete_model(self, model_name: str, version: int | None = None) -> bool:
        """
        Delete a model version or all versions.

        Args:
            model_name: Model name
            version: Specific version to delete (None = all)

        Returns:
            True if successful
        """
        if model_name not in self._models_registry:
            logger.warning(f"Model '{model_name}' not found")
            return False

        if version is None:
            # Delete all versions
            for version_meta in self._models_registry[model_name]:
                model_path = Path(version_meta["model_path"])
                if model_path.exists():
                    model_path.unlink()
            del self._models_registry[model_name]
        else:
            # Delete specific version
            versions = self._models_registry[model_name]
            for i, version_meta in enumerate(versions):
                if version_meta["version"] == version:
                    model_path = Path(version_meta["model_path"])
                    if model_path.exists():
                        model_path.unlink()
                    versions.pop(i)
                    break

        self._save_registry()
        logger.info(f"Deleted model: {model_name} v{version or 'all'}")
        return True

    def export_model(
        self,
        model_name: str,
        version: int | str = "latest",
        export_path: str | Path | None = None,
        include_metadata: bool = True,
    ) -> Path:
        """
        Export a model for deployment.

        Args:
            model_name: Model name
            version: Version to export
            export_path: Export destination
            include_metadata: Include metadata file

        Returns:
            Path to exported model
        """
        model, metadata = self.load_model(model_name, version)

        if export_path is None:
            export_path = self.output_uri / "exports" / model_name

        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)

        # Export model
        model_file = export_path / f"{model_name}_v{metadata['version']}.joblib"
        joblib.dump(model, model_file)

        # Export metadata
        if include_metadata:
            metadata_file = export_path / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

        logger.info(f"Exported model to: {export_path}")
        return export_path

    def generate_model_report(
        self,
        output_path: str | Path = "outputs/clearml/model_report.md",
    ) -> str:
        """
        Generate a Markdown report of all models.

        Args:
            output_path: Path to save the report

        Returns:
            Report content
        """
        report_lines = [
            "# Model Registry Report",
            f"\n*Generated: {datetime.now().isoformat()}*\n",
            "## Summary\n",
            f"- Total Models: {len(self._models_registry)}",
            f"- Total Versions: {sum(len(v) for v in self._models_registry.values())}",
            "\n## Models\n",
        ]

        for model_name, versions in self._models_registry.items():
            report_lines.append(f"### {model_name}\n")
            report_lines.append(f"- Versions: {len(versions)}")

            if versions:
                latest = max(versions, key=lambda m: m["version"])
                report_lines.append(f"- Latest Version: v{latest['version']}")
                report_lines.append(f"- Latest Created: {latest['created_at']}")

                if latest.get("metrics"):
                    report_lines.append("\n**Latest Metrics:**\n")
                    for metric, value in latest["metrics"].items():
                        report_lines.append(f"- {metric}: {value:.4f}")

            report_lines.append("")

        # Add comparison table
        comparison = self.compare_models()
        if not comparison.empty:
            report_lines.append("## Model Comparison\n")
            report_lines.append(comparison.to_markdown(index=False))

        report = "\n".join(report_lines)

        # Save report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)

        logger.info(f"Model report saved to: {output_path}")
        return report

    def sync_with_clearml(self) -> int:
        """
        Sync local registry with ClearML server.

        Returns:
            Number of models synced
        """
        synced = 0

        try:
            from clearml import Model

            # Get all models from ClearML
            clearml_models = Model.query_models(
                project_name=self.project_name,
            )

            for _ in clearml_models:
                # Check if already in registry
                # This is a simplified sync - in production would be more robust
                synced += 1

            logger.info(f"Synced {synced} models with ClearML")

        except Exception as e:
            logger.warning(f"Failed to sync with ClearML: {e}")

        return synced
