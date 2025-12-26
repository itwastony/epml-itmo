#!/usr/bin/env python3
"""
ClearML Dashboard and Analysis Utilities.

Provides utilities for:
- Creating experiment dashboards
- Generating comparison reports
- Visualizing model performance
- Exporting analysis results

Usage:
    python -m src.clearml_integration.dashboard --report
    python -m src.clearml_integration.dashboard --summary
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.clearml_integration.experiment_tracker import (  # noqa: E402
    ExperimentComparison,
)
from src.clearml_integration.model_manager import ClearMLModelManager  # noqa: E402

logger = logging.getLogger(__name__)


class ClearMLDashboard:
    """
    Dashboard utility for ClearML experiments and models.

    Features:
    - Generate experiment summaries
    - Create comparison reports
    - Export analysis to various formats
    """

    def __init__(
        self,
        project_name: str = "EPML-ITMO/Wine-Quality",
        output_dir: str = "outputs/clearml/dashboard",
    ):
        """
        Initialize dashboard.

        Args:
            project_name: ClearML project name
            output_dir: Output directory for reports
        """
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_comparison = ExperimentComparison(
            project_name=f"{project_name}/Experiments"
        )
        self.model_manager = ClearMLModelManager(project_name=f"{project_name}/Models")

    def get_experiments_summary(self) -> dict[str, Any]:
        """
        Get summary of all experiments.

        Returns:
            Summary dictionary
        """
        experiments = self.experiment_comparison.get_experiments()

        summary: dict[str, Any] = {
            "total_experiments": len(experiments),
            "successful": sum(1 for e in experiments if e.get("status") == "completed"),
            "failed": sum(1 for e in experiments if e.get("status") == "failed"),
            "running": sum(1 for e in experiments if e.get("status") == "running"),
            "experiments": experiments,
        }

        # Get best experiment by accuracy
        metrics_data = [e.get("metrics", {}) for e in experiments]
        accuracies = [
            m.get("classification/accuracy", m.get("metrics/accuracy", 0))
            for m in metrics_data
        ]
        if accuracies and max(accuracies) > 0:
            best_idx = accuracies.index(max(accuracies))
            summary["best_experiment"] = {
                "name": experiments[best_idx].get("name"),
                "accuracy": max(accuracies),
            }

        return summary

    def get_models_summary(self) -> dict[str, Any]:
        """
        Get summary of all registered models.

        Returns:
            Summary dictionary
        """
        all_models = self.model_manager.get_all_models()

        total_versions = sum(len(versions) for versions in all_models.values())

        summary: dict[str, Any] = {
            "total_models": len(all_models),
            "total_versions": total_versions,
            "models": {},
        }

        for model_name, versions in all_models.items():
            if versions:
                latest = max(versions, key=lambda v: v.get("version", 0))
                summary["models"][model_name] = {
                    "versions": len(versions),
                    "latest_version": latest.get("version"),
                    "latest_metrics": latest.get("metrics", {}),
                }

        # Get best model
        best = self.model_manager.get_best_model(metric="accuracy")
        if best:
            summary["best_model"] = {
                "id": best[0],
                "accuracy": best[1].get("metrics", {}).get("accuracy", 0),
            }

        return summary

    def generate_full_report(self) -> str:
        """
        Generate a comprehensive Markdown report.

        Returns:
            Report content as string
        """
        timestamp = datetime.now()

        report_lines = [
            "# ClearML Dashboard Report",
            f"\n*Generated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}*\n",
            f"*Project: {self.project_name}*\n",
            "---\n",
        ]

        # Experiments Section
        report_lines.append("## 📊 Experiments Summary\n")
        exp_summary = self.get_experiments_summary()
        report_lines.extend(
            [
                f"- **Total Experiments:** {exp_summary.get('total_experiments', 0)}",
                f"- **Successful:** {exp_summary.get('successful', 0)}",
                f"- **Failed:** {exp_summary.get('failed', 0)}",
                f"- **Running:** {exp_summary.get('running', 0)}",
            ]
        )

        if exp_summary.get("best_experiment"):
            best_exp = exp_summary["best_experiment"]
            report_lines.extend(
                [
                    f"\n**Best Experiment:** {best_exp.get('name')}",
                    f"- Accuracy: {best_exp.get('accuracy', 0):.4f}",
                ]
            )

        report_lines.append("\n")

        # Models Section
        report_lines.append("## 🤖 Models Summary\n")
        models_summary = self.get_models_summary()
        report_lines.extend(
            [
                f"- **Total Models:** {models_summary.get('total_models', 0)}",
                f"- **Total Versions:** {models_summary.get('total_versions', 0)}",
            ]
        )

        if models_summary.get("best_model"):
            best_model = models_summary["best_model"]
            report_lines.extend(
                [
                    f"\n**Best Model:** {best_model.get('id')}",
                    f"- Accuracy: {best_model.get('accuracy', 0):.4f}",
                ]
            )

        report_lines.append("\n")

        # Models Comparison Table
        if models_summary.get("models"):
            report_lines.append("### Model Versions\n")
            report_lines.append("| Model | Versions | Latest | Accuracy |")
            report_lines.append("|-------|----------|--------|----------|")

            for model_name, info in models_summary.get("models", {}).items():
                versions = info.get("versions", 0)
                latest = info.get("latest_version", "-")
                accuracy = info.get("latest_metrics", {}).get("accuracy", 0)
                report_lines.append(
                    f"| {model_name} | {versions} | v{latest} | {accuracy:.4f} |"
                )

            report_lines.append("\n")

        # Experiment Details
        report_lines.append("## 📋 Experiment Details\n")
        experiments = exp_summary.get("experiments", [])
        if experiments:
            report_lines.append("| Name | Status | Created |")
            report_lines.append("|------|--------|---------|")
            for exp in experiments[:10]:  # Limit to 10
                name = exp.get("name", "N/A")
                status = exp.get("status", "N/A")
                created = exp.get("created", "N/A")[:19]  # Trim timestamp
                report_lines.append(f"| {name} | {status} | {created} |")
        else:
            report_lines.append("*No experiments found.*")

        report_lines.append("\n---\n")

        # Footer
        report_lines.extend(
            [
                "## 🔗 Quick Links\n",
                "- **ClearML Web UI:** http://localhost:8080",
                "- **ClearML API:** http://localhost:8008",
                "- **Project Docs:** README.md",
            ]
        )

        report = "\n".join(report_lines)

        # Save report
        report_file = self.output_dir / "dashboard_report.md"
        with open(report_file, "w") as f:
            f.write(report)

        logger.info(f"Report saved to: {report_file}")
        return report

    def export_metrics_csv(self) -> Path:
        """
        Export all metrics to CSV.

        Returns:
            Path to CSV file
        """
        # Get model comparison
        df = self.model_manager.compare_models()

        if df.empty:
            logger.warning("No metrics to export")
            return Path()

        output_file = self.output_dir / "metrics_export.csv"
        df.to_csv(output_file, index=False)

        logger.info(f"Metrics exported to: {output_file}")
        return output_file

    def export_summary_json(self) -> Path:
        """
        Export summary to JSON.

        Returns:
            Path to JSON file
        """
        summary: dict[str, Any] = {
            "generated_at": datetime.now().isoformat(),
            "project": self.project_name,
            "experiments": self.get_experiments_summary(),
            "models": self.get_models_summary(),
        }

        output_file = self.output_dir / "summary.json"
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Summary exported to: {output_file}")
        return output_file

    def print_summary(self) -> None:
        """Print a quick summary to console."""
        print("\n" + "=" * 60)
        print("ClearML Dashboard Summary")
        print("=" * 60)

        # Experiments
        exp_summary = self.get_experiments_summary()
        print("\n📊 Experiments:")
        print(f"   Total: {exp_summary.get('total_experiments', 0)}")
        print(f"   Successful: {exp_summary.get('successful', 0)}")
        print(f"   Failed: {exp_summary.get('failed', 0)}")

        if exp_summary.get("best_experiment"):
            best = exp_summary["best_experiment"]
            print(f"\n   🏆 Best: {best.get('name')}")
            print(f"      Accuracy: {best.get('accuracy', 0):.4f}")

        # Models
        models_summary = self.get_models_summary()
        print("\n🤖 Models:")
        print(f"   Total: {models_summary.get('total_models', 0)}")
        print(f"   Versions: {models_summary.get('total_versions', 0)}")

        if models_summary.get("best_model"):
            best = models_summary["best_model"]
            print(f"\n   🏆 Best: {best.get('id')}")
            print(f"      Accuracy: {best.get('accuracy', 0):.4f}")

        # Model list
        if models_summary.get("models"):
            print("\n📋 Model Performance:")
            for model_name, info in models_summary.get("models", {}).items():
                acc = info.get("latest_metrics", {}).get("accuracy", 0)
                print(f"   - {model_name}: {acc:.4f}")

        print("\n" + "=" * 60)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ClearML Dashboard Utilities")
    parser.add_argument("--report", action="store_true", help="Generate full report")
    parser.add_argument("--summary", action="store_true", help="Print summary")
    parser.add_argument("--export-csv", action="store_true", help="Export metrics CSV")
    parser.add_argument(
        "--export-json", action="store_true", help="Export summary JSON"
    )
    parser.add_argument("--all", action="store_true", help="Run all exports")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    dashboard = ClearMLDashboard()

    if args.all:
        dashboard.print_summary()
        dashboard.generate_full_report()
        dashboard.export_metrics_csv()
        dashboard.export_summary_json()
    elif args.report:
        report = dashboard.generate_full_report()
        print(report)
    elif args.export_csv:
        dashboard.export_metrics_csv()
    elif args.export_json:
        dashboard.export_summary_json()
    else:
        dashboard.print_summary()


if __name__ == "__main__":
    main()
