"""
Модуль автоматической генерации отчётов об экспериментах.

Генерирует:
- Markdown отчёты с результатами
- Таблицы сравнения моделей
- Визуализации метрик
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Генератор отчётов об экспериментах."""

    def __init__(
        self,
        outputs_dir: Path | str = "outputs",
        reports_dir: Path | str = "outputs/reports",
    ):
        """
        Инициализация генератора отчётов.

        Args:
            outputs_dir: Директория с результатами экспериментов
            reports_dir: Директория для сохранения отчётов
        """
        self.outputs_dir = Path(outputs_dir)
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def collect_metrics(self) -> list[dict[str, Any]]:
        """
        Сбор метрик из всех экспериментов.

        Returns:
            Список словарей с метриками моделей
        """
        metrics_list = []

        # Поиск metrics.json в подпапках
        for metrics_file in self.outputs_dir.rglob("metrics.json"):
            try:
                with open(metrics_file) as f:
                    metrics = json.load(f)

                # Определение названия модели из пути
                model_name = metrics_file.parent.name

                metrics_list.append(
                    {
                        "model": model_name,
                        "path": str(metrics_file),
                        "metrics": metrics,
                    }
                )
                logger.info(f"Loaded metrics for {model_name}")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load {metrics_file}: {e}")

        return metrics_list

    def create_comparison_table(
        self, metrics_list: list[dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Создание таблицы сравнения моделей.

        Args:
            metrics_list: Список метрик моделей

        Returns:
            DataFrame с таблицей сравнения
        """
        rows = []

        for item in metrics_list:
            model_name = item["model"]
            metrics = item["metrics"]

            row = {
                "Модель": model_name,
                "Accuracy": metrics.get("accuracy", 0),
                "Precision": metrics.get("precision_weighted", 0),
                "Recall": metrics.get("recall_weighted", 0),
                "F1 Score": metrics.get("f1_weighted", 0),
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        # Сортировка по Accuracy
        if not df.empty:
            df = df.sort_values("Accuracy", ascending=False)

        return df

    def generate_ascii_bar(self, value: float, max_value: float = 1.0) -> str:
        """
        Генерация ASCII бара для визуализации.

        Args:
            value: Значение метрики
            max_value: Максимальное значение

        Returns:
            ASCII строка с баром
        """
        bar_length = 30
        filled = int((value / max_value) * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)
        return f"{bar} {value * 100:.2f}%"

    def generate_markdown_report(self, metrics_list: list[dict[str, Any]]) -> str:
        """
        Генерация полного Markdown отчёта.

        Args:
            metrics_list: Список метрик моделей

        Returns:
            Markdown строка с отчётом
        """
        lines = [
            "# Отчёт об экспериментах",
            "",
            f"*Сгенерировано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "## Обзор",
            "",
            "Результаты обучения моделей для задачи классификации качества вина.",
            "",
        ]

        if not metrics_list:
            lines.append("⚠️ Метрики не найдены. Запустите эксперименты:")
            lines.append("```bash")
            lines.append("make clearml_experiments_offline")
            lines.append("```")
            return "\n".join(lines)

        # Таблица сравнения
        df = self.create_comparison_table(metrics_list)

        lines.extend(
            [
                "## Сравнение моделей",
                "",
            ]
        )

        # Форматирование таблицы
        lines.append("| Модель | Accuracy | Precision | Recall | F1 Score |")
        lines.append("|--------|----------|-----------|--------|----------|")

        for _, row in df.iterrows():
            lines.append(
                f"| {row['Модель']} | "
                f"{row['Accuracy']:.4f} | "
                f"{row['Precision']:.4f} | "
                f"{row['Recall']:.4f} | "
                f"{row['F1 Score']:.4f} |"
            )

        lines.append("")

        # Лучшая модель
        if not df.empty:
            best_model = df.iloc[0]
            lines.extend(
                [
                    "## Лучшая модель",
                    "",
                    f"**{best_model['Модель']}** показывает лучшие результаты:",
                    "",
                    f"- Accuracy: {best_model['Accuracy']:.4f}",
                    f"- F1 Score: {best_model['F1 Score']:.4f}",
                    "",
                ]
            )

        # Визуализация
        lines.extend(
            [
                "## Визуализация Accuracy",
                "",
                "```",
            ]
        )

        for _, row in df.iterrows():
            model_name = row["Модель"][:18].ljust(18)
            bar = self.generate_ascii_bar(row["Accuracy"])
            lines.append(f"{model_name} {bar}")

        lines.extend(
            [
                "```",
                "",
            ]
        )

        # Визуализация F1
        lines.extend(
            [
                "## Визуализация F1 Score",
                "",
                "```",
            ]
        )

        for _, row in df.iterrows():
            model_name = row["Модель"][:18].ljust(18)
            bar = self.generate_ascii_bar(row["F1 Score"])
            lines.append(f"{model_name} {bar}")

        lines.extend(
            [
                "```",
                "",
            ]
        )

        # Инструкции по воспроизведению
        lines.extend(
            [
                "## Воспроизведение результатов",
                "",
                "```bash",
                "# Клонирование и установка",
                "git clone https://github.com/username/epml_itmo.git",
                "cd epml_itmo",
                "poetry install",
                "",
                "# Подготовка данных",
                "make prepare",
                "",
                "# Запуск экспериментов",
                "make clearml_experiments_offline",
                "",
                "# Генерация отчётов",
                "make generate_reports",
                "```",
                "",
            ]
        )

        return "\n".join(lines)

    def generate_csv_report(
        self, metrics_list: list[dict[str, Any]], output_path: Path | None = None
    ) -> Path:
        """
        Генерация CSV отчёта.

        Args:
            metrics_list: Список метрик моделей
            output_path: Путь для сохранения

        Returns:
            Путь к сохранённому файлу
        """
        df = self.create_comparison_table(metrics_list)

        if output_path is None:
            output_path = self.reports_dir / "metrics_comparison.csv"

        df.to_csv(output_path, index=False)
        logger.info(f"CSV report saved to {output_path}")

        return output_path

    def generate_json_report(
        self, metrics_list: list[dict[str, Any]], output_path: Path | None = None
    ) -> Path:
        """
        Генерация JSON отчёта.

        Args:
            metrics_list: Список метрик моделей
            output_path: Путь для сохранения

        Returns:
            Путь к сохранённому файлу
        """
        if output_path is None:
            output_path = self.reports_dir / "all_metrics.json"

        # Подготовка данных
        models_dict: dict[str, Any] = {}
        report_data: dict[str, Any] = {
            "generated_at": datetime.now().isoformat(),
            "total_models": len(metrics_list),
            "models": models_dict,
        }

        best_accuracy = 0.0
        best_model = None

        for item in metrics_list:
            model_name = item["model"]
            metrics = item["metrics"]
            models_dict[model_name] = metrics

            accuracy = metrics.get("accuracy", 0)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_name

        if best_model:
            report_data["best_model"] = {
                "name": best_model,
                "accuracy": best_accuracy,
            }

        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"JSON report saved to {output_path}")
        return output_path

    def update_docs_reports(self, metrics_list: list[dict[str, Any]]) -> None:
        """
        Обновление отчётов в документации.

        Args:
            metrics_list: Список метрик моделей
        """
        docs_reports_dir = Path("docs/reports")
        docs_reports_dir.mkdir(parents=True, exist_ok=True)

        # Генерация контента для experiments.md
        content = self.generate_markdown_report(metrics_list)

        # Обновление experiments.md (только секция с данными)
        experiments_path = docs_reports_dir / "experiments_generated.md"
        with open(experiments_path, "w") as f:
            f.write(content)

        logger.info(f"Updated docs report at {experiments_path}")

    def run(self) -> dict[str, Any]:
        """
        Запуск полной генерации отчётов.

        Returns:
            Словарь с путями к созданным отчётам
        """
        logger.info("Starting report generation...")

        # Сбор метрик
        metrics_list = self.collect_metrics()
        logger.info(f"Found {len(metrics_list)} experiment results")

        # Генерация отчётов
        md_path = self.reports_dir / "experiment_report.md"
        md_content = self.generate_markdown_report(metrics_list)
        with open(md_path, "w") as f:
            f.write(md_content)
        logger.info(f"Markdown report: {md_path}")

        csv_path = self.generate_csv_report(metrics_list)
        json_path = self.generate_json_report(metrics_list)

        # Обновление документации
        self.update_docs_reports(metrics_list)

        results = {
            "markdown_report": str(md_path),
            "csv_report": str(csv_path),
            "json_report": str(json_path),
            "models_found": len(metrics_list),
        }

        logger.info("Report generation complete!")
        return results


def main() -> None:
    """Точка входа для генерации отчётов."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    generator = ReportGenerator()
    results = generator.run()

    print("\n" + "=" * 60)
    print("Отчёты сгенерированы:")
    print("=" * 60)
    for key, value in results.items():
        print(f"  {key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    main()
