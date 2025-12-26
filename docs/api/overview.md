# API Reference

## Обзор модулей

Проект состоит из следующих основных модулей:

```
src/
├── data/                    # Работа с данными
│   └── make_dataset.py
├── models/                  # Обучение моделей
│   ├── train_model.py
│   └── predict_model.py
├── pipelines/               # ML пайплайны
│   ├── prepare_data.py
│   ├── train_pipeline.py
│   ├── run_all_models.py
│   ├── evaluate_models.py
│   └── monitoring.py
├── clearml_integration/     # ClearML интеграция
│   ├── experiment_tracker.py
│   ├── model_manager.py
│   ├── pipeline.py
│   ├── dashboard.py
│   └── run_experiments.py
└── visualization/           # Визуализация
    └── visualize.py
```

## Быстрые ссылки

| Модуль | Описание | Документация |
|--------|----------|--------------|
| Data | Подготовка и загрузка данных | [data](data.md) |
| Models | Обучение и предсказание | [models](models.md) |
| Pipelines | ML пайплайны | [pipelines](pipelines.md) |
| ClearML | ClearML интеграция | [clearml](clearml.md) |

## Пример использования

```python
# Подготовка данных
from src.data.make_dataset import prepare_data
prepare_data("data/raw", "data/processed")

# Обучение с пайплайном
from src.pipelines.train_pipeline import TrainingPipeline
pipeline = TrainingPipeline(cfg)
results = pipeline.run()

# ClearML эксперимент
from src.clearml_integration import ClearMLExperiment
with ClearMLExperiment("my_experiment") as exp:
    exp.log_metrics({"accuracy": 0.95})
```

