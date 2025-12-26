# Эксперименты

## Запуск экспериментов

### Все модели

```bash
# С ClearML трекингом
make clearml_experiments_all

# Офлайн режим
make clearml_experiments_offline

# Через DVC
dvc repro
```

### Отдельная модель

```bash
# Через Makefile
make clearml_experiment MODEL=RandomForest

# Через Python
python -m src.clearml_integration.run_experiments --model RandomForest
```

## Сравнение экспериментов

```bash
# Сравнение зарегистрированных моделей
make clearml_compare_models

# Генерация отчёта
make clearml_report
```

### Программное сравнение

```python
from src.clearml_integration import ClearMLModelManager

manager = ClearMLModelManager()

# Таблица сравнения
df = manager.compare_models()
print(df.to_markdown())

# Лучшая модель по accuracy
best_id, metadata = manager.get_best_model(metric="accuracy")
print(f"Best: {best_id}, Accuracy: {metadata['metrics']['accuracy']:.4f}")
```

## Просмотр результатов

### Файловая система

```
outputs/
├── clearml/
│   ├── models/              # Модели
│   │   ├── RandomForest/
│   │   ├── GradientBoosting/
│   │   └── ...
│   ├── dashboard/           # Отчёты
│   │   ├── dashboard_report.md
│   │   ├── metrics_export.csv
│   │   └── summary.json
│   └── model_report.md
├── comparison/              # DVC сравнение
│   ├── all_metrics.json
│   ├── best_model.json
│   └── comparison_report.txt
└── <model_name>/           # По каждой модели
    ├── metrics.json
    └── pipeline_report.json
```

### ClearML Web UI

После запуска экспериментов с ClearML сервером:

1. Откройте http://localhost:8080 (или app.clear.ml)
2. Перейдите в Projects → EPML-ITMO → Wine-Quality
3. Посмотрите:
   - Experiments - список экспериментов
   - Models - зарегистрированные модели
   - Compare - сравнение метрик

### MLflow UI

```bash
mlflow ui
# Откройте http://localhost:5000
```

## Воспроизведение эксперимента

```bash
# 1. Клонирование репозитория
git clone https://github.com/username/epml_itmo.git
cd epml_itmo

# 2. Установка зависимостей
poetry install
poetry shell

# 3. Подготовка данных
dvc repro prepare

# 4. Запуск экспериментов
make clearml_experiments_offline

# 5. Просмотр результатов
make clearml_compare_models
cat outputs/clearml/model_report.md
```

## Типичные результаты

| Модель | Accuracy | F1 Score |
|--------|----------|----------|
| Gradient Boosting | ~0.65 | ~0.64 |
| Random Forest | ~0.64 | ~0.62 |
| Logistic Regression | ~0.57 | ~0.54 |
| Decision Tree | ~0.55 | ~0.54 |
| SVM | ~0.51 | ~0.46 |
| KNN | ~0.46 | ~0.43 |

!!! note "Примечание"
    Точные значения могут отличаться в зависимости от 
    случайного разбиения данных.

