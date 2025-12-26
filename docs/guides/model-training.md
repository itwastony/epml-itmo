# Обучение моделей

## Доступные модели

| Модель | Конфигурация | Команда |
|--------|--------------|---------|
| Random Forest | `model=random_forest` | `make train_rf` |
| Gradient Boosting | `model=gradient_boosting` | `make train_gb` |
| Logistic Regression | `model=logistic_regression` | `make train MODEL=logistic_regression` |
| SVM | `model=svm` | `make train MODEL=svm` |
| Decision Tree | `model=decision_tree` | `make train MODEL=decision_tree` |
| KNN | `model=knn` | `make train MODEL=knn` |

## Обучение через Makefile

```bash
# Обучение конкретной модели
make train MODEL=random_forest

# Обучение всех моделей
make train_all

# Полный пайплайн (данные + обучение + оценка)
make pipeline
```

## Обучение через Hydra

```bash
# Базовое обучение
python -m src.pipelines.train_pipeline model=random_forest

# С переопределением параметров
python -m src.pipelines.train_pipeline \
    model=random_forest \
    model.params.n_estimators=200 \
    model.params.max_depth=15

# Multirun - несколько моделей
python -m src.pipelines.train_pipeline --multirun \
    model=random_forest,gradient_boosting,svm
```

## Программное обучение

```python
from src.pipelines.train_pipeline import TrainingPipeline
from omegaconf import OmegaConf

# Загрузка конфигурации
cfg = OmegaConf.load("conf/config.yaml")

# Создание и запуск пайплайна
pipeline = TrainingPipeline(cfg)
results = pipeline.run()

print(f"Model: {results['model']}")
print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
print(f"F1 Score: {results['metrics']['f1_score']:.4f}")
```

## Параметры моделей

### Random Forest

```yaml
params:
  n_estimators: 100     # Количество деревьев
  max_depth: 10         # Максимальная глубина
  min_samples_split: 2  # Минимум для разбиения
  min_samples_leaf: 1   # Минимум в листе
  random_state: 42
```

### Gradient Boosting

```yaml
params:
  n_estimators: 100     # Количество итераций
  max_depth: 5          # Глубина деревьев
  learning_rate: 0.1    # Скорость обучения
  random_state: 42
```

### SVM

```yaml
params:
  kernel: rbf          # Ядро (rbf, linear, poly)
  C: 1.0               # Параметр регуляризации
  gamma: scale         # Коэффициент ядра
  random_state: 42
```

## Метрики

После обучения вычисляются следующие метрики:

| Метрика | Описание |
|---------|----------|
| Accuracy | Доля правильных предсказаний |
| Precision | Точность (weighted) |
| Recall | Полнота (weighted) |
| F1 Score | Гармоническое среднее precision и recall |
| CV Accuracy | Кросс-валидационная точность |

## Логирование в MLflow

Все эксперименты автоматически логируются в MLflow:

```bash
# Просмотр экспериментов
mlflow ui

# Откройте http://localhost:5000
```

## Логирование в ClearML

```bash
# С сервером ClearML
make clearml_experiments_all

# Офлайн режим
make clearml_experiments_offline
```

## Результаты

Результаты сохраняются в:

- `outputs/<model_name>/metrics.json` - метрики
- `outputs/<model_name>/pipeline_report.json` - отчёт пайплайна
- `mlruns/` - MLflow эксперименты
- `outputs/clearml/models/` - ClearML модели

