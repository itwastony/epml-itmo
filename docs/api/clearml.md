# ClearML Module

Модуль интеграции с ClearML.

## ClearMLExperiment

Класс для трекинга экспериментов.

```python
from src.clearml_integration import ClearMLExperiment
```

### Использование как контекстный менеджер

```python
with ClearMLExperiment(
    experiment_name="my_experiment",
    project_name="EPML-ITMO/Wine-Quality",
    task_type="training",
    tags=["test", "classification"],
    offline_mode=False
) as exp:
    # Логирование параметров
    exp.log_parameters({"n_estimators": 100})
    
    # Логирование метрик
    exp.log_metrics({"accuracy": 0.95})
    
    # Логирование модели
    exp.log_model(model, "my_model")
```

### Методы

| Метод | Описание |
|-------|----------|
| `log_parameters(params, prefix)` | Логирование параметров |
| `log_metric(name, value, series, iteration)` | Логирование одной метрики |
| `log_metrics(metrics, series, iteration)` | Логирование нескольких метрик |
| `log_confusion_matrix(y_true, y_pred, labels)` | Логирование confusion matrix |
| `log_classification_report(y_true, y_pred)` | Полный classification report |
| `log_model(model, model_name, framework)` | Логирование модели |
| `log_artifact(name, artifact)` | Логирование артефакта |
| `log_dataset(train_df, test_df)` | Логирование датасетов |
| `set_comment(comment)` | Добавление комментария |
| `get_task_id()` | Получение ID задачи |

## ClearMLModelManager

Класс для управления моделями.

```python
from src.clearml_integration import ClearMLModelManager

manager = ClearMLModelManager()
```

### Методы

| Метод | Описание |
|-------|----------|
| `register_model(model, model_name, metrics, ...)` | Регистрация модели |
| `load_model(model_name, version)` | Загрузка модели |
| `get_model_versions(model_name)` | Получение версий |
| `compare_models(model_names, metric)` | Сравнение моделей |
| `get_best_model(model_name, metric)` | Лучшая модель |
| `export_model(model_name, version, path)` | Экспорт модели |
| `generate_model_report(output_path)` | Генерация отчёта |

### Пример

```python
manager = ClearMLModelManager()

# Регистрация
model_id = manager.register_model(
    model=trained_model,
    model_name="RandomForest",
    metrics={"accuracy": 0.95, "f1_score": 0.93},
    parameters={"n_estimators": 100},
    tags=["production"]
)

# Сравнение
df = manager.compare_models()
print(df.to_markdown())

# Лучшая модель
best_id, metadata = manager.get_best_model(metric="accuracy")
```

## ClearMLPipeline

Класс для создания ML пайплайнов.

```python
from src.clearml_integration import ClearMLPipeline

pipeline = ClearMLPipeline(
    pipeline_name="Wine-Quality-Pipeline",
    project_name="EPML-ITMO/Wine-Quality/Pipelines",
    version="1.0.0"
)

# Добавление шагов
pipeline.add_data_step(train_path, test_path)
pipeline.add_training_step(model_name, params)
pipeline.add_evaluation_step(metrics)
pipeline.add_model_registration_step()

# Запуск
results = pipeline.run(local_mode=True)
```

## Декоратор clearml_experiment

```python
from src.clearml_integration import clearml_experiment

@clearml_experiment(
    experiment_name="training",
    project_name="EPML-ITMO/Wine-Quality",
    tags=["training"]
)
def train_model(clearml_experiment=None):
    # clearml_experiment автоматически инжектируется
    clearml_experiment.log_metrics({"accuracy": 0.95})
```

