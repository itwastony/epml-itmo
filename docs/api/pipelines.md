# Pipelines Module

Модуль для оркестрации ML пайплайнов.

## TrainingPipeline

Основной класс для запуска пайплайна обучения.

```python
from src.pipelines.train_pipeline import TrainingPipeline
```

### Инициализация

```python
class TrainingPipeline:
    """
    ML Training Pipeline с поддержкой Hydra конфигурации.
    
    Attributes:
        cfg: Hydra конфигурация
        model: Обученная модель
        metrics: Словарь метрик
    """
    
    def __init__(self, cfg: DictConfig):
        """
        Args:
            cfg: Hydra конфигурация
        """
```

### Методы

#### run

```python
def run(self) -> dict[str, Any]:
    """
    Запуск полного пайплайна.
    
    Returns:
        dict с результатами:
            - success: bool
            - metrics: dict[str, float]
            - run_id: str (MLflow run ID)
            - model: str (название модели)
    """
```

#### load_data

```python
def load_data(self) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Загрузка данных.
    
    Returns:
        Tuple[X_train, y_train, X_test, y_test]
    """
```

#### create_model

```python
def create_model(self) -> ClassifierMixin:
    """
    Создание модели из конфигурации.
    
    Returns:
        Экземпляр sklearn классификатора
    """
```

#### train

```python
def train(
    self,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """
    Обучение и оценка модели.
    
    Returns:
        Словарь метрик
    """
```

## Пример использования

```python
from src.pipelines.train_pipeline import TrainingPipeline
from omegaconf import OmegaConf

# Загрузка конфигурации
cfg = OmegaConf.load("conf/config.yaml")

# Переопределение модели
cfg.model = OmegaConf.load("conf/model/random_forest.yaml")

# Запуск пайплайна
pipeline = TrainingPipeline(cfg)
results = pipeline.run()

if results["success"]:
    print(f"Model: {results['model']}")
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"F1 Score: {results['metrics']['f1_score']:.4f}")
else:
    print(f"Error: {results['error']}")
```

## PipelineMonitor

Класс для мониторинга выполнения пайплайна.

```python
from src.pipelines.monitoring import PipelineMonitor

monitor = PipelineMonitor(cfg)
monitor.start_pipeline()

# ... выполнение этапов ...

monitor.start_stage("training")
# ... обучение ...
monitor.end_stage("training", success=True)

report = monitor.end_pipeline(success=True)
monitor.save_report()
```

