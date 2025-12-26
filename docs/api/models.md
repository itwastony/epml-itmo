# Models Module

Модуль для обучения и использования моделей.

## train_model

```python
from src.models.train_model import train_rf
```

### train_rf

Обучение Random Forest модели с логированием в MLflow.

```python
@log_experiment(experiment_name="wine_quality_experiment")
def train_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_estimators: int,
    max_depth: int,
) -> None:
    """
    Обучение Random Forest модели.
    
    Args:
        X_train: Признаки обучающей выборки
        y_train: Метки обучающей выборки
        X_test: Признаки тестовой выборки
        y_test: Метки тестовой выборки
        n_estimators: Количество деревьев
        max_depth: Максимальная глубина
    
    Returns:
        None. Логирует результаты в MLflow.
    """
```

## predict_model

```python
from src.models.predict_model import predict
```

### predict

Предсказание с использованием обученной модели.

```python
def predict(
    model_path: str,
    input_data: pd.DataFrame
) -> np.ndarray:
    """
    Предсказание с использованием модели.
    
    Args:
        model_path: Путь к сохранённой модели
        input_data: DataFrame с признаками
    
    Returns:
        np.ndarray: Предсказанные метки
    """
```

## Пример использования

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Загрузка данных
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]
X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

# Обучение
model = RandomForestClassifier(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)

# Предсказание
y_pred = model.predict(X_test)

# Оценка
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
```

