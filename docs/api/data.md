# Data Module

Модуль для работы с данными.

## make_dataset

```python
from src.data.make_dataset import prepare_data, load_data
```

### prepare_data

Подготавливает данные: загружает, обрабатывает и разбивает на train/test.

```python
def prepare_data(
    input_path: str,
    output_path: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> None:
    """
    Подготовка данных для обучения.
    
    Args:
        input_path: Путь к исходным данным (CSV)
        output_path: Путь для сохранения обработанных данных
        test_size: Доля тестовой выборки (0.0-1.0)
        random_state: Seed для воспроизводимости
    
    Returns:
        None. Создаёт файлы train.csv и test.csv
    
    Example:
        >>> prepare_data("data/raw/wine.csv", "data/processed")
    """
```

### load_data

Загружает обработанные данные.

```python
def load_data(
    processed_path: str = "data/processed"
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Загрузка обработанных данных.
    
    Args:
        processed_path: Путь к папке с обработанными данными
    
    Returns:
        Tuple[X_train, y_train, X_test, y_test]
    
    Example:
        >>> X_train, y_train, X_test, y_test = load_data()
        >>> print(f"Training samples: {len(X_train)}")
    """
```

## Пример использования

```python
from src.data.make_dataset import prepare_data
import pandas as pd

# Подготовка данных
prepare_data(
    input_path="data/raw/winequality-red.csv",
    output_path="data/processed",
    test_size=0.2
)

# Загрузка данных
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
```

