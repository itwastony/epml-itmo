# Подготовка данных

## Описание датасета

Проект использует датасет **Wine Quality** из UCI Machine Learning Repository.

| Характеристика | Значение |
|----------------|----------|
| Тип вина | Красное |
| Количество образцов | 1599 |
| Количество признаков | 11 |
| Целевая переменная | quality (0-10) |

## Признаки

| Признак | Описание | Единицы |
|---------|----------|---------|
| fixed acidity | Фиксированная кислотность | g/dm³ |
| volatile acidity | Летучая кислотность | g/dm³ |
| citric acid | Лимонная кислота | g/dm³ |
| residual sugar | Остаточный сахар | g/dm³ |
| chlorides | Хлориды | g/dm³ |
| free sulfur dioxide | Свободный диоксид серы | mg/dm³ |
| total sulfur dioxide | Общий диоксид серы | mg/dm³ |
| density | Плотность | g/cm³ |
| pH | Кислотность | - |
| sulphates | Сульфаты | g/dm³ |
| alcohol | Содержание алкоголя | % |

## Структура данных

```
data/
├── raw/
│   ├── winequality-red.csv      # Исходные данные
│   └── winequality-red.csv.dvc  # DVC файл
├── interim/                      # Промежуточные данные
└── processed/
    ├── train.csv                 # Обучающая выборка (80%)
    └── test.csv                  # Тестовая выборка (20%)
```

## Подготовка данных

### Через DVC

```bash
# Получение данных из удалённого хранилища
dvc pull

# Запуск пайплайна подготовки
dvc repro prepare
```

### Через Makefile

```bash
make prepare
```

### Программно

```python
from src.data.make_dataset import prepare_data

# Подготовка данных
prepare_data(
    input_path="data/raw/winequality-red.csv",
    output_path="data/processed",
    test_size=0.2,
    random_state=42
)
```

## DVC пайплайн

```yaml title="dvc.yaml"
stages:
  prepare:
    cmd: python -m src.pipelines.prepare_data
    deps:
      - data/raw/winequality-red.csv
      - src/pipelines/prepare_data.py
    outs:
      - data/processed/train.csv
      - data/processed/test.csv
```

## Загрузка данных в коде

```python
import pandas as pd

# Загрузка обучающих данных
train_df = pd.read_csv("data/processed/train.csv")
X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]

# Загрузка тестовых данных
test_df = pd.read_csv("data/processed/test.csv")
X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {X_train.columns.tolist()}")
```

## Распределение классов

Целевая переменная `quality` имеет значения от 3 до 8:

| Quality | Количество | Процент |
|---------|------------|---------|
| 3 | ~10 | 0.6% |
| 4 | ~53 | 3.3% |
| 5 | ~681 | 42.6% |
| 6 | ~638 | 39.9% |
| 7 | ~199 | 12.4% |
| 8 | ~18 | 1.1% |

!!! warning "Несбалансированные классы"
    Датасет имеет дисбаланс классов. Большинство образцов 
    относятся к качеству 5 и 6.

