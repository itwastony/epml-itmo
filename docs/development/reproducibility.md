# Воспроизводимость

## Обзор

Проект спроектирован для полной воспроизводимости результатов.

## Ключевые механизмы

### 1. Фиксированный Random Seed

Все операции используют фиксированный seed:

```yaml title="conf/config.yaml"
seed: 42
```

```python
# В коде
import numpy as np
np.random.seed(42)

# В sklearn
model = RandomForestClassifier(random_state=42)
```

### 2. Версионирование данных (DVC)

```bash
# Воспроизведение с теми же данными
dvc pull
dvc repro
```

### 3. Фиксированные зависимости

```bash
# poetry.lock содержит точные версии
poetry install
```

### 4. Конфигурации Hydra

Все параметры задаются через конфигурации:

```bash
# Воспроизведение с теми же параметрами
python -m src.pipelines.train_pipeline model=random_forest
```

## Пошаговое воспроизведение

### Шаг 1: Клонирование

```bash
git clone https://github.com/username/epml_itmo.git
cd epml_itmo
git checkout HW6  # или нужная ветка
```

### Шаг 2: Установка зависимостей

```bash
# Установка Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Установка зависимостей
poetry install

# Активация окружения
poetry shell
```

### Шаг 3: Получение данных

```bash
# Через DVC (если настроено удалённое хранилище)
dvc pull

# Или подготовка из raw данных
dvc repro prepare
```

### Шаг 4: Запуск экспериментов

```bash
# Все эксперименты
make clearml_experiments_offline

# Или отдельные модели
make clearml_experiment MODEL=RandomForest
make clearml_experiment MODEL=GradientBoosting
```

### Шаг 5: Проверка результатов

```bash
# Сравнение моделей
make clearml_compare_models

# Просмотр метрик
cat outputs/clearml/model_report.md
```

## Ожидаемые результаты

При воспроизведении вы должны получить примерно такие метрики:

| Модель | Accuracy | F1 Score |
|--------|----------|----------|
| Gradient Boosting | ~0.65 | ~0.64 |
| Random Forest | ~0.64 | ~0.62 |
| Logistic Regression | ~0.57 | ~0.54 |
| Decision Tree | ~0.55 | ~0.54 |
| SVM | ~0.51 | ~0.46 |
| KNN | ~0.46 | ~0.43 |

!!! note "Вариативность"
    Небольшие отклонения возможны из-за различий в 
    реализации numpy/sklearn между платформами.

## Автоматическая проверка

```bash
# Скрипт проверки воспроизводимости
make verify_reproducibility
```

Этот скрипт:

1. Запускает эксперименты дважды
2. Сравнивает результаты
3. Выводит отчёт о различиях

## Документирование экспериментов

Каждый эксперимент сохраняет:

- `metrics.json` - метрики
- `pipeline_report.json` - отчёт выполнения
- `config.yaml` - использованная конфигурация

```bash
# Просмотр конфигурации эксперимента
cat outputs/<model>/config.yaml
```

## Версионирование моделей

```python
from src.clearml_integration import ClearMLModelManager

manager = ClearMLModelManager()

# Получение конкретной версии
model, metadata = manager.load_model("RandomForest", version=1)

# Просмотр всех версий
versions = manager.get_model_versions("RandomForest")
```

## Чеклист воспроизводимости

- [ ] Установлены правильные версии зависимостей (`poetry install`)
- [ ] Получены данные (`dvc pull` или `make prepare`)
- [ ] Используется правильная ветка Git
- [ ] Seed установлен в конфигурации (`seed: 42`)
- [ ] Эксперименты запущены с правильными параметрами

