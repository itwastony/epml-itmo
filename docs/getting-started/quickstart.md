# Быстрый старт

Это руководство поможет вам быстро запустить первый эксперимент.

## 1. Подготовка данных

Данные уже включены в репозиторий. Для подготовки выполните:

```bash
# Подготовка данных через DVC
dvc repro prepare

# Или через Makefile
make prepare
```

Это создаст обработанные файлы в `data/processed/`:

- `train.csv` - обучающая выборка
- `test.csv` - тестовая выборка

## 2. Обучение одной модели

```bash
# Обучение Random Forest
make train_rf

# Или любой другой модели
make train MODEL=gradient_boosting
make train MODEL=logistic_regression
make train MODEL=svm
make train MODEL=decision_tree
make train MODEL=knn
```

## 3. Обучение всех моделей

```bash
# Последовательное обучение всех моделей
make train_all

# Или через DVC пайплайн
dvc repro
```

## 4. Сравнение результатов

```bash
# Оценка и сравнение моделей
make evaluate

# Просмотр метрик DVC
make metrics
```

## 5. Использование ClearML

### Офлайн режим (без сервера)

```bash
make clearml_experiments_offline
```

### С ClearML сервером

```bash
# Настройка (один раз)
clearml-init

# Запуск экспериментов
make clearml_experiments_all

# Сравнение моделей
make clearml_compare_models
```

## 6. Просмотр результатов

После выполнения экспериментов результаты доступны в:

| Путь | Содержимое |
|------|------------|
| `outputs/comparison/` | Сравнительные отчёты |
| `outputs/clearml/models/` | Зарегистрированные модели |
| `outputs/clearml/dashboard/` | Дашборд отчёты |
| `mlruns/` | MLflow эксперименты |

## Пример: Полный воркфлоу

```bash
# 1. Подготовка данных
make prepare

# 2. Запуск всех экспериментов с ClearML
make clearml_experiments_all

# 3. Сравнение моделей
make clearml_compare_models

# 4. Генерация отчёта
make clearml_report

# 5. Генерация документации
make docs
```

## Следующие шаги

- [Конфигурация](configuration.md) - настройка параметров моделей
- [Подготовка данных](../guides/data-preparation.md) - подробнее о данных
- [ClearML интеграция](../guides/clearml-integration.md) - работа с ClearML

