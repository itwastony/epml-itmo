# ДЗ 4: Автоматизация ML пайплайнов

## Обзор

В данном домашнем задании реализована полная автоматизация ML пайплайнов с использованием:
- **DVC Pipelines** — для оркестрации пайплайнов и версионирования данных
- **Hydra** — для управления конфигурациями

### Выбор инструментов

**DVC Pipelines** выбран как инструмент оркестрации по следующим причинам:
- Уже использовался в проекте для версионирования данных
- Отлично интегрируется с Git workflow
- Поддерживает кэширование и параллельное выполнение
- Позволяет отслеживать метрики и параметры экспериментов

**Hydra** выбран для управления конфигурациями:
- Иерархическая композиция конфигураций
- Поддержка переопределения параметров из командной строки
- Интерполяция переменных между конфигурациями
- Автоматическое создание директорий для выходных данных

---

## 1. Настройка DVC Pipelines (4 балла)

### 1.1 Структура пайплайна

Реализован многоэтапный ML пайплайн со следующими стадиями:

```
prepare → train_random_forest ─────┐
        → train_gradient_boosting ─┤
        → train_logistic_regression┼→ evaluate
        → train_svm ───────────────┤
        → train_decision_tree ─────┤
        → train_knn ───────────────┘
```

### 1.2 Конфигурация DVC (`dvc.yaml`)

```yaml
stages:
  prepare:
    cmd: python -m src.pipelines.prepare_data
    deps:
      - src/pipelines/prepare_data.py
      - conf/data/default.yaml
    params:
      - conf/config.yaml:
          - seed
      - conf/data/default.yaml:
          - test_size
    outs:
      - data/processed:
          cache: true

  train_random_forest:
    cmd: python -m src.pipelines.train_pipeline model=random_forest
    deps:
      - data/processed
      - src/pipelines/train_pipeline.py
      - conf/model/random_forest.yaml
    params:
      - conf/model/random_forest.yaml:
          - params
    metrics:
      - outputs/randomforest/metrics.json:
          cache: false
  # ... аналогично для остальных моделей

  evaluate:
    cmd: python -m src.pipelines.evaluate_models
    deps:
      - outputs/randomforest/metrics.json
      - outputs/gradientboosting/metrics.json
      # ... остальные метрики
    metrics:
      - outputs/comparison/best_model.json
    plots:
      - outputs/comparison/metrics_comparison.csv
```

### 1.3 Зависимости между этапами

DVC автоматически определяет зависимости через:
- **deps** — файлы, от которых зависит стадия
- **outs** — выходные файлы стадии
- **params** — параметры из конфигурационных файлов

### 1.4 Кэширование и параллельное выполнение

- **Кэширование**: DVC кэширует все выходные файлы (`cache: true`), что позволяет пропускать стадии при повторных запусках
- **Параллельное выполнение**: Стадии обучения моделей могут выполняться параллельно, т.к. зависят только от `prepare`

Запуск с параллелизацией:
```bash
dvc repro --parallel
```

### 1.5 Визуализация DAG

```
            +---------+
            | prepare |
            +---------+
                |
    ┌───────────┼───────────┬───────────┬───────────┬───────────┐
    ↓           ↓           ↓           ↓           ↓           ↓
train_rf  train_gb    train_lr    train_svm  train_dt   train_knn
    │           │           │           │           │           │
    └───────────┴───────────┴─────┬─────┴───────────┴───────────┘
                                  ↓
                            +----------+
                            | evaluate |
                            +----------+
```

---

## 2. Настройка Hydra (3 балла)

### 2.1 Структура конфигураций

```
conf/
├── config.yaml          # Главный файл конфигурации
├── data/
│   └── default.yaml     # Конфигурация данных
├── model/
│   ├── random_forest.yaml
│   ├── gradient_boosting.yaml
│   ├── logistic_regression.yaml
│   ├── svm.yaml
│   ├── decision_tree.yaml
│   └── knn.yaml
└── training/
    └── default.yaml     # Конфигурация обучения
```

### 2.2 Главный файл конфигурации (`conf/config.yaml`)

```yaml
defaults:
  - model: random_forest
  - data: default
  - training: default
  - _self_

mlflow:
  tracking_uri: "file://${hydra:runtime.cwd}/mlruns"
  experiment_name: "wine_quality_hydra"

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

seed: 42
output_dir: "outputs"
```

### 2.3 Композиция конфигураций

Hydra автоматически объединяет конфигурации из разных файлов:

```yaml
# conf/model/random_forest.yaml
name: "RandomForest"
_target_: "sklearn.ensemble.RandomForestClassifier"
params:
  n_estimators: 100
  max_depth: 10
  random_state: ${seed}  # Интерполяция из главного конфига
```

### 2.4 Валидация конфигураций

Реализована валидация с использованием Pydantic (`src/config/schemas.py`):

```python
class ModelConfig(BaseModel):
    name: str = Field(..., description="Model name")
    _target_: str = Field(..., description="Full path to model class")
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v
```

### 2.5 Переопределение параметров

```bash
# Изменение модели
python -m src.pipelines.train_pipeline model=gradient_boosting

# Изменение гиперпараметров
python -m src.pipelines.train_pipeline model.params.n_estimators=200

# Multirun для нескольких моделей
python -m src.pipelines.train_pipeline --multirun model=random_forest,gradient_boosting
```

---

## 3. Интеграция и тестирование (2 балла)

### 3.1 Интеграция DVC + Hydra

Пайплайн интегрирует оба инструмента:
1. **DVC** управляет порядком выполнения и кэшированием
2. **Hydra** управляет конфигурациями для каждого этапа

### 3.2 Мониторинг выполнения

Реализован модуль мониторинга (`src/pipelines/monitoring.py`):

```python
class PipelineMonitor:
    def start_pipeline(self) -> None
    def end_pipeline(self, success: bool, error: str = None) -> dict
    def start_stage(self, stage_name: str) -> None
    def end_stage(self, stage_name: str, success: bool) -> None
    def save_report(self) -> Path
```

Пример вывода мониторинга:
```
============================================================
PIPELINE COMPLETED
Status: SUCCESS
Total duration: 3.77s
----------------------------------------
  ✓ data_loading: 0.01s
  ✓ model_creation: 0.57s
  ✓ training: 0.61s
  ✓ mlflow_logging: 2.58s
  ✓ save_results: 0.00s
============================================================
```

### 3.3 Уведомления о результатах

Система уведомлений логирует результаты в файл:
- `outputs/{model}/notifications.log` — лог уведомлений
- `outputs/{model}/pipeline_report.json` — детальный отчет

### 3.4 Тестирование воспроизводимости

Воспроизводимость обеспечивается через:
1. **Фиксированный seed** (`seed: 42` в конфигурации)
2. **DVC версионирование** данных и конфигураций
3. **MLflow tracking** для логирования экспериментов

Команда для воспроизведения:
```bash
# Клонирование репозитория
git clone <repo_url>
cd epml_itmo

# Установка зависимостей
poetry install

# Запуск полного пайплайна
dvc repro
```

---

## 4. Результаты

### 4.1 Сравнение моделей

| Model              | Accuracy | Precision | Recall  | F1 Score |
|--------------------|----------|-----------|---------|----------|
| GradientBoosting   | 0.6500   | 0.6394    | 0.6500  | **0.6393** |
| RandomForest       | 0.6438   | 0.6108    | 0.6438  | 0.6240   |
| DecisionTree       | 0.5531   | 0.5320    | 0.5531  | 0.5409   |
| LogisticRegression | 0.5719   | 0.5245    | 0.5719  | 0.5382   |
| SVM                | 0.5094   | 0.5645    | 0.5094  | 0.4618   |
| KNN                | 0.4562   | 0.4223    | 0.4562  | 0.4299   |

**Лучшая модель**: GradientBoosting с F1 Score = 0.6393

### 4.2 DVC Metrics

```bash
$ dvc metrics show
```

![DVC Metrics](reports/figures/dvc_metrics.png)

### 4.3 DVC DAG

```bash
$ dvc dag
```

```
                     +---------+
       **************| prepare |******************
      *              +---------+                  *
     *                   |                         *
    *    ┌───────────────┼───────────────┐          *
   *     ↓               ↓               ↓           *
+--------+  +------------+  +-------------+  +-----+  +--------+  +-----+
|train_rf|  |train_gb    |  |train_lr     |  |svm  |  |train_dt|  |knn  |
+--------+  +------------+  +-------------+  +-----+  +--------+  +-----+
     *           *               *              *          *         *
      *          *               *              *          *        *
       **********+-------+-------+--------------+----------+********
                         ↓
                   +----------+
                   | evaluate |
                   +----------+
```

---

## 5. Команды для воспроизведения

### Быстрый старт

```bash
# 1. Установка зависимостей
poetry install

# 2. Запуск полного пайплайна
make pipeline

# 3. Просмотр метрик
make metrics
```

### Отдельные команды

```bash
# Подготовка данных
make prepare

# Обучение конкретной модели
make train MODEL=random_forest

# Обучение всех моделей
make train_all

# Оценка и сравнение
make evaluate

# Просмотр DAG
make dag

# Очистка и перезапуск
make run_full
```

### Параметры Hydra

```bash
# Изменение числа деревьев в Random Forest
python -m src.pipelines.train_pipeline model=random_forest model.params.n_estimators=200

# Изменение глубины
python -m src.pipelines.train_pipeline model=random_forest model.params.max_depth=15

# Мультизапуск
python -m src.pipelines.train_pipeline --multirun model=random_forest,gradient_boosting
```

---

## 6. Структура проекта

```
epml_itmo/
├── conf/                           # Hydra конфигурации
│   ├── config.yaml                 # Главный конфиг
│   ├── data/default.yaml           # Конфиг данных
│   ├── model/                      # Конфиги моделей
│   │   ├── random_forest.yaml
│   │   ├── gradient_boosting.yaml
│   │   ├── logistic_regression.yaml
│   │   ├── svm.yaml
│   │   ├── decision_tree.yaml
│   │   └── knn.yaml
│   └── training/default.yaml       # Конфиг обучения
├── src/
│   ├── config/                     # Pydantic схемы валидации
│   │   ├── __init__.py
│   │   └── schemas.py
│   └── pipelines/                  # Скрипты пайплайнов
│       ├── __init__.py
│       ├── prepare_data.py         # Подготовка данных
│       ├── train_pipeline.py       # Обучение модели
│       ├── evaluate_models.py      # Оценка моделей
│       ├── run_all_models.py       # Запуск всех моделей
│       └── monitoring.py           # Мониторинг
├── dvc.yaml                        # DVC pipeline конфигурация
├── dvc.lock                        # DVC lock file
├── Makefile                        # Команды Make
├── outputs/                        # Выходные данные
│   ├── randomforest/
│   ├── gradientboosting/
│   ├── ...
│   └── comparison/
└── mlruns/                         # MLflow артефакты
```

---

## Заключение

В рамках данного домашнего задания реализована полная автоматизация ML пайплайнов:

1. **DVC Pipelines** обеспечивает:
   - Автоматическое определение зависимостей
   - Кэширование результатов
   - Параллельное выполнение
   - Отслеживание метрик и параметров

2. **Hydra** обеспечивает:
   - Иерархическую композицию конфигураций
   - Валидацию через Pydantic
   - Переопределение параметров из CLI
   - Интерполяцию переменных

3. **Интеграция**:
   - Мониторинг выполнения с детальными отчетами
   - Уведомления о результатах
   - Полная воспроизводимость экспериментов

Все результаты воспроизводимы через команду `dvc repro`.

