# ДЗ 6: Документация и отчёты

## Описание

В рамках данного задания была создана полная система документации проекта и автоматической генерации отчётов об экспериментах.

## Выполненные требования

### 1. Техническая документация (2 балла) ✅

#### Документация с помощью MkDocs

Создана полная документация проекта с использованием **MkDocs** и темы **Material for MkDocs**.

**Конфигурация MkDocs** (`mkdocs.yml`):

```yaml
site_name: EPML ITMO - Wine Quality Classification
theme:
  name: material
  language: ru
  palette:
    - scheme: default
      primary: deep purple
      accent: amber
  features:
    - navigation.tabs
    - navigation.sections
    - search.suggest
    - content.code.copy
plugins:
  - search
  - mkdocstrings  # Автогенерация API документации
```

#### Руководство по развёртыванию

Создано в `docs/development/deployment.md`:

- Системные требования
- Локальное развёртывание
- Docker развёртывание
- CI/CD Pipeline
- Конфигурация production
- Troubleshooting

#### Автоматическая генерация документации

- **mkdocstrings** - автоматическая генерация документации из docstrings
- **GitHub Actions** - автоматическая сборка при изменениях

```bash
# Локальная сборка
make docs_build

# Локальный просмотр
make docs_serve
```

#### Примеры использования

Созданы во всех руководствах:

- `docs/getting-started/quickstart.md` - быстрый старт
- `docs/guides/data-preparation.md` - подготовка данных
- `docs/guides/model-training.md` - обучение моделей
- `docs/guides/clearml-integration.md` - интеграция ClearML

### 2. Публикация в Git Pages (3 балла) ✅

#### GitHub Actions для автоматической публикации

Создан workflow `.github/workflows/docs.yml`:

```yaml
name: Deploy Documentation

on:
  push:
    branches: [main, HW6]
    paths:
      - 'docs/**'
      - 'mkdocs.yml'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install mkdocs mkdocs-material mkdocstrings mkdocstrings-python
      - run: mkdocs build --strict
      - uses: actions/upload-pages-artifact@v3

  deploy:
    uses: actions/deploy-pages@v4
```

#### Сайт с документацией

После push в main/HW6 документация автоматически публикуется на GitHub Pages.

**Структура документации:**

```
docs/
├── index.md                    # Главная страница
├── getting-started/
│   ├── installation.md         # Установка
│   ├── quickstart.md           # Быстрый старт
│   └── configuration.md        # Конфигурация
├── guides/
│   ├── data-preparation.md     # Подготовка данных
│   ├── model-training.md       # Обучение моделей
│   ├── clearml-integration.md  # ClearML интеграция
│   └── experiments.md          # Эксперименты
├── api/
│   ├── overview.md             # Обзор API
│   ├── data.md                 # Data module
│   ├── models.md               # Models module
│   ├── pipelines.md            # Pipelines module
│   └── clearml.md              # ClearML module
├── reports/
│   ├── experiments.md          # Результаты экспериментов
│   └── model-comparison.md     # Сравнение моделей
└── development/
    ├── deployment.md           # Развёртывание
    ├── reproducibility.md      # Воспроизводимость
    └── contributing.md         # Участие в проекте
```

#### Автоматическое обновление при изменениях

Workflow запускается автоматически при:

- Push в ветки `main` или `HW6`
- Изменениях в папке `docs/` или файле `mkdocs.yml`
- Ручном запуске через `workflow_dispatch`

### 3. Отчёты об экспериментах (2 балла) ✅

#### Отчёты в формате Markdown

Создан модуль `src/reports/generate_reports.py`:

```python
class ReportGenerator:
    def collect_metrics(self) -> list[dict[str, Any]]:
        """Сбор метрик из всех экспериментов."""
    
    def create_comparison_table(self, metrics_list) -> pd.DataFrame:
        """Создание таблицы сравнения моделей."""
    
    def generate_markdown_report(self, metrics_list) -> str:
        """Генерация полного Markdown отчёта."""
    
    def generate_csv_report(self, metrics_list) -> Path:
        """Генерация CSV отчёта."""
    
    def generate_json_report(self, metrics_list) -> Path:
        """Генерация JSON отчёта."""
```

#### Графики и визуализации

ASCII-визуализации метрик в отчётах:

```
Gradient Boosting  ██████████████████████████████ 64.69%
Random Forest      █████████████████████████████░ 64.38%
Logistic Regr.     ███████████████████████████░░░ 57.19%
Decision Tree      ██████████████████████████░░░░ 55.31%
SVM                ████████████████████████░░░░░░ 50.94%
KNN                ██████████████████████░░░░░░░░ 45.62%
```

#### Сравнительные таблицы экспериментов

Автоматически генерируемые таблицы:

| Модель | Accuracy | Precision | Recall | F1 Score |
|--------|----------|-----------|--------|----------|
| GradientBoosting | 0.6469 | 0.6606 | 0.6469 | 0.6402 |
| RandomForest | 0.6438 | 0.6108 | 0.6438 | 0.6240 |
| LogisticRegression | 0.5719 | 0.5245 | 0.5719 | 0.5382 |
| DecisionTree | 0.5531 | 0.5320 | 0.5531 | 0.5409 |
| SVM | 0.5094 | 0.5645 | 0.5094 | 0.4618 |
| KNN | 0.4562 | 0.4223 | 0.4562 | 0.4299 |

#### Автоматическая генерация отчётов

```bash
# Генерация отчётов
make generate_reports

# Результаты:
# outputs/reports/
#   ├── experiment_report.md     # Markdown отчёт
#   ├── metrics_comparison.csv   # CSV таблица
#   └── all_metrics.json         # JSON данные
```

### 4. Воспроизводимость (1 балл) ✅

#### Инструкции по воспроизведению

Создано в `docs/development/reproducibility.md`:

```bash
# Клонирование
git clone https://github.com/username/epml_itmo.git
cd epml_itmo
git checkout HW6

# Установка
poetry install
poetry shell

# Подготовка данных
dvc repro prepare

# Запуск экспериментов
make clearml_experiments_offline

# Генерация отчётов
make generate_reports

# Просмотр результатов
cat outputs/reports/experiment_report.md
```

#### README с полным описанием

README.md содержит:

- Описание проекта
- Структура проекта
- Быстрый старт
- Команды Makefile
- Ссылки на документацию

#### Автоматическая сборка документации

```bash
# Полный workflow документации
make docs_full

# Проверка воспроизводимости
make verify_reproducibility
```

## Структура проекта

```
epml_itmo/
├── docs/                           # Документация MkDocs
│   ├── index.md                    # Главная страница
│   ├── getting-started/            # Начало работы
│   ├── guides/                     # Руководства
│   ├── api/                        # API Reference
│   ├── reports/                    # Отчёты
│   └── development/                # Разработка
├── src/
│   └── reports/                    # Генерация отчётов
│       ├── __init__.py
│       └── generate_reports.py
├── .github/workflows/
│   ├── docs.yml                    # Публикация документации
│   └── generate-reports.yml        # Генерация отчётов
├── mkdocs.yml                      # Конфигурация MkDocs
└── Makefile                        # Команды автоматизации
```

## Команды Makefile

| Команда | Описание |
|---------|----------|
| `make docs_build` | Сборка документации |
| `make docs_serve` | Локальный просмотр документации |
| `make docs_deploy` | Деплой на GitHub Pages |
| `make generate_reports` | Генерация отчётов |
| `make reports_all` | Полный workflow отчётов |
| `make verify_reproducibility` | Проверка воспроизводимости |
| `make docs_full` | Полный workflow документации |

## Воспроизведение результатов

### Шаг 1: Клонирование и установка

```bash
git clone https://github.com/username/epml_itmo.git
cd epml_itmo
git checkout HW6

poetry install
poetry shell
```

### Шаг 2: Подготовка данных

```bash
dvc repro prepare
```

### Шаг 3: Запуск экспериментов

```bash
make clearml_experiments_offline
```

### Шаг 4: Генерация отчётов

```bash
make generate_reports
```

### Шаг 5: Сборка документации

```bash
make docs_build
```

### Шаг 6: Просмотр документации

```bash
make docs_serve
# Откройте http://127.0.0.1:8000
```

## Зависимости документации

```toml
[tool.poetry.group.dev.dependencies]
mkdocs = "^1.6.1"
mkdocs-material = "^9.7.1"
mkdocstrings = "^1.0.0"
mkdocstrings-python = "^2.0.1"
```

## Автоматизация

### GitHub Actions

1. **docs.yml** - автоматическая публикация документации при push
2. **generate-reports.yml** - генерация отчётов после обучения моделей

### Локальная автоматизация

```bash
# Полный workflow
make docs_full

# Эквивалентно:
make generate_reports
make docs_build
```

## Результаты

✅ **Техническая документация** - полная документация с MkDocs
✅ **Git Pages** - автоматическая публикация через GitHub Actions
✅ **Отчёты** - Markdown, CSV, JSON отчёты с визуализациями
✅ **Воспроизводимость** - инструкции и автоматизация

## Итого: 8 баллов

