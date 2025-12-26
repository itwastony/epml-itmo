# Установка

## Требования

- Python 3.12+
- Poetry (менеджер зависимостей)
- Git
- Docker & Docker Compose (опционально, для ClearML Server)

## Шаг 1: Клонирование репозитория

```bash
git clone https://github.com/username/epml_itmo.git
cd epml_itmo
```

## Шаг 2: Установка Poetry

Если Poetry ещё не установлен:

=== "macOS / Linux"

    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```

=== "Windows (PowerShell)"

    ```powershell
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
    ```

## Шаг 3: Установка зависимостей

```bash
# Установка всех зависимостей проекта
poetry install

# Активация виртуального окружения
poetry shell
```

## Шаг 4: Проверка установки

```bash
# Проверка версии Python
python --version

# Проверка доступности модулей
python -c "import sklearn; import pandas; import clearml; print('OK')"
```

## Шаг 5: Настройка Pre-commit hooks

```bash
# Установка pre-commit hooks
pre-commit install
```

## Опционально: Настройка ClearML

### Вариант 1: Облачный ClearML (рекомендуется)

1. Зарегистрируйтесь на [app.clear.ml](https://app.clear.ml)
2. Получите credentials в Settings → Workspace
3. Настройте клиент:

```bash
clearml-init
```

### Вариант 2: Локальный ClearML Server (требуется Docker)

```bash
# Запуск ClearML Server
make clearml_server_start

# Настройка credentials
clearml-init
```

!!! note "Примечание для Apple Silicon"
    На Mac с M1/M2/M3 рекомендуется использовать облачный ClearML,
    так как Docker образы ClearML собраны только для x86.

## Структура зависимостей

```toml
[tool.poetry.dependencies]
python = ">=3.12"
pandas = "^2.3.3"
numpy = "^2.3.5"
scikit-learn = "^1.7.2"
dvc = "^3.64.2"
mlflow = "^3.7.0"
hydra-core = "^1.3.2"
clearml = "^2.1.0"

[tool.poetry.group.dev.dependencies]
ruff = "^0.14.6"
mypy = "^1.18.2"
mkdocs = "^1.6.1"
mkdocs-material = "^9.7.1"
```

## Следующие шаги

После успешной установки переходите к [Быстрому старту](quickstart.md).

