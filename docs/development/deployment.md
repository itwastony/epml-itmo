# Развёртывание

## Требования к окружению

### Системные требования

- **OS**: Linux, macOS, Windows
- **Python**: 3.12+
- **RAM**: минимум 4GB
- **Disk**: 2GB для зависимостей

### Зависимости

```bash
# Основные
poetry install

# Только production зависимости
poetry install --only main
```

## Локальное развёртывание

### 1. Клонирование и установка

```bash
git clone https://github.com/username/epml_itmo.git
cd epml_itmo

poetry install
poetry shell
```

### 2. Подготовка данных

```bash
# Получение данных через DVC
dvc pull

# Или подготовка из raw данных
make prepare
```

### 3. Обучение моделей

```bash
# Обучение всех моделей
make train_all

# Или через ClearML
make clearml_experiments_offline
```

### 4. Проверка результатов

```bash
make clearml_compare_models
cat outputs/clearml/model_report.md
```

## Docker развёртывание

### Сборка образа

```bash
docker build -t epml-itmo .
```

### Запуск контейнера

```bash
# Интерактивный режим
docker run -it epml-itmo bash

# Запуск экспериментов
docker run epml-itmo make clearml_experiments_offline
```

### Docker Compose

```yaml title="docker-compose.yml"
version: "3.8"

services:
  app:
    build: .
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
    environment:
      - CLEARML_API_HOST=${CLEARML_API_HOST}
      - CLEARML_API_ACCESS_KEY=${CLEARML_API_ACCESS_KEY}
      - CLEARML_API_SECRET_KEY=${CLEARML_API_SECRET_KEY}
```

## CI/CD Pipeline

### GitHub Actions

```yaml title=".github/workflows/train.yml"
name: Train Models

on:
  push:
    branches: [main, develop]
  workflow_dispatch:

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      
      - name: Install Poetry
        run: pip install poetry
      
      - name: Install dependencies
        run: poetry install
      
      - name: Run experiments
        run: poetry run make clearml_experiments_offline
      
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: experiment-results
          path: outputs/
```

## Конфигурация production

### Переменные окружения

```bash
# ClearML (если используется)
CLEARML_API_HOST=https://api.clear.ml
CLEARML_API_ACCESS_KEY=<key>
CLEARML_API_SECRET_KEY=<secret>

# Логирование
LOG_LEVEL=INFO

# Пути
DATA_PATH=/app/data
OUTPUT_PATH=/app/outputs
```

### Настройка логирования

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
```

## Мониторинг

### Метрики

- Время обучения моделей
- Качество предсказаний
- Использование ресурсов

### Логи

```bash
# Просмотр логов пайплайна
cat outputs/<model_name>/notifications.log

# Просмотр отчёта пайплайна
cat outputs/<model_name>/pipeline_report.json
```

## Troubleshooting

### Частые проблемы

**Проблема**: `ModuleNotFoundError`
```bash
# Решение: активировать виртуальное окружение
poetry shell
```

**Проблема**: `FileNotFoundError: data/processed/train.csv`
```bash
# Решение: подготовить данные
make prepare
```

**Проблема**: ClearML connection error
```bash
# Решение: использовать офлайн режим
make clearml_experiments_offline
```

