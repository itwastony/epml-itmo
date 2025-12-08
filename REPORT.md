# Отчет по ДЗ 2: Версионирование данных и моделей

## Инструменты

- **Версионирование данных**: DVC (Data Version Control)
- **Версионирование моделей**: MLflow
- **Удаленное хранилище (Remote Storage)**: Local Storage (эмуляция remote)

## Настройка DVC

1. **Инициализация DVC**:
   ```bash
   dvc init
   ```

2. **Настройка Remote Storage**:
   Использована локальная директория `../dvc_remote` для имитации удаленного хранилища.
   ```bash
   mkdir -p ../dvc_remote
   dvc remote add -d localremote ../dvc_remote
   dvc config core.analytics false
   ```

3. **Версионирование данных и пайплайн**:
   - Датасет Wine Quality отслеживается (`data/raw/winequality-red.csv.dvc`).
   - Настроен DVC пайплайн (`dvc.yaml`) с этапами `prepare` и `train`.
   
   ```bash
   # Добавление данных
   dvc add data/raw/winequality-red.csv
   dvc push
   
   # Запуск пайплайна
   dvc repro
   ```

## Настройка MLflow

MLflow настроен для трекинга экспериментов и реестра моделей.

1. **Запуск сервера (опционально) или локальный трекинг**:
   В данном проекте используется локальный трекинг в директорию `mlruns`.

2. **Обучение и логирование**:
   Скрипт `src/models/train_model.py` обучает RandomForest и логирует параметры, метрики и модель.
   
   Пример запуска:
   ```bash
   poetry run python src/models/train_model.py data/processed
   ```

   Пример запуска с другими гиперпараметрами (версия 2):
   ```bash
   poetry run python src/models/train_model.py data/processed --n_estimators 200 --max_depth 10
   ```

## Результаты

### Логи запуска (Screenshots emulation)

**Запуск 1 (Default params):**
```text
2025-12-08 22:12:27,045 - __main__ - INFO - Training model...
2025/12/08 22:12:27 INFO mlflow.tracking.fluent: Experiment with name 'wine_quality_experiment' does not exist. Creating a new experiment.
2025-12-08 22:12:28,243 - __main__ - INFO - Accuracy: 0.659375
2025-12-08 22:12:28,243 - __main__ - INFO - F1 Score: 0.6442498546491976
Successfully registered model 'WineQualityRandomForest'.
Created version '1' of model 'WineQualityRandomForest'.
```

**Запуск 2 (Tuned params):**
```text
2025-12-08 22:12:56,617 - __main__ - INFO - Training model...
2025-12-08 22:12:57,279 - __main__ - INFO - Accuracy: 0.646875
2025-12-08 22:12:57,279 - __main__ - INFO - F1 Score: 0.6266469214465146
Registered model 'WineQualityRandomForest' already exists. Creating a new version of this model...
Created version '2' of model 'WineQualityRandomForest'.
```

## Воспроизводимость

Для обеспечения воспроизводимости используются:
1. **DVC** для данных (`dvc.lock` / `.dvc` файлы).
2. **Poetry** для зависимостей (`poetry.lock`).
3. **Git** для кода.

### Инструкция по воспроизведению

1. **Клонировать репозиторий и перейти в ветку**:
   ```bash
   git checkout HW2
   ```

2. **Установить зависимости**:
   ```bash
   poetry install
   ```

3. **Получить данные (DVC)**:
   ```bash
   poetry run dvc pull
   ```
   *Примечание: Так как remote локальный (`../dvc_remote`), он должен существовать на машине. В реальном проекте это был бы S3 bucket.*

4. **Запустить обучение (через DVC Pipeline)**:
   Это автоматически запустит подготовку данных (`prepare`) и обучение (`train`).
   ```bash
   poetry run dvc repro
   ```

   *Альтернативно (вручную)*:
   ```bash
   poetry run python src/data/make_dataset.py data/raw data/processed
   poetry run python src/models/train_model.py data/processed
   ```

5. **Просмотр результатов MLflow**:
   ```bash
   poetry run mlflow ui
   ```

## Docker

Docker образ собирается с помощью команды:
```bash
docker build -t epml-hw2 .
```

Запуск контейнера:
```bash
docker run -it epml-hw2 bash
```
Внутри контейнера можно выполнить скрипты обучения (при наличии данных или настроенном доступе к remote).
