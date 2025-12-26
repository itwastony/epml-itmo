# Конфигурация

Проект использует [Hydra](https://hydra.cc/) для управления конфигурациями.

## Структура конфигураций

```
conf/
├── config.yaml          # Главный конфиг
├── data/
│   └── default.yaml     # Настройки данных
├── model/
│   ├── random_forest.yaml
│   ├── gradient_boosting.yaml
│   ├── logistic_regression.yaml
│   ├── svm.yaml
│   ├── decision_tree.yaml
│   └── knn.yaml
├── training/
│   └── default.yaml     # Настройки обучения
└── clearml/
    └── default.yaml     # Настройки ClearML
```

## Главный конфиг

```yaml title="conf/config.yaml"
defaults:
  - model: random_forest
  - data: default
  - training: default
  - clearml: default
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

## Конфигурации моделей

### Random Forest

```yaml title="conf/model/random_forest.yaml"
name: RandomForest
_target_: sklearn.ensemble.RandomForestClassifier

params:
  n_estimators: 100
  max_depth: 10
  min_samples_split: 2
  min_samples_leaf: 1
  random_state: ${seed}
```

### Gradient Boosting

```yaml title="conf/model/gradient_boosting.yaml"
name: GradientBoosting
_target_: sklearn.ensemble.GradientBoostingClassifier

params:
  n_estimators: 100
  max_depth: 5
  learning_rate: 0.1
  random_state: ${seed}
```

## Переопределение параметров

Hydra позволяет переопределять параметры из командной строки:

```bash
# Изменение модели
python -m src.pipelines.train_pipeline model=svm

# Изменение параметров модели
python -m src.pipelines.train_pipeline model.params.n_estimators=200

# Изменение нескольких параметров
python -m src.pipelines.train_pipeline \
  model=random_forest \
  model.params.n_estimators=150 \
  model.params.max_depth=15

# Multirun - запуск с разными параметрами
python -m src.pipelines.train_pipeline --multirun \
  model=random_forest,gradient_boosting,svm
```

## Конфигурация обучения

```yaml title="conf/training/default.yaml"
cv_folds: 5
test_size: 0.2
register_model: true
```

## Конфигурация ClearML

```yaml title="conf/clearml/default.yaml"
server:
  api_host: "http://localhost:8008"
  web_host: "http://localhost:8080"
  files_host: "http://localhost:8081"

project:
  name: "EPML-ITMO/Wine-Quality"

experiment:
  auto_connect_frameworks: true
  offline_mode: false
```

## Переменные окружения

Для ClearML можно использовать переменные окружения:

```bash
export CLEARML_API_HOST=https://api.clear.ml
export CLEARML_WEB_HOST=https://app.clear.ml
export CLEARML_FILES_HOST=https://files.clear.ml
export CLEARML_API_ACCESS_KEY=<your_key>
export CLEARML_API_SECRET_KEY=<your_secret>
```

