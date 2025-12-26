# Сравнение моделей

## Визуализация сравнения

### ClearML сравнение экспериментов

![ClearML Comparison](../assets/images/clearml_comparison.jpg)
*Интерактивное сравнение метрик в ClearML Web UI*

### MLflow сравнение по моделям

![MLflow Scatter Plot](../assets/images/mlflow_exps_2.jpg)
*Scatter plot accuracy по типам моделей в MLflow*

## Детальное сравнение

### Таблица метрик

| Модель | Accuracy | Precision (weighted) | Recall (weighted) | F1 (weighted) | Precision (macro) | Recall (macro) | F1 (macro) |
|--------|----------|---------------------|-------------------|---------------|-------------------|----------------|------------|
| GradientBoosting | 0.6469 | 0.6606 | 0.6469 | 0.6402 | 0.4974 | 0.3408 | 0.3563 |
| RandomForest | 0.6438 | 0.6108 | 0.6438 | 0.6240 | 0.3167 | 0.3107 | 0.3108 |
| LogisticRegression | 0.5719 | 0.5245 | 0.5719 | 0.5382 | 0.2561 | 0.2492 | 0.2427 |
| DecisionTree | 0.5531 | 0.5320 | 0.5531 | 0.5409 | 0.2809 | 0.2820 | 0.2805 |
| SVM | 0.5094 | 0.5645 | 0.5094 | 0.4618 | 0.3432 | 0.2098 | 0.1933 |
| KNN | 0.4562 | 0.4223 | 0.4562 | 0.4299 | 0.2074 | 0.1995 | 0.1947 |

## Анализ по типам моделей

### Ансамблевые методы

**Gradient Boosting** и **Random Forest** показывают лучшие результаты:

| Метрика | Gradient Boosting | Random Forest | Разница |
|---------|-------------------|---------------|---------|
| Accuracy | 0.6469 | 0.6438 | +0.31% |
| F1 Score | 0.6402 | 0.6240 | +1.62% |

Gradient Boosting немного лучше за счёт последовательного улучшения ошибок.

### Линейные модели

**Logistic Regression** показывает средние результаты (57.19% accuracy),
что указывает на нелинейную природу данных.

### Метрические методы

**KNN** показывает худшие результаты без нормализации данных.
Рекомендуется предварительная нормализация для улучшения.

## Confusion Matrix анализ

### Gradient Boosting (лучшая модель)

```
Predicted:    3    4    5    6    7    8
Actual:
3             0    0    1    0    0    0
4             0    0    8    2    0    0
5            0    2   88   37    3    0
6             0    1   38   86    7    0
7             0    0    6   22   14    0
8             0    0    1    2    2    0
```

**Наблюдения:**

- Хорошо различает классы 5 и 6 (основные)
- Плохо работает с редкими классами (3, 4, 8)
- Смещение к центральным классам

## Время обучения

| Модель | Время обучения |
|--------|----------------|
| KNN | ~0.01s |
| Decision Tree | ~0.02s |
| Logistic Regression | ~0.2s |
| SVM | ~0.1s |
| Random Forest | ~0.2s |
| Gradient Boosting | ~2.0s |

## Рекомендации по выбору модели

### Для production

**Gradient Boosting** - лучший баланс качества и интерпретируемости.

```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
```

### Для быстрого прототипа

**Random Forest** - быстрее обучается, почти такое же качество.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
```

## Возможные улучшения

1. **Feature Engineering**
   - Создание новых признаков
   - Полиномиальные признаки

2. **Балансировка классов**
   - SMOTE
   - Class weights

3. **Гиперпараметры**
   - Grid Search
   - Random Search
   - Optuna

4. **Предобработка**
   - Нормализация для SVM/KNN
   - Удаление выбросов

