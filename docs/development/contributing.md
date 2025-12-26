# Contributing

## Рабочий процесс

### Git Flow

```
main          ─────────────────────────────────
                        ↑
develop       ──────────┼──────────────────────
                ↑       │       ↑
feature/xxx   ──┴───────┘       │
                                │
fix/xxx       ──────────────────┘
```

### Создание ветки

```bash
# Новая функциональность
git checkout develop
git pull
git checkout -b feature/my-feature

# Исправление бага
git checkout develop
git pull
git checkout -b fix/bug-description
```

### Коммиты

Используйте Conventional Commits:

```
feat: добавить новую модель XGBoost
fix: исправить загрузку данных
docs: обновить документацию API
refactor: реорганизовать структуру пайплайнов
test: добавить тесты для ClearML
```

## Требования к коду

### Стиль кода

Проект использует:

- **ruff** - линтер и форматтер
- **mypy** - проверка типов
- **bandit** - безопасность

```bash
# Проверка кода
poetry run ruff check .
poetry run mypy .
poetry run bandit -r src

# Автоформатирование
poetry run ruff format .
```

### Pre-commit hooks

```bash
# Установка хуков
pre-commit install

# Ручной запуск
pre-commit run --all-files
```

### Типизация

Весь новый код должен быть типизирован:

```python
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any]
) -> tuple[ClassifierMixin, dict[str, float]]:
    """
    Обучение модели.
    
    Args:
        X_train: Признаки
        y_train: Метки
        params: Параметры модели
    
    Returns:
        Tuple[модель, метрики]
    """
    ...
```

## Документация

### Docstrings

Используйте Google style:

```python
def my_function(arg1: str, arg2: int) -> bool:
    """
    Краткое описание функции.
    
    Подробное описание при необходимости.
    
    Args:
        arg1: Описание первого аргумента
        arg2: Описание второго аргумента
    
    Returns:
        Описание возвращаемого значения
    
    Raises:
        ValueError: Когда возникает ошибка
    
    Example:
        >>> my_function("test", 42)
        True
    """
```

### Обновление документации

```bash
# Локальный просмотр
make docs_serve

# Сборка
make docs_build
```

## Тестирование

### Запуск тестов

```bash
# Все тесты
pytest

# С покрытием
pytest --cov=src
```

### Написание тестов

```python
import pytest

def test_my_function():
    result = my_function("test", 42)
    assert result == True

def test_my_function_error():
    with pytest.raises(ValueError):
        my_function("", -1)
```

## Pull Request

### Чеклист

- [ ] Код соответствует стилю проекта
- [ ] Добавлены тесты для нового функционала
- [ ] Документация обновлена
- [ ] Pre-commit хуки пройдены
- [ ] CI/CD пайплайн успешен

### Шаблон PR

```markdown
## Описание

Краткое описание изменений.

## Тип изменения

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation

## Чеклист

- [ ] Тесты пройдены
- [ ] Документация обновлена
```

## Вопросы

Если у вас есть вопросы:

1. Проверьте документацию
2. Поищите в Issues
3. Создайте новый Issue

