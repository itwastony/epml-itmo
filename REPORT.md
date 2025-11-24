# Project Setup Report

## 1. Project Structure
The project structure was generated using `cookiecutter` with the `drivendata/cookiecutter-data-science` template. This ensures a standard, organized layout for data science projects.

**Key Directories:**
- `src/`: Source code
- `data/`: Data layers (raw, processed, etc.)
- `notebooks/`: Jupyter notebooks
- `models/`: Serialized models

## 2. Dependency Management
**Tool:** Poetry
**Rationale:** Poetry provides robust dependency resolution and virtual environment management, superior to simple `requirements.txt`.

**Configuration (`pyproject.toml`):**
```toml
[project]
name = "epml-itmo"
requires-python = ">=3.12"
dependencies = [
    "pandas (>=2.3.3,<3.0.0)",
    "numpy (>=2.3.5,<3.0.0)",
    "scikit-learn (>=1.7.2,<2.0.0)",
    "ipykernel (>=7.1.0,<8.0.0)"
]

[dependency-groups]
dev = [
    "ruff (>=0.14.6,<0.15.0)",
    "mypy (>=1.18.2,<2.0.0)",
    "bandit (>=1.9.2,<2.0.0)",
    "pre-commit (>=4.5.0,<5.0.0)"
]
```

## 3. Code Quality
The following tools are configured:

1.  **Ruff**: Fast linter and formatter (replaces Black, isort, Flake8).
2.  **MyPy**: Static type checker.
3.  **Bandit**: Security linter.

**Pre-commit Hooks (`.pre-commit-config.yaml`):**
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
        args: [ --fix ]
      - id: ruff-format
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        args: [--ignore-missing-imports]

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: ["-c", "pyproject.toml"]
        additional_dependencies: ["bandit[toml]"]
```

**Linter Execution Proof:**
The linters are active and detecting issues in the generated template code (e.g., unused imports in `conf.py`):
```text
UP009 [*] UTF-8 encoding declaration is unnecessary
 --> docs/conf.py:1:1
  |
1 | # -*- coding: utf-8 -*-
  | ^^^^^^^^^^^^^^^^^^^^^^^
  
F401 [*] `os` imported but unused
  --> docs/conf.py:14:8
   |
14 | import os
   |        ^^
```

## 4. Git Workflow
- **Repository Initialized**: Yes
- **Branching**: `main` branch established.
- **.gitignore**: Configured for Python, Data Science, and IDE files.

## 5. Containerization
**Dockerfile** created based on `python:3.12-slim`, using multi-stage build concepts (installing deps via poetry).

```dockerfile
FROM python:3.12-slim
# ... configuration ...
RUN poetry install --no-root --only main
COPY . .
CMD ["python"]
```
