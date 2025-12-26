# EPML ITMO Project

Data Science Project for EPML ITMO.

## Project Organization

```
├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration and dependencies.
├── poetry.lock        <- Locked dependency versions.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
```

## Getting Started

### Prerequisites

- Python 3.12+
- Poetry (for dependency management)

### Installation

1. Clone the repository
2. Install dependencies with Poetry:

```bash
poetry install
```

3. Activate the virtual environment:

```bash
poetry shell
```

### Code Quality

This project uses `ruff`, `mypy`, and `bandit` for code quality.

Run linters:

```bash
poetry run ruff check .
poetry run mypy .
poetry run bandit -r src
```

Pre-commit hooks are configured to run automatically on commit.

### Development Workflow

This project follows a simplified Git Flow:
- `main`: Stable releases. **Direct commits are disabled.**
- `develop`: Main integration branch.
- `feature/name`: New features, branched from `develop`.
- `fix/name`: Bug fixes, branched from `develop`.

To start a new feature:
```bash
git checkout develop
git pull
git checkout -b feature/my-new-feature
```

When finished, open a Pull Request to `develop`.

### Docker

Build the docker image:

```bash
docker build -t epml-itmo .
```
