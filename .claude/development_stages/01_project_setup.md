# 01 - Project Setup & Infrastructure

## Overview
Initialize the project structure, dependencies, and development tools.

## Subtasks

### 1.1 Poetry Project Initialization
- [ ] Run `poetry init` to create `pyproject.toml`
- [ ] Set project name: `doclint`
- [ ] Set version: `0.1.0`
- [ ] Set description: "Data quality linting for AI knowledge bases"
- [ ] Set Python version requirement: `^3.10`
- [ ] Configure MIT license
- [ ] Set up project metadata (authors, repository URL, etc.)

### 1.2 Core Dependencies Installation
- [ ] Add CLI dependencies: `typer ^0.12.0`, `rich ^13.7.0`, `click ^8.1.7`
- [ ] Add validation: `pydantic ^2.6.0`, `pydantic-settings ^2.2.0`, `tomli ^2.0.1`
- [ ] Add ML dependencies: `sentence-transformers ^2.3.1`, `torch ^2.2.0`, `numpy ^1.26.0`, `scikit-learn ^1.4.0`
- [ ] Add parsing libraries: `pypdf ^4.0.0`, `python-docx ^1.1.0`, `beautifulsoup4 ^4.12.0`, `lxml ^5.1.0`, `markdown ^3.5.0`, `python-magic ^0.4.27`
- [ ] Add caching: `diskcache ^5.6.0`, `platformdirs ^4.2.0`
- [ ] Add async support: `aiofiles ^23.2.0`, `tqdm ^4.66.0`
- [ ] Add utilities: `python-dateutil ^2.8.0`
- [ ] Run `poetry install` to create virtual environment

### 1.3 Development Dependencies
- [ ] Add testing: `pytest ^8.0.0`, `pytest-asyncio ^0.23.0`, `pytest-cov ^4.1.0`, `pytest-mock ^3.12.0`
- [ ] Add code quality: `black ^24.1.0`, `ruff ^0.2.0`, `mypy ^1.8.0`, `pre-commit ^3.6.0`
- [ ] Add documentation: `mkdocs ^1.5.0`, `mkdocs-material ^9.5.0`
- [ ] Run `poetry install --with dev`

### 1.4 Project Directory Structure
- [ ] Create `doclint/` package directory
- [ ] Create `doclint/__init__.py`
- [ ] Create `doclint/__main__.py` (entry point)
- [ ] Create `doclint/version.py`
- [ ] Create `doclint/cli/` directory and `__init__.py`
- [ ] Create `doclint/core/` directory and `__init__.py`
- [ ] Create `doclint/parsers/` directory and `__init__.py`
- [ ] Create `doclint/embeddings/` directory and `__init__.py`
- [ ] Create `doclint/detectors/` directory and `__init__.py`
- [ ] Create `doclint/reporters/` directory and `__init__.py`
- [ ] Create `doclint/cache/` directory and `__init__.py`
- [ ] Create `doclint/utils/` directory and `__init__.py`
- [ ] Create `tests/` directory and `__init__.py`
- [ ] Create `tests/conftest.py`
- [ ] Create `docs/` directory
- [ ] Create `examples/` directory

### 1.5 Configuration Files
- [ ] Create `.gitignore` (Python, venv, cache, IDE files)
- [ ] Create `.pre-commit-config.yaml` (black, ruff, mypy hooks)
- [ ] Create `mkdocs.yml` for documentation
- [ ] Create `README.md` with project overview
- [ ] Create `LICENSE` file (MIT)
- [ ] Create `.python-version` file (3.10)

### 1.6 Version Management
- [ ] Implement `doclint/version.py`:
  ```python
  __version__ = "0.1.0"
  ```
- [ ] Update `doclint/__init__.py` to export version

### 1.7 Entry Point Setup
- [ ] Implement `doclint/__main__.py` to allow `python -m doclint`
- [ ] Configure Poetry scripts in `pyproject.toml`:
  ```toml
  [tool.poetry.scripts]
  doclint = "doclint.cli.main:app"
  ```

### 1.8 Code Quality Configuration
- [ ] Configure Black in `pyproject.toml`:
  ```toml
  [tool.black]
  line-length = 100
  target-version = ['py310']
  ```
- [ ] Configure Ruff in `pyproject.toml`:
  ```toml
  [tool.ruff]
  line-length = 100
  select = ["E", "F", "I"]
  ```
- [ ] Configure MyPy in `pyproject.toml`:
  ```toml
  [tool.mypy]
  python_version = "3.10"
  strict = true
  ```
- [ ] Configure pytest in `pyproject.toml`:
  ```toml
  [tool.pytest.ini_options]
  testpaths = ["tests"]
  asyncio_mode = "auto"
  ```

### 1.9 Pre-commit Hooks Setup
- [ ] Run `pre-commit install`
- [ ] Test pre-commit hooks with sample file
- [ ] Verify black, ruff, mypy run on commit

### 1.10 Verification
- [ ] Test Poetry environment: `poetry shell`
- [ ] Verify Python version: `python --version`
- [ ] Test package import: `python -c "import doclint"`
- [ ] Run initial test suite: `poetry run pytest`
- [ ] Build package: `poetry build`
- [ ] Test CLI stub: `poetry run doclint --help`

## Success Criteria
- ✅ Poetry environment created and dependencies installed
- ✅ Complete directory structure in place
- ✅ All configuration files created
- ✅ Pre-commit hooks working
- ✅ Package can be imported successfully
- ✅ Initial build succeeds

## Notes
- This sets the foundation for all other components
- Follow the exact structure from the architecture document
- Ensure all paths are correct for Windows development
