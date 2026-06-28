---
name: python-uv
description: This skill should be used when the user asks to work with Python, including "create a Python project", "run Python", "install Python packages", "add a library", "initialize Python", "setup Python environment", "write a Python script", "run a Python script", "Python CLI", or any Python development task. Enforces uv-based workflow for all Python operations.
---

# Python uv Workflow

Use `uv` for all Python project management and execution. Never use bare `python`, `pip`, or `virtualenv` directly — always go through `uv`.

## Project Initialization

To create a new Python project, use `uv init`:

```bash
uv init -p 3.13               # Python 3.13 (default unless specified otherwise)
uv init -p 3.12               # Python 3.12
uv init -p 3.11               # Python 3.11
uv init --lib -p 3.13         # Library project (creates src/ layout)
uv init --app -p 3.13         # Application project (creates flat layout)
```

Ask the user which Python version they want if not specified. Default to 3.13.

## Running Python

To execute Python scripts or modules, always use `uv run`:

```bash
uv run python script.py            # Run a script
uv run python -m module_name       # Run a module
uv run python -c "print('hello')"  # One-liner
uv run ipython                     # Interactive shell (if ipython installed)
uv run pytest                      # Run tests
uv run black .                     # Formatter
uv run ruff check .                # Linter
uv run mypy .                      # Type checker
```

For convenience, `uv run` can execute binaries directly without `python`:

```bash
uv run pytest          # Equivalent to uv run python -m pytest
uv run black .         # Equivalent to uv run python -m black
```

## Managing Dependencies

To add or remove packages, use `uv add` and `uv remove`:

```bash
uv add requests                  # Add a package
uv add "fastapi>=0.100"          # Add with version constraint
uv add --dev pytest black ruff   # Add dev dependencies
uv add --group test pytest       # Add to a dependency group
uv remove requests               # Remove a package
uv sync                          # Sync environment to lockfile
uv sync --no-dev                 # Production sync (skip dev deps)
uv lock                          # Regenerate lockfile
uv lock --upgrade                # Upgrade all deps in lockfile
```

## Common Patterns

### Creating and running a quick script

```bash
uv init -p 3.13 my-script
cd my-script
uv add requests
# Write script to main.py (or any .py file)
uv run python main.py
```

### Adding a dev tool

```bash
uv add --dev pytest black ruff mypy
```

### Running tests

```bash
uv run pytest
uv run pytest -v
uv run pytest --cov
```

### Type checking / Linting / Formatting

```bash
uv run ruff check .
uv run ruff format .
uv run mypy .
```

## Rules

1. Never use `pip install` — always use `uv add`
2. Never use `python` directly — always use `uv run python`
3. Never use `python -m venv` — uv manages venvs automatically
4. Never manually activate a venv — `uv run` handles it
5. When a `pyproject.toml` exists in the project, read it first to understand the project structure before running commands
6. For existing projects that use `pip`/`poetry`/`pipenv`, convert them to `uv` if the user is open to migration
