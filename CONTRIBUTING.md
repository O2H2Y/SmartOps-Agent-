# Contributing

Thanks for your interest in SmartOps.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env
```

Fill in `DASHSCOPE_API_KEY` locally. Never commit `.env`.

## Before opening a pull request

```bash
python -m pytest
ruff check .
```

Keep changes focused, update documentation when behavior changes, and leave
credentials and local runtime data out of commits.
