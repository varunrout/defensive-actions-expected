# Local workflow

## Environment setup

From the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Run tests

```bash
python -m pytest -q
```

For targeted checks during development:

```bash
python -m pytest -q tests/test_coach_analysis.py
python -m pytest -q tests/test_end_to_end_fixture.py
```

## Run Ruff

```bash
python -m ruff check src scripts tests
```

## Run coach-analysis scripts

After the required local data and OOF artifacts exist, run:

```bash
python scripts/coach_analysis/00_check_coach_analysis_readiness.py
python scripts/coach_analysis/01_analyze_cb_box_defence.py
```

Both scripts accept explicit path arguments. Prefer explicit paths when working outside the default local layout.

## Check generated outputs

Generated coach-analysis outputs are written under local output directories such as:

```text
outputs/coach_analysis/readiness/
outputs/coach_analysis/cb_box_defence/
```

Typical files include markdown reports, execution summary JSON, tables, figures and video-review candidate CSVs. These files are generated artifacts and should not be committed.

## Before opening a PR

Run the standard checks:

```bash
python -m pytest -q
python -m ruff check src scripts tests
```

Then confirm that only intended files are staged:

```bash
git status --short
git diff --stat
git diff --cached --stat
```
