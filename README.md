# Project Meezan

Personal decision-support tool for ranking business-idea candidates. Full
spec: [`docs/FRD.md`](docs/FRD.md).

**Status:** scaffolding only (Task 0). Nothing is implemented yet — see
`docs/FRD.md` Section 14 for the task-by-task build plan.

## Setup

```bash
# 1. Create and activate a virtualenv
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure secrets
cp .env.example .env
# then fill in .env with your real API keys — never commit this file
```

## Project layout

```
adapters/   data source adapters (Reddit, Google Trends, Meta Ads, ...)
scoring/    deterministic composite-score engine
llm/        DeepSeek provider, cost guard, prompt/parsing logic
db/         SQLite schema + migrations
config/     weights.yaml, market_profiles.yaml
app.py      Streamlit UI entry point
```

## Running

Not yet available — the Streamlit app (`app.py`) is a stub until Tasks
14-17 are built. `run.sh` will be added once there's something to run.
