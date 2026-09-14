"""AppTest coverage for app.py's Deep Dive button (docs/FRD.md, Section 10, Task 16).

Runs the real app.py through streamlit.testing.v1.AppTest against a
throwaway DB, so the click -> session_state guard -> deep_score() wiring
is exercised the same way a browser click would trigger it, instead of
only unit-testing deep_score() itself (already covered by
tests/llm/test_deep_pass.py).
"""

import shutil
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from db.migrate import migrate

APP_PATH = Path(__file__).parent.parent / "app.py"

CHEAP_SCORES = {
    "market_demand": 8,
    "competitive_intensity": 4,
    "margin_signal": 6,
    "trend_momentum": 7,
}


def _seed_cheap_pass_idea(conn: sqlite3.Connection) -> tuple[int, int]:
    """Same shape sweep_engine.run_sweep (Task 12) leaves behind: one
    sweep_runs row, one preliminary ideas row, and its 4 cheap-pass scores."""
    now = "2026-09-13T00:00:00Z"
    cursor = conn.execute(
        "INSERT INTO sweep_runs (category, market, triggered_at, status) VALUES (?, ?, ?, 'completed')",
        ("ecommerce", "morocco", now),
    )
    sweep_run_id = cursor.lastrowid
    cursor = conn.execute(
        """
        INSERT INTO ideas (
            sweep_run_id, name, category, market, seed_or_suggested,
            composite_score, composite_confidence, research_priority, has_deep_pass, created_at
        ) VALUES (?, 'Prayer rugs and Islamic home items', 'ecommerce', 'morocco', 'seed',
                  6.31, 0.72, 'revisit_later', 0, ?)
        """,
        (sweep_run_id, now),
    )
    idea_id = cursor.lastrowid
    for dimension, value in CHEAP_SCORES.items():
        conn.execute(
            """
            INSERT INTO scores (idea_id, dimension, value, confidence, evidence_level, rationale, pass_type)
            VALUES (?, ?, ?, 0.7, 'sufficient', 'cheap-pass rationale', 'cheap')
            """,
            (idea_id, dimension, value),
        )
    conn.commit()
    return sweep_run_id, idea_id


@pytest.fixture
def app_dir(tmp_path):
    """A throwaway copy of app.py next to a fresh migrated meezan.db, since
    app.py resolves DB_PATH relative to its own file location — this lets
    the test point it at a disposable DB without touching the real one."""
    shutil.copy(APP_PATH, tmp_path / "app.py")
    db_path = tmp_path / "meezan.db"
    migrate(db_path)
    conn = sqlite3.connect(db_path)
    sweep_run_id, idea_id = _seed_cheap_pass_idea(conn)
    conn.close()
    return tmp_path, sweep_run_id, idea_id


def test_clicking_deep_dive_button_calls_deep_score_exactly_once(app_dir):
    tmp_path, sweep_run_id, idea_id = app_dir

    def fake_deep_score(idea_id_arg, conn, *args, **kwargs):
        conn.execute("UPDATE ideas SET has_deep_pass = 1 WHERE id = ?", (idea_id_arg,))
        conn.commit()
        return {"deep_score": None, "composite": None}

    with patch("llm.deep_pass.deep_score", side_effect=fake_deep_score) as mock_deep_score:
        at = AppTest.from_file(str(tmp_path / "app.py"))
        at.run(timeout=30)
        assert not at.exception

        # "Sweep run" is the second sidebar selectbox (after "Seed list").
        sweep_selectbox = at.sidebar.selectbox[1]
        sweep_selectbox.set_value(sweep_run_id).run(timeout=30)
        assert not at.exception

        deep_dive_button = next(b for b in at.button if b.key == f"deep_dive_{idea_id}")
        deep_dive_button.click().run(timeout=30)
        assert not at.exception

    assert mock_deep_score.call_count == 1
    assert mock_deep_score.call_args.args[0] == idea_id

    conn = sqlite3.connect(tmp_path / "meezan.db")
    has_deep_pass = conn.execute("SELECT has_deep_pass FROM ideas WHERE id = ?", (idea_id,)).fetchone()[0]
    conn.close()
    assert has_deep_pass == 1
