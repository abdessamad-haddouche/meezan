# Project Meezan — Streamlit entry point (docs/FRD.md, Section 10).
#
# Task 14 built the sidebar sweep trigger. Task 15 (this pass) builds the
# main-area results view: a "View sweep" picker, ranked cards (default) with
# a toggle to a sortable table, and filters over both. The per-idea evidence
# expander and Deep Dive button are Task 16 — cards/rows here are read-only.

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from llm.deepseek_provider import DeepSeekProvider
from sweep_engine import SEED_LISTS_DIR, load_seed_list, run_sweep

load_dotenv()

DB_PATH = Path(__file__).parent / "meezan.db"

st.set_page_config(page_title="Project Meezan", layout="wide")

if "sweep_running" not in st.session_state:
    st.session_state.sweep_running = False
if "last_sweep_run_id" not in st.session_state:
    st.session_state.last_sweep_run_id = None
if "last_sweep_tokens" not in st.session_state:
    st.session_state.last_sweep_tokens = None


def _available_seed_lists() -> list[str]:
    return sorted(p.stem for p in SEED_LISTS_DIR.glob("*.yaml"))


@st.cache_data
def load_sweep_runs() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, category, market, triggered_at, status FROM sweep_runs ORDER BY id DESC"
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


@st.cache_data
def load_ideas(sweep_run_id: int) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT name, description, composite_score, research_priority,
                   margin_min_pct, margin_max_pct, cost_min_usd, cost_max_usd,
                   complexity, has_deep_pass
            FROM ideas
            WHERE sweep_run_id = ?
            ORDER BY composite_score DESC
            """,
            (sweep_run_id,),
        ).fetchall()
    finally:
        conn.close()
    return pd.DataFrame([dict(row) for row in rows])


def _display_description(row: pd.Series) -> str:
    description = row["description"]
    if isinstance(description, str) and description.strip():
        return description
    return row["name"]


def _display_margin(row: pd.Series) -> str:
    if row["has_deep_pass"] and pd.notna(row["margin_min_pct"]) and pd.notna(row["margin_max_pct"]):
        return f"{row['margin_min_pct']:.0f}–{row['margin_max_pct']:.0f}%"
    return "pending deep dive"


def _display_cost(row: pd.Series) -> str:
    if row["has_deep_pass"] and pd.notna(row["cost_min_usd"]) and pd.notna(row["cost_max_usd"]):
        return f"${row['cost_min_usd']:,.0f}–${row['cost_max_usd']:,.0f}"
    return "pending deep dive"


def render_idea_card(row: pd.Series) -> None:
    with st.container(border=True):
        st.subheader(row["name"])
        st.write(row["description_display"])
        cols = st.columns(5)
        score = row["composite_score"]
        cols[0].metric("Composite score", f"{score:.2f}" if pd.notna(score) else "—")
        cols[1].metric("Research priority", row["research_priority"] or "—")
        cols[2].metric("Margin", row["margin_display"])
        cols[3].metric("Cost to start", row["cost_display"])
        complexity = row["complexity"]
        cols[4].metric("Complexity", f"{complexity:.0f}" if pd.notna(complexity) else "—")


with st.sidebar:
    st.header("Run a sweep")

    seed_list_options = _available_seed_lists()
    category = st.selectbox("Seed list", seed_list_options)
    market = st.text_input("Market", value="morocco")

    run_clicked = st.button("Run Sweep", disabled=st.session_state.sweep_running)

    if run_clicked:
        st.session_state.sweep_running = True
        st.rerun()

    if st.session_state.sweep_running:
        with st.spinner(f"Running sweep for {category!r} in {market!r} — this can take a while..."):
            seed_niches = load_seed_list(category)
            provider = DeepSeekProvider()
            conn = sqlite3.connect(DB_PATH)
            try:
                sweep_run_id = run_sweep(conn, category, market, seed_niches, provider=provider)
            finally:
                conn.close()

            st.session_state.last_sweep_run_id = sweep_run_id
            st.session_state.last_sweep_tokens = {
                "today": provider.usage.today_tokens,
                "sweep": provider.usage.sweep_tokens,
            }
        st.session_state.sweep_running = False
        load_sweep_runs.clear()
        st.rerun()

    if st.session_state.last_sweep_run_id is not None:
        st.success(f"Last sweep: sweep_run_id={st.session_state.last_sweep_run_id}")
        tokens = st.session_state.last_sweep_tokens
        st.metric("DeepSeek tokens — this sweep", tokens["sweep"])
        st.metric("DeepSeek tokens — today", tokens["today"])

    st.divider()
    st.header("View sweep")

    sweep_runs = load_sweep_runs()
    if not sweep_runs:
        st.caption("No sweeps in the database yet.")
        selected_sweep_id = None
    else:
        sweep_run_ids = [run["id"] for run in sweep_runs]
        sweep_run_labels = {
            run["id"]: f"#{run['id']} — {run['category']} / {run['market']} — {run['triggered_at']}"
            for run in sweep_runs
        }
        default_sweep_id = st.session_state.last_sweep_run_id
        default_index = (
            sweep_run_ids.index(default_sweep_id) if default_sweep_id in sweep_run_ids else 0
        )
        selected_sweep_id = st.selectbox(
            "Sweep run",
            sweep_run_ids,
            index=default_index,
            format_func=lambda run_id: sweep_run_labels[run_id],
        )


st.title("Project Meezan")

if selected_sweep_id is None:
    st.write("Run a sweep to see results.")
else:
    ideas_df = load_ideas(selected_sweep_id)

    if ideas_df.empty:
        st.info("This sweep has no ideas yet.")
    else:
        filter_cols = st.columns([2, 2, 3])
        with filter_cols[0]:
            priority_options = sorted(p for p in ideas_df["research_priority"].dropna().unique())
            selected_priorities = st.multiselect(
                "Research priority", priority_options, default=priority_options
            )
        with filter_cols[1]:
            min_score = st.slider("Min composite score", 0.0, 10.0, 0.0, 0.1)
        with filter_cols[2]:
            search_term = st.text_input("Search by name", "")

        filtered_df = ideas_df
        if selected_priorities:
            filtered_df = filtered_df[filtered_df["research_priority"].isin(selected_priorities)]
        filtered_df = filtered_df[filtered_df["composite_score"].fillna(0) >= min_score]
        if search_term:
            filtered_df = filtered_df[
                filtered_df["name"].str.contains(search_term, case=False, na=False)
            ]
        filtered_df = filtered_df.sort_values("composite_score", ascending=False, na_position="last")

        filtered_df = filtered_df.assign(
            description_display=filtered_df.apply(_display_description, axis=1),
            margin_display=filtered_df.apply(_display_margin, axis=1),
            cost_display=filtered_df.apply(_display_cost, axis=1),
        )

        st.caption(f"{len(filtered_df)} of {len(ideas_df)} ideas shown")
        show_as_table = st.toggle("Show as table", value=False)

        if show_as_table:
            table_df = filtered_df[
                [
                    "name",
                    "description_display",
                    "composite_score",
                    "research_priority",
                    "margin_display",
                    "cost_display",
                    "complexity",
                ]
            ].rename(
                columns={
                    "description_display": "description",
                    "margin_display": "margin",
                    "cost_display": "cost",
                }
            )
            st.dataframe(
                table_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "name": st.column_config.TextColumn("Name"),
                    "description": st.column_config.TextColumn("Description", width="large"),
                    "composite_score": st.column_config.NumberColumn(
                        "Composite score", format="%.2f"
                    ),
                    "research_priority": st.column_config.TextColumn("Research priority"),
                    "margin": st.column_config.TextColumn("Margin"),
                    "cost": st.column_config.TextColumn("Cost to start"),
                    "complexity": st.column_config.NumberColumn("Complexity", format="%d"),
                },
            )
        else:
            for _, idea_row in filtered_df.iterrows():
                render_idea_card(idea_row)
