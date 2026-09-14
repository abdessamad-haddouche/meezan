# Project Meezan — Streamlit entry point (docs/FRD.md, Section 10, Task 14).
#
# This task builds only the sidebar: seed-list/market inputs and the
# "Run Sweep" button wired to the real `run_sweep()` from sweep_engine.py,
# guarded against Streamlit's rerun-on-interaction re-triggering a sweep
# (Section 10's session_state pattern). The ranked list, cards, table,
# filters, and deep-dive button are Task 15/16 — the main area here is a
# placeholder.

import sqlite3
from pathlib import Path

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
        st.rerun()

    if st.session_state.last_sweep_run_id is not None:
        st.success(f"Last sweep: sweep_run_id={st.session_state.last_sweep_run_id}")
        tokens = st.session_state.last_sweep_tokens
        st.metric("DeepSeek tokens — this sweep", tokens["sweep"])
        st.metric("DeepSeek tokens — today", tokens["today"])


st.title("Project Meezan")
st.write("Run a sweep to see results.")
