import sqlite3
from types import SimpleNamespace
from typing import Any
import streamlit as st
from config import JUDGE_DISPLAY
from dashboard_data import validate_experiments_db
from ui.icons import (
    get_database_icon,
    get_file_text_icon,
    get_judge_icon_svg,
    get_scales_icon,
    get_server_icon,
)
from views.eval_tabs.common import (
    EXP_DB_PATH,
    _JUDGE_PEEK_CONTROLLER_HTML,
    get_cached_adversarial_summary,
    get_cached_system_health,
    load_cached_experiments,
)
from views.eval_tabs.tab_adversarial import render_tab_adversarial
from views.eval_tabs.tab_architecture import render_tab_architecture
from views.eval_tabs.tab_case_studies import render_tab_case_studies
from views.eval_tabs.tab_judge_comparison import render_tab_judge_comparison
from views.eval_tabs.tab_leaderboard import render_tab_leaderboard
from views.eval_tabs.tab_reasoning import render_tab_reasoning

__all__ = [
    "render_evaluation_view",
    "get_cached_system_health",
    "load_cached_experiments",
    "get_cached_adversarial_summary",
]


def render_evaluation_view(idx: Any) -> None:
    database_error = validate_experiments_db(EXP_DB_PATH)
    if database_error:
        st.error("Evaluation results could not be loaded. The Chat page remains available.")
        with st.expander("Technical details"):
            st.code(database_error)
        st.stop()

    health = get_cached_system_health()

    st.markdown(_JUDGE_PEEK_CONTROLLER_HTML, unsafe_allow_html=True)

    hero_html = """<div class="eval-hero-banner">
<h1 class="eval-hero-title">Πειραματική Αξιολόγηση Μοντέλων &amp; Αρχιτεκτονικών RAG</h1>
<p class="eval-hero-sub">Συγκριτική εμπειρική αξιολόγηση <strong>τοπικών γλωσσικών μοντέλων (3.2B-9.7B)</strong> και προηγμένων <strong>αρχιτεκτονικών ανάκτησης πληροφορίας</strong> (<em>Dense Semantic Search</em>, <em>BM25 Hybrid</em>, <em>Cross-Encoder Reranker</em>, <em>HyDE</em>) σε βάθος δεκαετίας επίσημων οικονομικών εκθέσεων (<strong>Apple Inc. 10-K FY2016-FY2025</strong>). Ανάλυση της ισορροπίας ποιότητας και ταχύτητας (Pareto Frontier), της πιστότητας των πηγών (Faithfulness) και της ανθεκτικότητας σε παραπλανητικές επιθέσεις (Adversarial Robustness).</p>
</div>"""
    st.markdown(hero_html, unsafe_allow_html=True)

    if "active_judge_provider" not in st.session_state:
        st.session_state["active_judge_provider"] = "azure_gpt5_mini"

    def _sync_judge_from_top():
        val = st.session_state.get("top_active_judge_selector")
        if val:
            st.session_state["active_judge_provider"] = val
            st.session_state["tab7_judge_selector"] = val

    def _sync_judge_from_tab7():
        val = st.session_state.get("tab7_judge_selector")
        if val:
            st.session_state["active_judge_provider"] = val
            st.session_state["top_active_judge_selector"] = val

    def _cycle_judge():
        curr = st.session_state.get("active_judge_provider", "azure_gpt5_mini")
        new_judge = "deepseek_v4" if curr == "azure_gpt5_mini" else "azure_gpt5_mini"
        st.session_state["active_judge_provider"] = new_judge
        st.session_state["top_active_judge_selector"] = new_judge
        st.session_state["tab7_judge_selector"] = new_judge

    if "toggle_judge" in st.query_params:
        target = st.query_params.get("toggle_judge")
        try:
            del st.query_params["toggle_judge"]
        except Exception:
            pass
        if target in ("azure_gpt5_mini", "deepseek_v4"):
            st.session_state["active_judge_provider"] = target
            st.session_state["top_active_judge_selector"] = target
            st.session_state["tab7_judge_selector"] = target
        else:
            _cycle_judge()

    active_judge = st.session_state["active_judge_provider"]
    if st.session_state.get("top_active_judge_selector") != active_judge:
        st.session_state["top_active_judge_selector"] = active_judge
    if st.session_state.get("tab7_judge_selector") != active_judge:
        st.session_state["tab7_judge_selector"] = active_judge

    provider_names = JUDGE_DISPLAY[active_judge]
    active_provider = SimpleNamespace(
        display_name=provider_names["display_name"],
        short_name=provider_names["short_name"],
    )

    with sqlite3.connect(EXP_DB_PATH) as conn:
        try:
            cov_row = conn.execute("""
                SELECT COUNT(*) FROM (
                    SELECT e.id
                    FROM experiments e
                    JOIN judge_scores j ON e.id = j.experiment_id
                    WHERE j.judge_provider = ? AND j.status = 'completed'
                    GROUP BY e.id
                    HAVING COUNT(j.id) >= e.sample_size
                )
            """, (active_judge,)).fetchone()
            tot_row = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()
            scored_runs = cov_row[0] if cov_row else 0
            total_runs = tot_row[0] if tot_row else 0
        except Exception:
            scored_runs, total_runs = 0, 0

    led_ollama = "#22c55e" if health['ollama_running'] else "#ef4444"
    ollama_text = "Ollama" if health['ollama_running'] else "Offline"
    chroma_meta = "Local index available" if health["chromadb_ready"] else "Optional for Chat; not needed for Dashboard"

    active_judge_display = "GPT-5 Mini" if active_judge == "azure_gpt5_mini" else "DeepSeek V4 Flash"
    judge_icon = get_judge_icon_svg(active_judge, size=22, opacity=1.0, grayscale=False)

    col_m1, col_m2, col_m3, col_m4 = st.columns([1.0, 1.0, 1.0, 1.0])
    with col_m1:
        st.markdown(f"""
        <div class="eval-kpi-card">
            <div class="eval-kpi-label" style="display: flex; justify-content: space-between; align-items: center;">
                <span>Local Model Server (optional)</span>
                {get_server_icon()}
            </div>
            <div class="eval-kpi-val" style="color: {led_ollama}; display: flex; align-items: center; gap: 8px;">
                <span class="status-pulse-dot" style="background-color:{led_ollama}; color:{led_ollama};"></span>
                <span>{ollama_text}</span>
            </div>
            <div class="eval-kpi-meta">Only needed for Chat</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown(f"""
        <div class="eval-kpi-card">
            <div class="eval-kpi-label" style="display: flex; justify-content: space-between; align-items: center;">
                <span>Vector Database</span>
                {get_database_icon()}
            </div>
            <div class="eval-kpi-val">
                {health['chromadb_chunks_count']:,} <span class="eval-kpi-denom">Chunks</span>
            </div>
            <div class="eval-kpi-meta">{chroma_meta}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m3:
        st.markdown(f"""
        <div class="eval-kpi-card">
            <div class="eval-kpi-label" style="display: flex; justify-content: space-between; align-items: center;">
                <span>Evaluation Dataset</span>
                {get_file_text_icon()}
            </div>
            <div class="eval-kpi-val">
                {health['benchmark_questions_count']} <span class="eval-kpi-denom">Queries</span>
            </div>
            <div class="eval-kpi-meta">+20 Adversarial Questions</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m4:
        st.markdown(f"""
        <div class="eval-kpi-card eval-kpi-judge-card">
            <div class="eval-kpi-label" style="display: flex; justify-content: space-between; align-items: center; padding-right: 32px;">
                <span>Judge Evaluator</span>
                {get_scales_icon()}
            </div>
            <div class="eval-kpi-val" style="font-size: 19px; display: flex; align-items: center; gap: 8px;">
                {judge_icon}
                <span style="white-space: nowrap; font-size: 19px; letter-spacing: -0.01em;">{active_judge_display}</span>
            </div>
            <div class="eval-kpi-meta">{scored_runs}/{total_runs} Scored Fully</div>
        </div>
        """, unsafe_allow_html=True)
        st.button(":material/swap_horiz:", key="top_cycle_judge_btn", help="Εναλλαγή Judge", on_click=_cycle_judge)

    if health.get("warnings"):
        for w in health["warnings"]:
            st.warning(f"Σημείωση συστήματος: {w}", icon=":material/warning:")

    tab_pareto, tab_rq2, tab_rq3, tab_adversarial, tab_inspect, tab_judge = st.tabs([
        ":material/dashboard: Overview",
        ":material/account_tree: Ablation",
        ":material/psychology: Reasoning",
        ":material/security: Robustness",
        ":material/science: Case Studies",
        ":material/compare_arrows: Judge Comparison",
    ])

    with tab_pareto:
        render_tab_leaderboard(active_judge, active_provider)

    with tab_rq2:
        render_tab_architecture(active_judge, active_provider)

    with tab_rq3:
        render_tab_reasoning(active_judge, active_provider)

    with tab_adversarial:
        render_tab_adversarial()

    with tab_inspect:
        render_tab_case_studies(active_judge, active_provider)

    with tab_judge:
        render_tab_judge_comparison()
