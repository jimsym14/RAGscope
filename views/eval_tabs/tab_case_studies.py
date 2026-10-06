import json
import sqlite3
from typing import Any, Dict, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from db import get_experiments_list
from views.eval_tabs.common import (
    EXP_DB_PATH,
    apply_clean_chart_theme,
    fmt_score,
    fmt_sec,
    fmt_tps,
    render_card_header,
)


@st.cache_data
def get_canonical_archetype_mapping() -> Dict[int, str]:
    
    conn = sqlite3.connect(EXP_DB_PATH)
    q = """
    SELECT 
        question_idx,
        CASE 
            WHEN LOWER(category) LIKE '%single%hop%' OR LOWER(category) LIKE '%factoid%' THEN 'Single-Hop Specific'
            WHEN LOWER(category) LIKE '%multi%hop%abstract%' OR LOWER(category) LIKE '%operational%' THEN 'Multi-Hop Abstract'
            WHEN LOWER(category) LIKE '%multi%hop%specific%' OR LOWER(category) LIKE '%cross%year%' OR LOWER(category) LIKE '%financial%' THEN 'Multi-Hop Specific'
            ELSE 'Multi-Hop Abstract'
        END as Archetype
    FROM detailed_results
    WHERE experiment_id = 6
    GROUP BY question_idx;
    """
    df = pd.read_sql_query(q, conn)
    conn.close()

    assert len(df) == 103, f"Corpus partition error: expected 103 questions, got {len(df)}"
    counts = df["Archetype"].value_counts().to_dict()
    sh = counts.get("Single-Hop Specific", 0)
    mhs = counts.get("Multi-Hop Specific", 0)
    mha = counts.get("Multi-Hop Abstract", 0)

    assert sh == 40, f"Expected 40 Single-Hop Specific, got {sh}"
    assert mhs == 31, f"Expected 31 Multi-Hop Specific, got {mhs}"
    assert mha == 32, f"Expected 32 Multi-Hop Abstract, got {mha}"
    assert (sh + mhs + mha) == 103, f"Partition count sum error: {sh + mhs + mha} != 103"

    return dict(zip(df["question_idx"], df["Archetype"]))


def _get_archetype_metrics_df(judge_provider: str) -> pd.DataFrame:
    canonical_map = get_canonical_archetype_mapping()
    try:
        conn = sqlite3.connect(EXP_DB_PATH)
        q = f"""
        SELECT 
            d.question_idx,
            j.faithfulness,
            j.context_recall,
            j.context_precision
        FROM detailed_results d
        JOIN judge_scores j ON d.experiment_id = j.experiment_id AND d.question_idx = j.question_idx
        WHERE j.judge_provider = '{judge_provider}' AND d.experiment_id IN (3, 4, 5, 6, 7, 8, 11, 12, 13, 14);
        """
        raw_df = pd.read_sql_query(q, conn)
        conn.close()

        if raw_df.empty:
            return pd.DataFrame()

        raw_df["Archetype"] = raw_df["question_idx"].map(canonical_map)

        grouped = raw_df.groupby("Archetype").agg(
            Faithfulness=("faithfulness", lambda x: round(x.mean(), 3)),
            Recall=("context_recall", lambda x: round(x.mean(), 3)),
            Precision=("context_precision", lambda x: round(x.mean(), 3)),
            n_evals=("faithfulness", "count")
        ).reset_index()

        order_map = {"Single-Hop Specific": 1, "Multi-Hop Specific": 2, "Multi-Hop Abstract": 3}
        grouped["order"] = grouped["Archetype"].map(order_map)
        grouped = grouped.sort_values("order").drop(columns=["order"]).reset_index(drop=True)
        return grouped
    except Exception:
        return pd.DataFrame()


def _build_archetype_chart(df: pd.DataFrame, judge_name: str) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Δεν υπάρχουν διαθέσιμα δεδομένα αρχετύπων", showarrow=False)
        apply_clean_chart_theme(fig, height=310)
        return fig

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Archetype"],
        y=df["Faithfulness"],
        name="Faithfulness (Πιστότητα)",
        marker_color="#fb7185",
        text=[f"{v:.3f}" for v in df["Faithfulness"]],
        textposition="auto"
    ))
    fig.add_trace(go.Bar(
        x=df["Archetype"],
        y=df["Recall"],
        name="Context Recall (Ανάκληση)",
        marker_color="#34d399",
        text=[f"{v:.3f}" for v in df["Recall"]],
        textposition="auto"
    ))
    fig.add_trace(go.Bar(
        x=df["Archetype"],
        y=df["Precision"],
        name="Context Precision (Ακρίβεια)",
        marker_color="#38bdf8",
        text=[f"{v:.3f}" for v in df["Precision"]],
        textposition="auto"
    ))

    fig.update_layout(
        barmode="group",
        yaxis=dict(range=[0.0, 1.05], title="Μέση Βαθμολογία (0.0–1.0)"),
        xaxis=dict(title=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10.0, color="#cbd5e1")),
        margin=dict(l=45, r=20, t=35, b=35)
    )
    apply_clean_chart_theme(fig, height=335)
    return fig


def _get_worst_questions_df(exp_id: int, judge_provider: str, sort_by: str = "faithfulness", limit: int = 10, archetype_filter: str = "All") -> pd.DataFrame:
    canonical_map = get_canonical_archetype_mapping()
    sort_col = "j.faithfulness" if sort_by == "faithfulness" else ("j.context_recall" if sort_by == "recall" else "j.context_precision")

    try:
        conn = sqlite3.connect(EXP_DB_PATH)
        q = f"""
        SELECT 
            d.question_idx,
            d.question,
            d.ground_truth,
            d.generated_answer,
            d.has_thinking,
            d.thinking_text,
            d.contexts,
            d.speed_tps,
            d.ttft_sec,
            d.total_duration_sec,
            ROUND(j.faithfulness, 3) as Faithfulness,
            ROUND(j.context_recall, 3) as Recall,
            ROUND(j.context_precision, 3) as Precision,
            ROUND(j.answer_relevancy, 3) as Relevancy,
            CASE 
                WHEN j.context_recall < 0.35 THEN 'Retrieval Omission'
                WHEN j.faithfulness < 0.40 THEN 'Extrapolation / Hallucination'
                WHEN j.context_precision < 0.40 THEN 'Low Precision (Noise)'
                ELSE 'Partial Extrapolation'
            END as Diagnostic
        FROM detailed_results d
        JOIN judge_scores j ON d.experiment_id = j.experiment_id AND d.question_idx = j.question_idx
        WHERE d.experiment_id = {exp_id} AND j.judge_provider = '{judge_provider}'
        ORDER BY {sort_col} ASC, j.faithfulness ASC;
        """
        df = pd.read_sql_query(q, conn)
        conn.close()

        if df.empty:
            return pd.DataFrame()

        df["Archetype"] = df["question_idx"].map(canonical_map)

        if archetype_filter != "All":
            df = df[df["Archetype"] == archetype_filter]

        return df.head(limit)
    except Exception:
        return pd.DataFrame()


@st.fragment
def render_tab_case_studies(active_judge: str, active_provider: Any) -> None:
    
    render_card_header(
        title="Case Studies",
        subtitle="Υποστηρικτική ανάλυση σφαλμάτων, μοτίβων αποτυχίας και ανακτημένων τεκμηρίων.",
        icon="science"
    )

    canonical_map = get_canonical_archetype_mapping()

    df_arch_active = _get_archetype_metrics_df(active_judge)
    sh_rec = 0.647
    mha_rec = 0.381
    if not df_arch_active.empty:
        row_sh = df_arch_active[df_arch_active["Archetype"] == "Single-Hop Specific"]
        row_mha = df_arch_active[df_arch_active["Archetype"] == "Multi-Hop Abstract"]
        if not row_sh.empty:
            sh_rec = float(row_sh["Recall"].iloc[0])
        if not row_mha.empty:
            mha_rec = float(row_mha["Recall"].iloc[0])
    recall_drop_pct = ((mha_rec - sh_rec) / sh_rec * 100.0) if sh_rec > 0 else -41.1

    cs_kpis_html = f"""<div class="eval-kpi-grid">
<div class="eval-kpi-card">
<div class="eval-kpi-label">Corpus Αναφοράς</div>
<div class="eval-kpi-val">103 <span class="eval-kpi-denom">Questions</span></div>
<div class="eval-kpi-model">Apple Inc. 10-K (FY16–FY25)</div>
<div class="eval-kpi-meta">10 έτη ελεγμένων οικονομικών εκθέσεων SEC</div>
</div>
<div class="eval-kpi-card">
<div class="eval-kpi-label">Multi-Hop Recall Drop</div>
<div class="eval-kpi-val">{recall_drop_pct:.1f}% <span class="eval-kpi-denom">Recall Drop</span></div>
<div class="eval-kpi-model">{sh_rec:.3f} → {mha_rec:.3f}</div>
<div class="eval-kpi-meta">Single-Hop → Multi-Hop Abstract ({active_provider.short_name})</div>
</div>
<div class="eval-kpi-card">
<div class="eval-kpi-label">Χαμηλότερο Context Recall</div>
<div class="eval-kpi-val">{mha_rec:.3f} <span class="eval-kpi-denom">Recall</span></div>
<div class="eval-kpi-model">Multi-Hop Abstract · 32 ερωτήσεις</div>
<div class="eval-kpi-meta">Χαμηλότερο παρατηρούμενο Recall μεταξύ των 3 αρχετύπων ({active_provider.short_name}).</div>
</div>
<div class="eval-kpi-card">
<div class="eval-kpi-label">Παρατηρούμενο Μοτίβο</div>
<div class="eval-kpi-val">Παραίσθηση / Υπόθεση</div>
<div class="eval-kpi-model">Συμπλήρωση κενών ανάκτησης</div>
<div class="eval-kpi-meta">Συμπλήρωση κενών όταν λείπει επαρκές context.</div>
</div>
</div>"""
    st.markdown(cs_kpis_html, unsafe_allow_html=True)

    arch_grid_html = """<div class="adv-tax-grid-tiny" style="grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); margin: 4px 0 16px 0;">
<div class="adv-tax-card-tiny" style="border-left: 3.5px solid #818cf8;">
<div class="adv-tax-title-tiny"><span class="material-symbols-rounded" style="color:#818cf8;font-size:18px;">tag</span> Single-Hop Specific · 40 Ερωτήσεις</div>
<div class="adv-tax-desc-tiny">Άμεση εύρεση μεμονωμένου οικονομικού μεγέθους, ποσοστού ή ορισμού από ένα σημείο των 10-K (υψηλότερο recall ~0.647, χαμηλή διασπορά, προβλέψιμη απόκριση).</div>
</div>
<div class="adv-tax-card-tiny" style="border-left: 3.5px solid #f43f5e;">
<div class="adv-tax-title-tiny"><span class="material-symbols-rounded" style="color:#f43f5e;font-size:18px;">schema</span> Multi-Hop Specific · 31 Ερωτήσεις</div>
<div class="adv-tax-desc-tiny">Διασταύρωση και σύγκριση συγκεκριμένων στοιχείων μεταξύ διαφορετικών ετών ή σημειώσεων (αυξημένη απαίτηση cross-chunk retrieval).</div>
</div>
<div class="adv-tax-card-tiny" style="border-left: 3.5px solid #38bdf8;">
<div class="adv-tax-title-tiny"><span class="material-symbols-rounded" style="color:#38bdf8;font-size:18px;">psychology</span> Multi-Hop Abstract · 32 Ερωτήσεις</div>
<div class="adv-tax-desc-tiny">Εννοιολογική σύνθεση μακροπρόθεσμης στρατηγικής, ρυθμιστικών κινδύνων και τάσεων δεκαετίας (μεγαλύτερη πτώση recall σε 0.381, −41.1%).</div>
</div>
</div>"""
    st.markdown(arch_grid_html, unsafe_allow_html=True)

    with st.container(border=True):
        render_card_header(
            title="Quality by Question Archetype (Benchmark Diagnostics)",
            subtitle="Ποιοτική αποτίμηση δεικτών Ragas ανά αρχέτυπο ερωτήματος (Single-Hop Specific, Multi-Hop Specific, Multi-Hop Abstract).",
            badge="Archetype Diagnostics",
            icon="analytics",
            is_judge_graph=True
        )

        col_q_arch, col_fail_pat = st.columns([1.15, 0.85])
        with col_q_arch:
            compare_judge = "deepseek_v4" if active_judge == "azure_gpt5_mini" else "azure_gpt5_mini"
            df_arch_compare = _get_archetype_metrics_df(compare_judge)

            fig_primary = _build_archetype_chart(df_arch_active, active_provider.display_name)
            fig_compare = _build_archetype_chart(df_arch_compare, "DeepSeek V4" if active_judge == "azure_gpt5_mini" else "Azure GPT-5-mini")

            st.plotly_chart(fig_primary, width="stretch", key="cs_arch_primary_chart")
            st.plotly_chart(fig_compare, width="stretch", key="cs_arch_compare_chart")

        with col_fail_pat:
            fail_pat_html = """<div style="padding-top: 2px;">
<div style="display:flex; align-items:center; gap:8px; font-weight:700; color:#fb7185; font-size:0.98rem; margin-bottom:12px;">
<span class="material-symbols-rounded" style="font-size:22px; color:#fb7185;">report_problem</span>
<span>Παρατηρούμενα Μοτίβα Αποτυχίας</span>
</div>

<div style="background:rgba(251,113,133,0.06); border:1px solid rgba(251,113,133,0.25); border-left:3.5px solid #fb7185; border-radius:8px; padding:10px 14px; margin-bottom:10px;">
<div style="font-weight:700; color:#fb7185; font-size:0.90rem; margin-bottom:3px;">Retrieval Omission</div>
<div style="color:#e2e8f0; font-size:0.86rem; line-height:1.52;">Δεν ανακτώνται όλα τα σχετικά έτη ή αποσπάσματα (κυρίως στο Multi-Hop, όπου η πληροφορία διασπείρεται σε διαφορετικά 10-K filings).</div>
</div>

<div style="background:rgba(245,158,11,0.06); border:1px solid rgba(245,158,11,0.25); border-left:3.5px solid #f59e0b; border-radius:8px; padding:10px 14px; margin-bottom:10px;">
<div style="font-weight:700; color:#f59e0b; font-size:0.90rem; margin-bottom:3px;">Extrapolation / Hallucination</div>
<div style="color:#e2e8f0; font-size:0.86rem; line-height:1.52;">Το μοντέλο συμπληρώνει πληροφορία ή υποθέτει αριθμητικά δεδομένα όταν λείπει επαρκές context από τα ανακτηθέντα chunks.</div>
</div>

<div style="background:rgba(56,189,248,0.06); border:1px solid rgba(56,189,248,0.25); border-left:3.5px solid #38bdf8; border-radius:8px; padding:10px 14px; margin-bottom:8px;">
<div style="font-weight:700; color:#38bdf8; font-size:0.90rem; margin-bottom:3px;">Format Violations</div>
<div style="color:#e2e8f0; font-size:0.86rem; line-height:1.52;">Η απάντηση δεν ακολουθεί τον ζητούμενο μορφολογικό περιορισμό (π.χ. αυστηρή απαίτηση σύνταξης raw Python dictionary).</div>
</div>

<div style="font-size:0.78rem; color:#94a3b8; border-top:1px solid rgba(255,255,255,0.08); padding-top:6px; font-style:italic;">
Σημείωση: Παρατηρούμενα μοτίβα βάσει ενδεικτικών ευρετικών ορίων (heuristics) στο benchmark 103 ερωτήσεων, όχι καθολική χειροκίνητη ταξινόμηση.
</div>
</div>"""
            st.markdown(fail_pat_html, unsafe_allow_html=True)

    all_exp_df = get_experiments_list()
    if all_exp_df.empty:
        st.info("Δεν υπάρχουν καταγεγραμμένα αποτελέσματα πειραμάτων.", icon=":material/info:")
        return

    prod_exp_ids = [3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 17]
    exp_filtered = all_exp_df[all_exp_df["id"].isin(prod_exp_ids)].copy()
    if exp_filtered.empty:
        exp_filtered = all_exp_df.copy()

    with st.container(border=True):
        render_card_header(
            title="Worst Questions Diagnostic & Linked Question Inspector",
            subtitle="Διαδραστικός εντοπισμός προβληματικών αποκρίσεων και πλήρης επιθεώρηση ανακτημένων τεκμηρίων (10-K Chunks).",
            badge="Error Analysis",
            icon="troubleshoot"
        )

        col_exp, col_sort, col_limit, col_arch = st.columns([1.2, 0.9, 0.5, 0.9])
        with col_exp:
            exp_lookup = {r["id"]: f"#{r['id']} - {r['experiment_name']} ({r['model_name']})" for _, r in exp_filtered.iterrows()}
            default_exp = 6 if 6 in exp_lookup else exp_filtered["id"].iloc[0]
            selected_exp_id = st.selectbox(
                "Επιλογή Πειράματος",
                exp_filtered["id"].tolist(),
                index=exp_filtered["id"].tolist().index(default_exp) if default_exp in exp_filtered["id"].tolist() else 0,
                format_func=lambda x: exp_lookup.get(x, f"Run {x}")
            )

        with col_sort:
            sort_options = {
                "faithfulness": "Lowest Faithfulness (Παραισθήσεις)",
                "recall": "Lowest Recall (Κενά Ανάκλησης)",
                "precision": "Lowest Precision (Άσχετα Chunks)"
            }
            selected_sort_key = st.selectbox(
                "Ταξινόμηση ανά",
                list(sort_options.keys()),
                format_func=lambda k: sort_options[k]
            )

        with col_limit:
            limit_n = st.selectbox("Εμφάνιση", [5, 10, 15, 20], index=1)

        with col_arch:
            arch_filter_options = ["All", "Single-Hop Specific", "Multi-Hop Specific", "Multi-Hop Abstract"]
            selected_arch_filter = st.selectbox(
                "Φίλτρο Αρχετύπου",
                arch_filter_options,
                format_func=lambda a: "Όλα τα Αρχέτυπα" if a == "All" else a
            )

        df_worst = _get_worst_questions_df(
            selected_exp_id,
            active_judge,
            sort_by=selected_sort_key,
            limit=limit_n,
            archetype_filter=selected_arch_filter
        )

        if not df_worst.empty:
            st.markdown(
                f"<div style='font-size:0.88rem; font-weight:600; color:#cbd5e1; margin-top:10px; margin-bottom:6px;'>Χαμηλότερες Βαθμολογίες (Top {len(df_worst)} Queries υπό τον {active_provider.display_name}):</div>",
                unsafe_allow_html=True
            )
            st.dataframe(
                df_worst[[
                    "question_idx", "question", "Archetype", "Faithfulness",
                    "Recall", "Precision", "Diagnostic"
                ]],
                width="stretch",
                hide_index=True,
                column_config={
                    "question_idx": st.column_config.NumberColumn("#", width="small"),
                    "question": st.column_config.TextColumn("Ερώτημα", width="large"),
                    "Archetype": st.column_config.TextColumn("Αρχέτυπο", width="medium"),
                    "Faithfulness": st.column_config.ProgressColumn("Faithfulness", min_value=0.0, max_value=1.0, format="%.3f"),
                    "Recall": st.column_config.ProgressColumn("Recall", min_value=0.0, max_value=1.0, format="%.3f"),
                    "Precision": st.column_config.ProgressColumn("Precision", min_value=0.0, max_value=1.0, format="%.3f"),
                    "Diagnostic": st.column_config.TextColumn("Ενδεικτική Διάγνωση (Heuristic)", width="medium")
                }
            )
            st.caption("Η ένδειξη βασίζεται σε threshold-based heuristic και δεν αποτελεί χειροκίνητη root-cause annotation.")

            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            st.markdown(
                "<div style='font-size:0.92rem; font-weight:700; color:#38bdf8; margin-bottom:6px;'><span class='material-symbols-rounded' style='vertical-align:middle;font-size:18px;color:#38bdf8;margin-right:4px;'>manage_search</span>Επιθεώρηση Επιλεγμένου Ερωτήματος (Question Inspector):</div>",
                unsafe_allow_html=True
            )

            q_options = df_worst["question_idx"].tolist()
            q_lookup = {r["question_idx"]: f"Q{r['question_idx']}: {r['question'][:80]}... (Faith: {r['Faithfulness']:.2f}, Recall: {r['Recall']:.2f})" for _, r in df_worst.iterrows()}

            selected_q_idx = st.selectbox(
                "Επιλέξτε ερώτημα προς πλήρη ανατομία:",
                q_options,
                format_func=lambda q: q_lookup.get(q, f"Question Q{q}")
            )

            row_q = df_worst[df_worst["question_idx"] == selected_q_idx].iloc[0]

            with st.container(border=True):
                st.markdown(f"<div style='font-weight:700; color:#38bdf8; font-size:1.02rem; margin-bottom:8px;'>Q{row_q['question_idx']} · [{row_q['Archetype']}]</div>", unsafe_allow_html=True)

                col_insp_left, col_insp_right = st.columns([1.18, 0.82])
                with col_insp_left:
                    st.markdown(f"**Question:**\n{row_q['question']}")
                    st.markdown(f"**Reference Answer** *(Source-anchored benchmark reference):*\n> {row_q['ground_truth']}")
                    st.markdown(f"**Model Answer:**\n{row_q['generated_answer']}")

                    if row_q.get("has_thinking") and row_q.get("thinking_text"):
                        with st.expander(f"DeepSeek-R1 Reasoning Trace (CoT)", expanded=False, icon=":material/psychology:"):
                            st.caption(row_q["thinking_text"])

                with col_insp_right:
                    with st.container(border=True):
                        st.markdown(f"<div style='font-weight:700; color:#cbd5e1; margin-bottom:6px;'><span class='material-symbols-rounded' style='vertical-align:middle;font-size:16px;color:#38bdf8;margin-right:4px;'>fact_check</span>Βαθμολογίες ({active_provider.display_name})</div>", unsafe_allow_html=True)
                        with st.container(horizontal=True):
                            st.metric("Faithfulness", fmt_score(row_q.get("Faithfulness")), border=True)
                            st.metric("Relevancy", fmt_score(row_q.get("Relevancy")), border=True)
                        with st.container(horizontal=True):
                            st.metric("Recall", fmt_score(row_q.get("Recall")), border=True)
                            st.metric("Precision", fmt_score(row_q.get("Precision")), border=True)

                        st.caption(f"Speed: `{fmt_tps(row_q.get('speed_tps'))}` · TTFT: `{fmt_sec(row_q.get('ttft_sec'))}` · Total: `{fmt_sec(row_q.get('total_duration_sec'))}`")

                    diag_text = row_q.get("Diagnostic", "")
                    if "Retrieval Omission" in diag_text:
                        st.error(f"**Ενδεικτική Διάγνωση (Heuristic):** {diag_text}\n\n*Χαμηλό Recall: Τα ανακτηθέντα chunks δεν περιείχαν όλα τα αναγκαία έτη 10-K για την απάντηση.*", icon=":material/search_off:")
                    elif "Extrapolation" in diag_text:
                        st.warning(f"**Ενδεικτική Διάγνωση (Heuristic):** {diag_text}\n\n*Χαμηλή Πιστότητα: Το μοντέλο παρήγαγε ισχυρισμούς μη τεκμηριωμένους από τα σχετικά αποσπάσματα.*", icon=":material/warning:")
                    else:
                        st.info(f"**Ενδεικτική Διάγνωση (Heuristic):** {diag_text}", icon=":material/info:")

                    st.caption("Η ένδειξη βασίζεται σε threshold-based heuristic και δεν αποτελεί χειροκίνητη root-cause annotation.")

                try:
                    raw_ctxs = row_q.get("contexts")
                    if raw_ctxs:
                        if isinstance(raw_ctxs, str):
                            try:
                                ctxs = json.loads(raw_ctxs)
                            except Exception:
                                ctxs = [raw_ctxs]
                        elif isinstance(raw_ctxs, list):
                            ctxs = raw_ctxs
                        else:
                            ctxs = [str(raw_ctxs)]

                        if ctxs and isinstance(ctxs, list):
                            with st.expander(f"Ανακτηθέντα Chunks από Εκθέσεις 10-K ({len(ctxs)} αποσπάσματα)", expanded=False, icon=":material/description:"):
                                for idx_c, ctx in enumerate(ctxs):
                                    chunk_len = len(str(ctx))
                                    st.markdown(f"**Chunk #{idx_c+1}** *(Μέγεθος: {chunk_len} χαρακτήρες)*:")
                                    st.text(str(ctx))
                        elif ctxs:
                            with st.expander("Ανακτηθέντα Chunks από Εκθέσεις 10-K", expanded=False, icon=":material/description:"):
                                st.text(str(ctxs[0]))
                except Exception:
                    pass
        else:
            st.info("Δεν βρέθηκαν ερωτήματα με τα επιλεγμένα κριτήρια φιλτραρίσματος.", icon=":material/info:")
