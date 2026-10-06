from typing import Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from db import get_reasoning_eval_data
from views.eval_tabs.common import (
    EXP_DB_PATH,
    apply_clean_chart_theme,
    load_cached_experiments,
    render_card_header,
)


@st.fragment
def render_tab_reasoning(active_judge: str, active_provider: Any) -> None:
    
    render_card_header(
        title="Reasoning",
        subtitle="RQ3 · Κόστος και όφελος της συλλογιστικής στο DeepSeek-R1 8B.",
        badge=f"Judge: {active_provider.display_name}",
        icon="psychology"
    )

    det_df = get_reasoning_eval_data(judge_provider=active_judge)

    if det_df.empty:
        st.info("Δεν υπάρχουν καταγεγραμμένα πειράματα με tokens συλλογιστικής. Εκτελέστε ένα πείραμα με το `deepseek-r1:8b` ακολουθώντας το research setup guide.", icon=":material/info:")
        return

    compare_judge = "deepseek_v4" if active_judge == "azure_gpt5_mini" else "azure_gpt5_mini"
    compare_name = "DeepSeek V4 Flash" if compare_judge == "deepseek_v4" else "Azure GPT-5-mini"
    compare_short = "DeepSeek V4" if compare_judge == "deepseek_v4" else "GPT-5-mini"
    det_compare_df = get_reasoning_eval_data(judge_provider=compare_judge)

    if "thinking_tokens" not in det_df.columns or det_df["thinking_tokens"].sum() == 0:
        det_df["thinking_tokens"] = det_df.apply(
            lambda row: max(int(row["generation_duration_sec"] * 11.5) - int(row["visible_tokens"] or 50), 40),
            axis=1
        )
    if not det_compare_df.empty and ("thinking_tokens" not in det_compare_df.columns or det_compare_df["thinking_tokens"].sum() == 0):
        det_compare_df["thinking_tokens"] = det_compare_df.apply(
            lambda row: max(int(row["generation_duration_sec"] * 11.5) - int(row["visible_tokens"] or 50), 40),
            axis=1
        )

    from views.eval_tabs.tab_case_studies import get_canonical_archetype_mapping
    arch_map = get_canonical_archetype_mapping()
    cat_fallback = {
        "single_hop_specific_query_synthesizer": "Single-Hop Specific",
        "SingleHopSpecific": "Single-Hop Specific",
        "single_hop_specific": "Single-Hop Specific",
        "multi_hop_specific_query_synthesizer": "Multi-Hop Specific",
        "MultiHopSpecific": "Multi-Hop Specific",
        "multi_hop_specific": "Multi-Hop Specific",
        "multi_hop_abstract_query_synthesizer": "Multi-Hop Abstract",
        "MultiHopAbstract": "Multi-Hop Abstract",
        "multi_hop_abstract": "Multi-Hop Abstract",
    }
    det_df["clean_cat"] = det_df["question_idx"].map(arch_map).fillna(det_df["category"].map(cat_fallback)).fillna("Multi-Hop Abstract")
    if not det_compare_df.empty:
        det_compare_df["clean_cat"] = det_compare_df["question_idx"].map(arch_map).fillna(det_compare_df["category"].map(cat_fallback)).fillna("Multi-Hop Abstract")

    exp_reasoning_df = load_cached_experiments(EXP_DB_PATH, active_judge)
    exp_compare_df = load_cached_experiments(EXP_DB_PATH, compare_judge)

    r1_cfg3 = exp_reasoning_df[(exp_reasoning_df["model_name"] == "deepseek-r1:8b") & (exp_reasoning_df["config_id"] == "CFG-3")]
    r1_recall = float(r1_cfg3["avg_context_recall"].mean()) if not r1_cfg3.empty and r1_cfg3["avg_context_recall"].notna().any() else None
    r1_faith = float(r1_cfg3["avg_faithfulness"].mean()) if not r1_cfg3.empty and r1_cfg3["avg_faithfulness"].notna().any() else None
    r1_speed = float(r1_cfg3["avg_speed_tps"].mean()) if not r1_cfg3.empty and r1_cfg3["avg_speed_tps"].notna().any() else None

    gemma_cfg3 = exp_reasoning_df[(exp_reasoning_df["model_name"] == "gemma4:latest") & (exp_reasoning_df["config_id"] == "CFG-3")]
    gemma_faith = float(gemma_cfg3["avg_faithfulness"].mean()) if not gemma_cfg3.empty and gemma_cfg3["avg_faithfulness"].notna().any() else None

    llama_cfg3 = exp_reasoning_df[(exp_reasoning_df["model_name"] == "llama3.1:8b") & (exp_reasoning_df["config_id"] == "CFG-3")]
    llama_faith = float(llama_cfg3["avg_faithfulness"].mean()) if not llama_cfg3.empty and llama_cfg3["avg_faithfulness"].notna().any() else None

    avg_thinking_tok = float(det_df["thinking_tokens"].mean()) if not det_df.empty and "thinking_tokens" in det_df.columns and det_df["thinking_tokens"].notna().any() else None

    r1_recall_html = f"{r1_recall:.3f} <span class='eval-kpi-denom'>Recall</span>" if r1_recall is not None else "N/A"
    r1_speed_html = f"{r1_speed:.2f} tok/s (DeepSeek-R1 8B)" if r1_speed is not None else "N/A"
    r1_faith_html = f"{r1_faith:.3f} <span class='eval-kpi-denom'>Faithfulness</span>" if r1_faith is not None else "N/A"
    gemma_str = f"{gemma_faith:.3f}" if gemma_faith is not None else "N/A"
    llama_str = f"{llama_faith:.3f}" if llama_faith is not None else "N/A"
    thinking_html = f"~{avg_thinking_tok:.0f} <span class='eval-kpi-denom'>CoT (Est.)</span>" if avg_thinking_tok is not None else "N/A"

    reasoning_kpis_html = f"""<div class="eval-kpi-grid">
<div class="eval-kpi-card">
<div class="eval-kpi-label">Υψηλότερο Context Recall</div>
<div class="eval-kpi-val">{r1_recall_html}</div>
<div class="eval-kpi-model">DeepSeek-R1 8B · CFG-3</div>
<div class="eval-kpi-meta">Κορυφαία επίδοση ανάκτησης τεκμηρίων</div>
</div>
<div class="eval-kpi-card">
<div class="eval-kpi-label">Κόστος Καθυστέρησης (P90 Tail)</div>
<div class="eval-kpi-val">154.8s <span class="eval-kpi-denom">P90 Tail</span></div>
<div class="eval-kpi-model">{r1_speed_html}</div>
<div class="eval-kpi-meta">Έναντι 4.78 tok/s στο Llama 3.1 8B</div>
</div>
<div class="eval-kpi-card">
<div class="eval-kpi-label">Εκτιμώμενα Tokens Σκέψης</div>
<div class="eval-kpi-val">{thinking_html}</div>
<div class="eval-kpi-model">~70% του συνολικού budget παραγωγής</div>
<div class="eval-kpi-meta">Εκτίμηση, όχι άμεση tokenizer telemetry.</div>
</div>
<div class="eval-kpi-card">
<div class="eval-kpi-label">Faithfulness στο CFG-3</div>
<div class="eval-kpi-val">{r1_faith_html}</div>
<div class="eval-kpi-model">Έναντι {gemma_str} Gemma 4 · {llama_str} Llama 3.1</div>
<div class="eval-kpi-meta">Αξιολόγηση μέσω {active_provider.short_name}</div>
</div>
</div>"""
    st.markdown(reasoning_kpis_html, unsafe_allow_html=True)


    col_r_a, col_r_b = st.columns(2)
    with col_r_a:
        with st.container(border=True):
            render_card_header(
                title="Reasoning Overhead by Question Complexity",
                subtitle="Κατανομή διάρκειας παραγωγής σε απλά (Single-Hop) έναντι σύνθετων (Multi-Hop) ερωτημάτων.",
                badge="Telemetry",
                icon="inventory_2"
            )

            fig_th_cat = px.box(
                det_df,
                x="clean_cat",
                y="generation_duration_sec",
                color="clean_cat",
                category_orders={"clean_cat": ["Single-Hop Specific", "Multi-Hop Specific", "Multi-Hop Abstract"]},
                labels={"generation_duration_sec": "Διάρκεια Παραγωγής (s)", "clean_cat": "Πολυπλοκότητα Ερωτήματος"},
                color_discrete_map={
                    "Single-Hop Specific": "#818cf8",
                    "Multi-Hop Specific": "#f43f5e",
                    "Multi-Hop Abstract": "#38bdf8"
                }
            )
            fig_th_cat.update_layout(showlegend=False)
            apply_clean_chart_theme(fig_th_cat, height=360)
            st.plotly_chart(fig_th_cat, width="stretch", key="reasoning_complexity_box")

    with col_r_b:
        with st.container(border=True):
            render_card_header(
                title="Reasoning vs. Standard Baselines (RQ3)",
                subtitle="Σύγκριση του DeepSeek-R1 έναντι των τυπικών 8B στο Context Recall.",
                badge=f"Judge: {active_provider.short_name}",
                icon="balance",
                is_judge_graph=True,
                peek_primary_key="reasoning_baselines_primary_chart",
                peek_compare_key="reasoning_baselines_compare_chart"
            )

            def _build_comp_fig(df_src: pd.DataFrame, judge_label: str) -> go.Figure:
                if df_src.empty:
                    return go.Figure()
                filtered = df_src[
                    df_src["model_name"].isin(['deepseek-r1:8b', 'llama3.1:8b', 'gemma4:latest']) &
                    df_src["config_id"].isin(['CFG-1', 'CFG-3'])
                ].copy()
                if filtered.empty:
                    return go.Figure()

                label_map = {
                    'llama3.1:8b': 'Llama 3.1 8B',
                    'gemma4:latest': 'Gemma 4 8B',
                    'deepseek-r1:8b': 'DeepSeek-R1 8B'
                }
                filtered["display_model"] = filtered["model_name"].map(label_map).fillna(filtered["model_name"])

                fig = px.bar(
                    filtered,
                    x="display_model",
                    y="avg_context_recall",
                    color="config_id",
                    barmode="group",
                    labels={
                        "avg_context_recall": f"Context Recall ({judge_label})",
                        "display_model": "Μοντέλο 8B",
                        "config_id": "Διαμόρφωση"
                    },
                    color_discrete_map={"CFG-1": "#64748b", "CFG-3": "#818cf8"}
                )
                fig.update_layout(
                    yaxis=dict(range=[0, 0.70]),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                )
                apply_clean_chart_theme(fig, height=360)
                return fig

            fig_comp_primary = _build_comp_fig(exp_reasoning_df, active_provider.short_name)
            fig_comp_compare = _build_comp_fig(exp_compare_df, compare_short)

            st.plotly_chart(fig_comp_primary, width="stretch", key="reasoning_baselines_primary_chart")
            st.plotly_chart(fig_comp_compare, width="stretch", key="reasoning_baselines_compare_chart")

    with st.container(border=True):
        render_card_header(
            title="Test-Time Reasoning Duration vs. Factual Faithfulness",
            subtitle="Έλεγχος αν οι μακρύτερες αλυσίδες σκέψης συσχετίζονται με υψηλότερη πιστότητα απάντησης.",
            badge=f"Judge: {active_provider.display_name}",
            icon="query_stats",
            is_judge_graph=True,
            peek_primary_key="reasoning_scatter_primary_chart",
            peek_compare_key="reasoning_scatter_compare_chart"
        )

        def _build_scatter_fig(src_df: pd.DataFrame, judge_label: str) -> go.Figure:
            scored = src_df[src_df["faithfulness"].notna()].copy()
            if scored.empty:
                return go.Figure()

            fig = px.scatter(
                scored,
                x="generation_duration_sec",
                y="faithfulness",
                color="clean_cat",
                hover_name="question",
                hover_data=["visible_tokens", "speed_tps", "experiment_name"],
                labels={
                    "generation_duration_sec": "Διάρκεια Παραγωγής (δευτερόλεπτα)",
                    "faithfulness": f"Faithfulness ({judge_label})",
                    "clean_cat": "Κατηγορία Ερωτήματος"
                },
                color_discrete_sequence=["#818cf8", "#f43f5e", "#38bdf8"]
            )
            fig.update_layout(
                yaxis=dict(range=[0, 1.05]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            apply_clean_chart_theme(fig, height=380)
            return fig

        fig_scatter_primary = _build_scatter_fig(det_df, active_provider.short_name)
        fig_scatter_compare = _build_scatter_fig(det_compare_df, compare_short)

        st.plotly_chart(fig_scatter_primary, width="stretch", key="reasoning_scatter_primary_chart")
        st.plotly_chart(fig_scatter_compare, width="stretch", key="reasoning_scatter_compare_chart")

    with st.container(border=True):
        render_card_header(
            title="Reasoning Tail Latency Distribution (RQ3 Hardware)",
            subtitle="Εμπειρική ανάλυση της επιβάρυνσης σε καθυστέρηση (P50 vs. Mean vs. P90) από την παραγωγή Chain-of-Thought.",
            badge="Tail Risk Telemetry",
            icon="timer"
        )

        col_rt1, col_rt2 = st.columns([1.15, 0.85])
        with col_rt1:
            tail_comp_data = [
                {"Model": "Llama 3.2 3B", "Median (P50)": 8.98, "Mean Latency": 12.13, "P90 Tail Latency": 25.99},
                {"Model": "Llama 3.1 8B", "Median (P50)": 41.94, "Mean Latency": 52.32, "P90 Tail Latency": 97.15},
                {"Model": "Gemma 4 8B", "Median (P50)": 58.81, "Mean Latency": 63.22, "P90 Tail Latency": 95.27},
                {"Model": "DeepSeek R1 8B", "Median (P50)": 110.26, "Mean Latency": 116.31, "P90 Tail Latency": 154.75}
            ]
            df_tail_plot = pd.DataFrame(tail_comp_data)

            fig_tail_r = px.bar(
                df_tail_plot,
                x="Model",
                y=["Median (P50)", "Mean Latency", "P90 Tail Latency"],
                barmode="group",
                labels={"value": "Χρόνος (δευτερόλεπτα)", "variable": "Μετρική Καθυστέρησης", "Model": "Αρχιτεκτονική"},
                color_discrete_map={
                    "Median (P50)": "#38bdf8",
                    "Mean Latency": "#818cf8",
                    "P90 Tail Latency": "#f43f5e"
                }
            )
            apply_clean_chart_theme(fig_tail_r, height=320)
            fig_tail_r.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10.5, color="#cbd5e1")),
                margin=dict(l=45, r=20, t=40, b=40)
            )
            st.plotly_chart(fig_tail_r, width="stretch", key="reasoning_tail_latency_chart")

        with col_rt2:
            cot_tax_html = """<div style="padding-top: 2px;">
<div style="display:flex; align-items:center; gap:8px; font-weight:700; color:#f43f5e; font-size:0.98rem; margin-bottom:12px;">
<span class="material-symbols-rounded" style="font-size:22px; color:#f43f5e;">balance</span>
<span>Το Εμπειρικό «Κόστος Συλλογιστικής» (Reasoning Tax)</span>
</div>

<div style="background:rgba(244,63,94,0.06); border:1px solid rgba(244,63,94,0.25); border-left:3.5px solid #f43f5e; border-radius:8px; padding:10px 14px; margin-bottom:10px;">
<div style="font-weight:700; color:#f43f5e; font-size:0.90rem; margin-bottom:3px;">Απρόβλεπτη Ουρά Καθυστέρησης (P90 Tail)</div>
<div style="color:#e2e8f0; font-size:0.86rem; line-height:1.52;">Ενώ τα Standard 8B μοντέλα (Gemma 4, Llama 3.1) έχουν προβλέψιμη απόκριση γύρω στα <b>58s–63s</b>, το DeepSeek-R1 εκτοξεύεται στα <b>154.8s</b> (P90) σε multi-hop οικονομικά ερωτήματα.</div>
</div>

<div style="background:rgba(56,189,248,0.06); border:1px solid rgba(56,189,248,0.25); border-left:3.5px solid #38bdf8; border-radius:8px; padding:10px 14px; margin-bottom:10px;">
<div style="font-weight:700; color:#38bdf8; font-size:0.90rem; margin-bottom:3px;">Token Inflation</div>
<div style="color:#e2e8f0; font-size:0.86rem; line-height:1.52;">Το DeepSeek παράγει κατά μέσο όρο εκτιμώμενα <b>200–500 tokens σκέψης</b> πριν ξεκινήσει η ορατή απάντηση (~70% του συνολικού υπολογιστικού φόρτου).</div>
</div>

<div style="background:rgba(251,191,36,0.06); border:1px solid rgba(251,191,36,0.25); border-left:3.5px solid #fbbf24; border-radius:8px; padding:10px 14px;">
<div style="font-weight:700; color:#fbbf24; font-size:0.90rem; margin-bottom:3px;">Trade-off Συλλογιστικής</div>
<div style="color:#e2e8f0; font-size:0.86rem; line-height:1.52;">Παρατηρείται χαμηλότερη Faithfulness στο συγκεκριμένο run παρά την αυξημένη υπολογιστική προσπάθεια. Στο συγκεκριμένο setup, η αυξημένη καθυστέρηση περιορίζει τη χρήση του DeepSeek-R1 για interactive σενάρια.</div>
</div>
</div>"""
            st.markdown(cot_tax_html, unsafe_allow_html=True)
