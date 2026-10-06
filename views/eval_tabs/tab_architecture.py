from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from views.eval_tabs.common import (
    EXP_DB_PATH,
    MODEL_COLORS,
    apply_clean_chart_theme,
    load_cached_experiments,
    render_card_header,
)
from views.eval_tabs.sensitivity_section import render_sensitivity_section


def _safe_float(val: Any) -> Optional[float]:
    
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


@st.fragment
def render_tab_architecture(active_judge: str, active_provider: Any) -> None:
    
    render_card_header(
        title="Ablation",
        subtitle="RQ2 · Ποσοτική αποτίμηση της συνεισφοράς των επιμέρους επιπέδων RAG (ανάκτηση, BM25, reranker, HyDE) στην ποιότητα και την ταχύτητα.",
        badge=f"Judge: {active_provider.display_name}",
        icon="account_tree"
    )

    exp_df = load_cached_experiments(EXP_DB_PATH, active_judge)
    if exp_df.empty:
        st.info("Δεν υπάρχουν διαθέσιμα πειράματα για τη σύγκριση αρχιτεκτονικών RAG.", icon=":material/info:")
        return

    scored_df = exp_df[exp_df["avg_faithfulness"].notna() & (exp_df["judge_status"].isin(["completed", "partial"]))].copy()
    if scored_df.empty:
        st.info(f"ℹ️ Δεν υπάρχουν ακόμη βαθμολογημένα πειράματα από το {active_provider.display_name}. Μεταβείτε στην καρτέλα 'Judge Scoring' για βαθμολόγηση.", icon=":material/gavel:")

    CFG_ORDER = ["CFG-1", "CFG-2", "CFG-3", "CFG-4", "CFG-4-LITE"]
    clean_cfg_names = {
        "CFG-1": "Naive Dense",
        "CFG-2": "Hybrid BM25",
        "CFG-3": "Advanced Rerank",
        "CFG-4": "Full HyDE",
        "CFG-4-LITE": "HyDE-Lite",
    }

    compare_judge = "deepseek_v4" if active_judge == "azure_gpt5_mini" else "azure_gpt5_mini"
    exp_compare_df = load_cached_experiments(EXP_DB_PATH, compare_judge)

    if not scored_df.empty:
        scope_df = scored_df[scored_df["model_name"].astype(str).str.contains("llama3.1:8b", case=False, na=False)].copy()

        cfg_agg = scope_df.groupby("config_id").agg({
            "avg_faithfulness": "mean",
            "avg_context_precision": "mean",
            "avg_context_recall": "mean",
            "avg_answer_relevancy": "mean",
            "avg_overall_score": "mean",
        }).to_dict(orient="index")

        if "CFG-2" not in cfg_agg:
            l_df = scored_df[scored_df["model_name"].astype(str).str.contains("llama3.1:8b", case=False, na=False)]
            c2_rows = l_df[l_df["config_id"] == "CFG-2"]
            if not c2_rows.empty:
                cfg_agg["CFG-2"] = c2_rows.iloc[0].to_dict()

        cfg1_r = _safe_float(cfg_agg.get("CFG-1", {}).get("avg_context_recall"))
        cfg2_r = _safe_float(cfg_agg.get("CFG-2", {}).get("avg_context_recall"))

        if cfg1_r is not None and cfg2_r is not None and cfg1_r > 0:
            recall_lift = ((cfg2_r - cfg1_r) / cfg1_r) * 100
            val_1_html = f"+{recall_lift:.1f}% <span class=\"eval-kpi-denom\">Recall</span>"
            sub_1_html = f"CFG-2 έναντι Dense ({cfg2_r:.3f} vs {cfg1_r:.3f})"
        else:
            val_1_html = "N/A"
            sub_1_html = "Μη διαθέσιμη μέτρηση"

        cfg1_f = _safe_float(cfg_agg.get("CFG-1", {}).get("avg_faithfulness"))
        cfg3_f = _safe_float(cfg_agg.get("CFG-3", {}).get("avg_faithfulness"))

        if cfg1_f is not None and cfg3_f is not None and cfg1_f > 0:
            rerank_f_lift = ((cfg3_f - cfg1_f) / cfg1_f) * 100
            val_2_html = f"+{rerank_f_lift:.1f}% <span class=\"eval-kpi-denom\">Faithfulness</span>"
            sub_2_html = f"CFG-3 έναντι Dense ({cfg3_f:.3f} vs {cfg1_f:.3f})"
        else:
            val_2_html = "N/A"
            sub_2_html = "Μη διαθέσιμη μέτρηση"

        ov_candidates = {
            k: _safe_float(v.get("avg_overall_score"))
            for k, v in cfg_agg.items()
            if _safe_float(v.get("avg_overall_score")) is not None
        }
        if ov_candidates:
            best_ov_cfg = max(ov_candidates.keys(), key=lambda k: ov_candidates[k])
            best_ov_score = ov_candidates[best_ov_cfg]
            val_3_html = f"{best_ov_cfg} · {best_ov_score:.3f}"
            sub_3_html = f"Κορυφαία μέση ποιότητα Ragas"
            meta_3_html = f"Αξιολόγηση μέσω {active_provider.short_name} · Κορυφαία επίδοση ποιότητας"
        else:
            val_3_html = "N/A"
            sub_3_html = "N/A"
            meta_3_html = "Μη διαθέσιμη μέτρηση"

        val_4_html = "−41.5% <span class=\"eval-kpi-denom\">Latency</span>"
        sub_4_html = "59.9s έναντι 102.4s (CFG-4-LITE)"
        meta_4_html = "99.7% διατήρηση συνολικής ποιότητας"

        arch_kpis_html = f"""<div class="eval-kpi-grid">
<div class="eval-kpi-card">
<div class="eval-kpi-label">BM25 Sparse Recall</div>
<div class="eval-kpi-val">{val_1_html}</div>
<div class="eval-kpi-model">{sub_1_html}</div>
<div class="eval-kpi-meta">Ανάκτηση αριθμών &amp; πινάκων 10-K</div>
</div>
<div class="eval-kpi-card">
<div class="eval-kpi-label">Cross-Encoder Reranker</div>
<div class="eval-kpi-val">{val_2_html}</div>
<div class="eval-kpi-model">{sub_2_html}</div>
<div class="eval-kpi-meta">Κορυφαία πιστότητα απάντησης</div>
</div>
<div class="eval-kpi-card">
<div class="eval-kpi-label">Best Quality Architecture</div>
<div class="eval-kpi-val">{val_3_html}</div>
<div class="eval-kpi-model">{sub_3_html}</div>
<div class="eval-kpi-meta">{meta_3_html}</div>
</div>
<div class="eval-kpi-card">
<div class="eval-kpi-label">Lite vs Full HyDE</div>
<div class="eval-kpi-val">{val_4_html}</div>
<div class="eval-kpi-model">{sub_4_html}</div>
<div class="eval-kpi-meta">{meta_4_html}</div>
</div>
</div>"""
        st.markdown(arch_kpis_html, unsafe_allow_html=True)

    col_cfg_a, col_cfg_b = st.columns(2)

    with col_cfg_a:
        with st.container(border=True):
            render_card_header(
                title="Pipeline",
                subtitle="Σταδιακή εξέλιξη των μετρικών ποιότητας (Faithfulness, Precision, Recall) κατά μήκος των διαμορφώσεων RAG στο μοντέλο αναφοράς Llama 3.1 8B.",
                badge="Llama 3.1 8B",
                icon="trending_up",
                is_judge_graph=True,
                peek_primary_key="rq2_pipeline_primary_chart",
                peek_compare_key="rq2_pipeline_compare_chart"
            )

            def _build_prog_fig(source_df: pd.DataFrame) -> go.Figure:
                l_df = source_df[
                    source_df["model_name"].astype(str).str.contains("llama3.1:8b", case=False, na=False)
                    & source_df["avg_faithfulness"].notna()
                    & (source_df["judge_status"].isin(["completed", "partial"]))
                ].copy() if not source_df.empty else pd.DataFrame()

                if l_df.empty:
                    return go.Figure()

                cfg_agg_df = l_df.groupby(["config_id"]).agg({
                    "avg_faithfulness": "mean",
                    "avg_context_precision": "mean",
                    "avg_context_recall": "mean",
                }).reset_index()

                cfg_agg_df["order"] = cfg_agg_df["config_id"].map(lambda x: CFG_ORDER.index(x) if x in CFG_ORDER else 99)
                cfg_agg_df = cfg_agg_df.sort_values("order").drop(columns=["order"])

                f_prog = px.bar(
                    cfg_agg_df,
                    x="config_id",
                    y=["avg_faithfulness", "avg_context_precision", "avg_context_recall"],
                    barmode="group",
                    labels={"value": "Βαθμολογία (0.0–1.0)", "variable": "Μετρική", "config_id": "Στάδιο RAG"},
                    color_discrete_map={
                        "avg_faithfulness": "#fb7185",
                        "avg_context_precision": "#38bdf8",
                        "avg_context_recall": "#34d399"
                    }
                )
                f_prog.for_each_trace(lambda t: t.update(name=t.name.replace("avg_", "").replace("_", " ").capitalize()))
                apply_clean_chart_theme(f_prog, height=380)
                f_prog.update_layout(
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5,
                        font=dict(size=10.5, color="#cbd5e1"),
                        bgcolor="rgba(20,20,19,0.75)",
                        bordercolor="rgba(255,255,255,0.1)"
                    ),
                    yaxis=dict(range=[0.0, 1.05], title="Βαθμολογία (0.0–1.0)"),
                    margin=dict(l=45, r=20, t=35, b=40)
                )
                return f_prog

            fig_prog_primary = _build_prog_fig(scored_df)
            fig_prog_compare = _build_prog_fig(exp_compare_df)

            st.plotly_chart(fig_prog_primary, width="stretch", key="rq2_pipeline_primary_chart")
            st.plotly_chart(fig_prog_compare, width="stretch", key="rq2_pipeline_compare_chart")

    with col_cfg_b:
        with st.container(border=True):
            render_card_header(
                title="Quality Gain",
                subtitle="Καθαρή μεταβολή (Δ) κάθε μετρικής σε σύγκριση με το βασικό επίπεδο Naive Dense (CFG-1) στο μοντέλο αναφοράς Llama 3.1 8B, αναδεικνύοντας τα οφέλη και τις υποχωρήσεις κάθε σταδίου.",
                badge=f"Judge: {active_provider.short_name}",
                icon="bar_chart",
                is_judge_graph=True,
                peek_primary_key="rq2_gain_primary_chart",
                peek_compare_key="rq2_gain_compare_chart"
            )

            def _build_gain_fig(source_df: pd.DataFrame) -> go.Figure:
                l_df = pd.DataFrame(source_df[source_df["model_name"] == "llama3.1:8b"]).copy()
                if l_df.empty:
                    return go.Figure()

                cfg1_rows = l_df[l_df["config_id"] == "CFG-1"]
                if cfg1_rows.empty:
                    return go.Figure()

                base_f = _safe_float(cfg1_rows.iloc[0].get("avg_faithfulness"))
                base_p = _safe_float(cfg1_rows.iloc[0].get("avg_context_precision"))
                base_r = _safe_float(cfg1_rows.iloc[0].get("avg_context_recall"))

                delta_rows = []
                for stage in ["CFG-2", "CFG-3", "CFG-4", "CFG-4-LITE"]:
                    s_rows = l_df[l_df["config_id"] == stage]
                    if not s_rows.empty:
                        row = s_rows.iloc[0]
                        f_val = _safe_float(row.get("avg_faithfulness"))
                        p_val = _safe_float(row.get("avg_context_precision"))
                        r_val = _safe_float(row.get("avg_context_recall"))
                        delta_rows.append({
                            "Stage": stage,
                            "Δ Faithfulness": round(f_val - base_f, 4) if (f_val is not None and base_f is not None) else None,
                            "Δ Precision": round(p_val - base_p, 4) if (p_val is not None and base_p is not None) else None,
                            "Δ Recall": round(r_val - base_r, 4) if (r_val is not None and base_r is not None) else None,
                        })


                if not delta_rows:
                    return go.Figure()

                df_deltas = pd.DataFrame(delta_rows)
                f_gain = px.bar(
                    df_deltas,
                    x="Stage",
                    y=["Δ Faithfulness", "Δ Precision", "Δ Recall"],
                    barmode="group",
                    labels={"value": "Δ μεταβολή (έναντι CFG-1)", "variable": "Μετρική", "Stage": "Στάδιο Αρχιτεκτονικής"},
                    color_discrete_map={
                        "Δ Faithfulness": "#fb7185",
                        "Δ Precision": "#38bdf8",
                        "Δ Recall": "#34d399",
                    }
                )
                apply_clean_chart_theme(f_gain, height=380)
                f_gain.add_hline(y=0.0, line_dash="dash", line_color="rgba(255,255,255,0.25)", line_width=1)
                f_gain.update_layout(
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5,
                        font=dict(size=10.5, color="#cbd5e1"),
                        bgcolor="rgba(20,20,19,0.75)",
                        bordercolor="rgba(255,255,255,0.1)"
                    ),
                    yaxis=dict(title="Δ μεταβολή (έναντι CFG-1)", zeroline=True, zerolinecolor="rgba(255,255,255,0.3)"),
                    margin=dict(l=45, r=20, t=35, b=40)
                )
                return f_gain

            fig_gain_primary = _build_gain_fig(scored_df)
            fig_gain_compare = _build_gain_fig(exp_compare_df)

            st.plotly_chart(fig_gain_primary, width="stretch", key="rq2_gain_primary_chart")
            st.plotly_chart(fig_gain_compare, width="stretch", key="rq2_gain_compare_chart")

    llama_ablation_df = exp_df[exp_df["model_name"].astype(str).str.contains("llama3.1:8b", case=False, na=False)].copy()
    if not llama_ablation_df.empty:
        with st.container(border=True):
            has_lite_in_tbl = "CFG-4-LITE" in llama_ablation_df["config_id"].values and bool(llama_ablation_df[llama_ablation_df["config_id"] == "CFG-4-LITE"]["avg_faithfulness"].notna().any())
            render_card_header(
                title="Ablation Results",
                subtitle=f"Εμπειρική αποτίμηση και αναλυτικός πίνακας των σταδίων αρχιτεκτονικής ({'CFG-1 έως CFG-4-LITE' if has_lite_in_tbl else 'CFG-1 έως CFG-4'}) στο μοντέλο αναφοράς Llama 3.1 8B με πλήρεις μετρικές Ragas και χρόνους εκτέλεσης.",
                badge=f"Judge: {active_provider.display_name}",
                icon="table_chart",
                is_judge_graph=False
            )
            abl_cols = [
                "config_id", "config_name", "avg_faithfulness", "avg_answer_relevancy",
                "avg_context_precision", "avg_context_recall", "avg_retrieval_duration_sec",
                "avg_rerank_duration_sec", "avg_speed_tps", "avg_total_duration_sec"
            ]
            avail_abl_cols = [c for c in abl_cols if c in llama_ablation_df.columns]
            llama_ablation_df["order"] = llama_ablation_df["config_id"].map(lambda x: CFG_ORDER.index(x) if x in CFG_ORDER else 99)
            sorted_abl_df = pd.DataFrame(llama_ablation_df.sort_values("order")[avail_abl_cols])
            st.dataframe(
                sorted_abl_df,
                width="stretch",
                column_config={
                    "config_id": st.column_config.TextColumn("Config"),
                    "config_name": st.column_config.TextColumn("Architecture Description"),
                    "avg_faithfulness": st.column_config.ProgressColumn("Faithfulness", min_value=0.0, max_value=1.0, format="%.3f"),
                    "avg_answer_relevancy": st.column_config.ProgressColumn("Relevancy", min_value=0.0, max_value=1.0, format="%.3f"),
                    "avg_context_precision": st.column_config.ProgressColumn("Precision", min_value=0.0, max_value=1.0, format="%.3f"),
                    "avg_context_recall": st.column_config.ProgressColumn("Recall", min_value=0.0, max_value=1.0, format="%.3f"),
                    "avg_speed_tps": st.column_config.NumberColumn("Speed (t/s)", format="%.1f"),
                    "avg_total_duration_sec": st.column_config.NumberColumn("Total (s)", format="%.2f"),
                }
            )

    col_lat, col_hyde = st.columns([1.1, 0.9])

    with col_lat:
        with st.container(border=True):
            render_card_header(
                title="Latency",
                subtitle="Ανάλυση χρόνου ανά στάδιο εκτέλεσης (Αναζήτηση διανυσμάτων/BM25 → Reranking Cross-Encoder → Προ-παραγωγή HyDE → Τελική Απάντηση LLM) με απομονωμένη μέτρηση στο Llama 3.1 8B.",
                badge="Ανάλυση Σταδίων",
                icon="timer"
            )

            lat_decomp_data = [
                {"Architecture": "CFG-1", "Vector/BM25 Search": 0.13, "Cross-Encoder Rerank": 0.00, "HyDE Pre-generation": 0.00, "LLM Generation": 26.47},
                {"Architecture": "CFG-2", "Vector/BM25 Search": 0.13, "Cross-Encoder Rerank": 0.00, "HyDE Pre-generation": 0.00, "LLM Generation": 27.38},
                {"Architecture": "CFG-3", "Vector/BM25 Search": 0.29, "Cross-Encoder Rerank": 1.05, "HyDE Pre-generation": 0.00, "LLM Generation": 35.57},
                {"Architecture": "CFG-4", "Vector/BM25 Search": 0.13, "Cross-Encoder Rerank": 2.12, "HyDE Pre-generation": 47.89, "LLM Generation": 36.37},
                {"Architecture": "CFG-4-LITE", "Vector/BM25 Search": 0.29, "Cross-Encoder Rerank": 2.10, "HyDE Pre-generation": 9.91, "LLM Generation": 34.63},
            ]
            df_lat_decomp = pd.DataFrame(lat_decomp_data)

            fig_time_ovh = px.bar(
                df_lat_decomp,
                x="Architecture",
                y=["Vector/BM25 Search", "Cross-Encoder Rerank", "HyDE Pre-generation", "LLM Generation"],
                barmode="stack",
                labels={"value": "Χρόνος (δευτερόλεπτα)", "variable": "Στάδιο", "Architecture": "Διαμόρφωση RAG"},
                color_discrete_map={
                    "Vector/BM25 Search": "#38bdf8",
                    "Cross-Encoder Rerank": "#e11d48",
                    "HyDE Pre-generation": "#a855f7",
                    "LLM Generation": "#818cf8"
                }
            )
            apply_clean_chart_theme(fig_time_ovh, height=310)
            fig_time_ovh.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10.0, color="#cbd5e1"),
                    bgcolor="rgba(20,20,19,0.75)",
                    bordercolor="rgba(255,255,255,0.1)"
                ),
                margin=dict(l=45, r=20, t=35, b=40)
            )
            st.plotly_chart(fig_time_ovh, width="stretch")

    with col_hyde:
        with st.container(border=True):
            render_card_header(
                title="HyDE Cost vs Quality",
                subtitle="Συσχέτιση υπολογιστικού χρόνου (s) και συνολικής ποιότητας Ragas: Full HyDE (8B) έναντι HyDE-Lite (3B).",
                badge="Trade-off",
                icon="scatter_plot"
            )

            fig_hyde_scatter = go.Figure()

            fig_hyde_scatter.add_trace(go.Scatter(
                x=[102.44],
                y=[0.665],
                mode="markers+text",
                name="Full HyDE (8B, 512 tok)",
                text=["Full HyDE<br><b>102.4s · 0.665</b>"],
                textposition="bottom right",
                textfont=dict(size=11, color="#fb7185"),
                marker=dict(size=18, color="#fb7185", symbol="circle", line=dict(width=2, color="#ffffff"))
            ))

            fig_hyde_scatter.add_trace(go.Scatter(
                x=[59.88],
                y=[0.663],
                mode="markers+text",
                name="HyDE-Lite (3B, 128 tok)",
                text=["HyDE-Lite<br><b>59.9s · 0.663</b>"],
                textposition="top right",
                textfont=dict(size=11, color="#38bdf8"),
                marker=dict(size=18, color="#38bdf8", symbol="diamond", line=dict(width=2, color="#ffffff"))
            ))

            fig_hyde_scatter.add_annotation(
                x=59.88,
                y=0.663,
                ax=102.44,
                ay=0.665,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.4,
                arrowwidth=2,
                arrowcolor="rgba(56, 189, 248, 0.75)",
                text="<b>−41.5% χρόνος</b><br>(99.7% διατήρηση ποιότητας)",
                font=dict(size=10.5, color="#38bdf8"),
                bgcolor="rgba(15, 23, 42, 0.85)",
                bordercolor="rgba(56, 189, 248, 0.35)",
                borderwidth=1,
                borderpad=4
            )

            apply_clean_chart_theme(fig_hyde_scatter, height=310)
            fig_hyde_scatter.update_layout(
                xaxis=dict(title="Χρόνος Απόκρισης / Latency (δευτερόλεπτα)", range=[45, 125]),
                yaxis=dict(title="Συνολική Ποιότητα (Overall Score)", range=[0.655, 0.675]),
                showlegend=False,
                margin=dict(l=45, r=25, t=35, b=40)
            )
            st.plotly_chart(fig_hyde_scatter, width="stretch")

    with st.container(border=True):
        render_card_header(
            title="Resource Cost",
            subtitle="Άμεση σύγκριση χρόνου απόκρισης και δέσμευσης μνήμης SSD swap μεταξύ Full HyDE και HyDE-Lite.",
            badge="Hardware Impact",
            icon="memory"
        )

        col_rc1, col_rc2 = st.columns(2)

        with col_rc1:
            df_rc_lat = pd.DataFrame([
                {"Διαμόρφωση": "Full HyDE (8B)", "Χρόνος (s)": 102.44, "Ετικέτα": "102.4 s"},
                {"Διαμόρφωση": "HyDE-Lite (3B)", "Χρόνος (s)": 59.88, "Ετικέτα": "59.9 s (−41.5%)"}
            ])
            fig_rc_lat = px.bar(
                df_rc_lat,
                x="Χρόνος (s)",
                y="Διαμόρφωση",
                orientation="h",
                text="Ετικέτα",
                color="Διαμόρφωση",
                color_discrete_map={"Full HyDE (8B)": "#fb7185", "HyDE-Lite (3B)": "#38bdf8"}
            )
            apply_clean_chart_theme(fig_rc_lat, height=150)
            fig_rc_lat.update_layout(
                showlegend=False,
                xaxis=dict(title="Latency (δευτερόλεπτα)", range=[0, 125]),
                yaxis=dict(title=""),
                margin=dict(l=10, r=20, t=15, b=25)
            )
            fig_rc_lat.update_traces(textposition="inside", textfont=dict(size=11, color="#ffffff"))
            st.plotly_chart(fig_rc_lat, width="stretch")

        with col_rc2:
            df_rc_swap = pd.DataFrame([
                {"Διαμόρφωση": "Full HyDE (8B)", "Peak Swap (GB)": 11.80, "Ετικέτα": "11.80 GB"},
                {"Διαμόρφωση": "HyDE-Lite (3B)", "Peak Swap (GB)": 3.58, "Ετικέτα": "3.58 GB (−69.7%)"}
            ])
            fig_rc_swap = px.bar(
                df_rc_swap,
                x="Peak Swap (GB)",
                y="Διαμόρφωση",
                orientation="h",
                text="Ετικέτα",
                color="Διαμόρφωση",
                color_discrete_map={"Full HyDE (8B)": "#fb7185", "HyDE-Lite (3B)": "#34d399"}
            )
            apply_clean_chart_theme(fig_rc_swap, height=150)
            fig_rc_swap.update_layout(
                showlegend=False,
                xaxis=dict(title="Peak SSD Swap (GB)", range=[0, 14]),
                yaxis=dict(title=""),
                margin=dict(l=10, r=20, t=15, b=25)
            )
            fig_rc_swap.update_traces(textposition="inside", textfont=dict(size=11, color="#ffffff"))
            st.plotly_chart(fig_rc_swap, width="stretch")

        st.markdown(
            """<div style="font-size: 0.84rem; color: #94a3b8; line-height: 1.5; margin-top: 4px; padding-top: 6px; border-top: 1px solid rgba(255,255,255,0.06);">
<span class="material-symbols-rounded" style="color:#38bdf8; font-size:1rem; vertical-align:middle; margin-right:4px;">info</span>
Το HyDE-Lite επιτυγχάνει εξοικονόμηση <b>42.6 δευτερολέπτων (−41.5%)</b> και <b>8.22 GB μνήμης swap (−69.7%)</b>, διατηρώντας το <b>99.7% της συνολικής ποιότητας Ragas</b> (0.663 έναντι 0.665 στον ελεγχόμενο μικρο-έλεγχο).
</div>""",
            unsafe_allow_html=True
        )

    adv_grid_html = """<div class="adv-tax-grid-tiny">
<div class="adv-tax-card-tiny">
<div class="adv-tax-title-tiny"><span class="material-symbols-rounded" style="color:#38bdf8;font-size:14px;">filter_1</span> CFG-1 · Naive Dense</div>
<div class="adv-tax-desc-tiny">Vector search για εύρεση σχετικών αποσπασμάτων από τα 10-K.</div>
</div>
<div class="adv-tax-card-tiny">
<div class="adv-tax-title-tiny"><span class="material-symbols-rounded" style="color:#818cf8;font-size:14px;">filter_2</span> CFG-2 · Hybrid BM25</div>
<div class="adv-tax-desc-tiny">Προσθέτει BM25 για να βρίσκει καλύτερα ακριβείς αριθμούς, ποσοστά και όρους.</div>
</div>
<div class="adv-tax-card-tiny">
<div class="adv-tax-title-tiny"><span class="material-symbols-rounded" style="color:#fb7185;font-size:14px;">filter_3</span> CFG-3 · Advanced Rerank</div>
<div class="adv-tax-desc-tiny">Ξαναταξινομεί τα αποτελέσματα με Cross-Encoder και κρατά τα πιο χρήσιμα.</div>
</div>
<div class="adv-tax-card-tiny">
<div class="adv-tax-title-tiny"><span class="material-symbols-rounded" style="color:#34d399;font-size:14px;">filter_4</span> CFG-4 · Full HyDE</div>
<div class="adv-tax-desc-tiny">Το 8B μοντέλο δημιουργεί πρώτα ένα υποθετικό κείμενο 512 tokens για την αναζήτηση.</div>
</div>
<div class="adv-tax-card-tiny">
<div class="adv-tax-title-tiny"><span class="material-symbols-rounded" style="color:#a7f3d0;font-size:14px;">bolt</span> CFG-4-LITE · HyDE Lite</div>
<div class="adv-tax-desc-tiny">Το 3B μοντέλο δημιουργεί ένα σύντομο κείμενο 128 tokens και το 8B δίνει την τελική απάντηση.</div>
</div>
</div>"""
    st.markdown(adv_grid_html, unsafe_allow_html=True)

    render_sensitivity_section()
