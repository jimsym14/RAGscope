from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from db import delete_experiment_from_db
from views.eval_tabs.common import (
    EXP_DB_PATH,
    MODEL_COLORS,
    apply_clean_chart_theme,
    get_cached_adversarial_summary,
    load_cached_experiments,
    render_card_header,
)


def _clean_model_display(name: str) -> str:
    s = str(name or "")
    s = s.replace(" (via llama3.2:3b HyDE)", " + HyDE")
    s = s.replace(" (via llama3.2:3b)", " + HyDE")
    s = s.replace(" (HyDE)", " + HyDE")
    return s


def _fmt_cfg(cid: str) -> str:
    c = (cid or "").upper().strip()
    mapping = {
        "CFG-1": "CFG-1 (Naive)",
        "CFG-2": "CFG-2 (Hybrid)",
        "CFG-3": "CFG-3 (Reranker)",
        "CFG-4": "CFG-4 (HyDE)",
        "CFG-4-LITE": "CFG-4-Lite",
    }
    if c in mapping:
        return mapping[c]
    raw = {
        "CFG-1": "CFG-1 - Naive / Dense RAG",
        "CFG-2": "CFG-2 - Hybrid RAG (Ablation)",
        "CFG-3": "CFG-3 - Advanced RAG / Reranking (Core)",
        "CFG-4": "CFG-4 - Advanced RAG + HyDE (Ablation)",
    }.get(c, c)
    raw = raw.split(" - ", 1)[-1].strip() if " - " in raw else raw
    return f"{c} ({raw})" if c not in raw else raw


@st.fragment
def render_tab_leaderboard(active_judge: str, active_provider: Any) -> None:
    
    render_card_header(
        title="Overview",
        subtitle="RQ1 · Συνολική αποτίμηση τοπικών μοντέλων (3.2B–8B): ισορροπία πιστότητας απάντησης (Faithfulness), ταχύτητας παραγωγής (tok/s) και ανθεκτικότητας (Refusal Rate).",
        badge=f"Judge: {active_provider.display_name}",
        icon="dashboard"
    )

    exp_df = load_cached_experiments(EXP_DB_PATH, active_judge)
    if exp_df.empty:
        st.info("Δεν βρέθηκαν πειράματα στη βάση δεδομένων. Ακολουθήστε το research setup guide για να εκτελέσετε νέα πειράματα.", icon=":material/info:")
        return

    scored_df = pd.DataFrame(exp_df[exp_df["avg_faithfulness"].notna() & (exp_df["judge_status"].isin(["completed", "partial"]))].copy())

    if not scored_df.empty:
        scored_df["overall_score"] = scored_df["avg_overall_score"].fillna(scored_df[["avg_faithfulness", "avg_answer_relevancy", "avg_context_precision", "avg_context_recall"]].mean(axis=1))
        best_overall = scored_df.iloc[int(np.argmax(np.asarray(scored_df["overall_score"])))]
        best_faith = scored_df.iloc[int(np.argmax(np.asarray(scored_df["avg_faithfulness"])))]

        fastest_exp = exp_df.iloc[int(np.argmax(np.asarray(exp_df["avg_speed_tps"])))] if ("avg_speed_tps" in exp_df.columns and not exp_df.empty and bool(exp_df["avg_speed_tps"].notna().any())) else {"avg_speed_tps": 0.0, "model_name": "N/A", "config_id": "N/A"}

        adv_summary = get_cached_adversarial_summary(EXP_DB_PATH)
        best_adv = adv_summary.iloc[int(np.argmax(np.asarray(adv_summary["refusal_rate_pct"])))] if ("refusal_rate_pct" in adv_summary.columns and not adv_summary.empty and bool(adv_summary["refusal_rate_pct"].notna().any())) else {"refusal_rate_pct": 0.0, "model_name": "N/A", "config_id": "N/A"}

        kpi_cards_html = f"""<div class="eval-kpi-grid">
<div class="eval-kpi-card">
<div class="eval-kpi-label">Top Overall Quality</div>
<div class="eval-kpi-val">{best_overall['overall_score']:.3f} <span class="eval-kpi-denom">/ 1.0</span></div>
<div class="eval-kpi-subline"><span class="eval-kpi-model">{_clean_model_display(best_overall['model_name'])}</span><span class="eval-kpi-meta">{_fmt_cfg(best_overall.get('config_id', 'CFG-3'))}</span></div>
</div>
<div class="eval-kpi-card">
<div class="eval-kpi-label">Highest Faithfulness</div>
<div class="eval-kpi-val">{best_faith['avg_faithfulness']:.3f} <span class="eval-kpi-denom">/ 1.0</span></div>
<div class="eval-kpi-subline"><span class="eval-kpi-model">{_clean_model_display(best_faith['model_name'])}</span><span class="eval-kpi-meta">{_fmt_cfg(best_faith.get('config_id', 'CFG-3'))}</span></div>
</div>
<div class="eval-kpi-card">
<div class="eval-kpi-label">Fastest Inference Speed</div>
<div class="eval-kpi-val">{fastest_exp['avg_speed_tps']:.1f} <span class="eval-kpi-denom">tok/s</span></div>
<div class="eval-kpi-subline"><span class="eval-kpi-model">{_clean_model_display(fastest_exp['model_name'])}</span><span class="eval-kpi-meta">{_fmt_cfg(fastest_exp.get('config_id', 'CFG-1'))}</span></div>
</div>
<div class="eval-kpi-card">
<div class="eval-kpi-label">Highest Refusal Rate</div>
<div class="eval-kpi-val" style="font-size: 32px !important; font-weight: 700 !important;">{best_adv['refusal_rate_pct']:.1f}% <span class="eval-kpi-denom">Refusal</span></div>
<div class="eval-kpi-subline"><span class="eval-kpi-model">{_clean_model_display(best_adv['model_name'])}</span><span class="eval-kpi-meta">{_fmt_cfg(best_adv.get('config_id', 'CFG-3'))}</span></div>
</div>
</div>"""
        st.markdown(kpi_cards_html, unsafe_allow_html=True)
    else:
        st.info("Experiments generated with local inference. Score them in the Judge scoring tab to populate quality metrics.", icon=":material/gavel:")

    col_p1, col_p2 = st.columns(2)

    compare_judge = "deepseek_v4" if active_judge == "azure_gpt5_mini" else "azure_gpt5_mini"
    exp_compare_df = load_cached_experiments(EXP_DB_PATH, compare_judge)

    plot_df = exp_df[exp_df["avg_faithfulness"].notna() & exp_df["avg_speed_tps"].notna()].copy()
    plot_df["param_size_num"] = plot_df["model_params"].astype(str).str.replace("B", "", case=False).apply(pd.to_numeric, errors="coerce").fillna(8.0)
    plot_df["plot_faithfulness"] = plot_df["avg_faithfulness"]
    plot_df["short_config"] = plot_df["config_id"].fillna("CFG")

    comb_speeds = pd.concat([plot_df["avg_speed_tps"], exp_compare_df["avg_speed_tps"]]).dropna()
    pareto_x_min = max(0.0, float(comb_speeds.min()) * 0.85) if not comb_speeds.empty else 0.0
    pareto_x_max = float(comb_speeds.max()) * 1.08 if not comb_speeds.empty else 60.0
    pareto_y_min = 0.35
    pareto_y_max = 0.95

    def _build_pareto_fig(source_df: pd.DataFrame) -> go.Figure:
        p_df = source_df[source_df["avg_faithfulness"].notna() & source_df["avg_speed_tps"].notna()].copy()
        p_df["plot_faithfulness"] = p_df["avg_faithfulness"]
        p_df["short_config"] = p_df["config_id"].apply(_fmt_cfg)

        valid_pts = p_df.sort_values("avg_speed_tps", ascending=False)
        pareto_x, pareto_y = [], []
        if len(valid_pts) >= 2:
            max_y = -1
            for _, row in valid_pts.iterrows():
                if row["plot_faithfulness"] > max_y:
                    pareto_x.append(row["avg_speed_tps"])
                    pareto_y.append(row["plot_faithfulness"])
                    max_y = row["plot_faithfulness"]

        pareto_trace = go.Scatter(
            x=pareto_x,
            y=pareto_y,
            mode="lines",
            name="Pareto Frontier",
            line=dict(color="#fb7185", width=2.5, dash="dash"),
            hoverinfo="skip"
        ) if len(pareto_x) >= 2 else None

        CONFIG_SYMBOLS = {
            "CFG-1": "x",
            "CFG-2": "diamond",
            "CFG-3": "square",
            "CFG-4": "circle",
            "CFG-4-LITE": "triangle-up"
        }
        fig_scatter = px.scatter(
            p_df,
            x="avg_speed_tps",
            y="plot_faithfulness",
            color="model_name",
            symbol="short_config",
            symbol_map=CONFIG_SYMBOLS,
            color_discrete_map=MODEL_COLORS,
            hover_name="experiment_name",
            hover_data=["config_name", "avg_ttft_sec", "avg_total_duration_sec", "sample_size"],
            labels={"avg_speed_tps": "Generation Speed (tok/s)", "plot_faithfulness": "Ragas Faithfulness", "model_name": "Model", "short_config": "Config"},
        )
        traces = ([pareto_trace] if pareto_trace else []) + list(fig_scatter.data)
        fig_res = go.Figure(data=traces, layout=fig_scatter.layout)
        start_idx = 1 if pareto_trace else 0
        for trace in fig_res.data[start_idx:]:
            if hasattr(trace, "marker") and getattr(trace, "marker", None) is not None:
                getattr(trace, "marker").size = 13
                getattr(trace, "marker").line = dict(width=1.5, color="rgba(255, 255, 255, 0.9)")

        apply_clean_chart_theme(fig_res, height=420)
        fig_res.update_layout(
            xaxis=dict(title_text="Generation Speed (tok/s)", range=[pareto_x_min, pareto_x_max]),
            yaxis=dict(title_text="Ragas Faithfulness", range=[pareto_y_min, pareto_y_max]),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.03,
                xanchor="center",
                x=0.5,
                font=dict(size=10.5, color="#cbd5e1"),
                bgcolor="rgba(20,20,19,0.75)",
                bordercolor="rgba(255,255,255,0.1)"
            ),
            margin=dict(l=50, r=25, t=55, b=45)
        )
        return fig_res

    with col_p1:
        with st.container(border=True):
            render_card_header(
                title="Pareto Frontier",
                subtitle="Συσχέτιση ταχύτητας παραγωγής (tok/s) και πιστότητας (Faithfulness). Τα σημεία κατά μήκος της διακεκομμένης γραμμής ορίζουν το βέλτιστο μέτωπο υπεροχής Pareto.",
                badge="Pareto",
                icon="show_chart",
                is_judge_graph=True,
                peek_primary_key="pareto_primary_chart",
                peek_compare_key="pareto_compare_chart"
            )
            fig_pareto_primary = _build_pareto_fig(plot_df)
            fig_pareto_compare = _build_pareto_fig(exp_compare_df)

            st.plotly_chart(fig_pareto_primary, width="stretch", key="pareto_primary_chart")
            st.plotly_chart(fig_pareto_compare, width="stretch", key="pareto_compare_chart")

    with col_p2:
        with st.container(border=True):
            SCALING_MODES = [
                {
                    "id": "speed",
                    "metric_col": "avg_speed_tps",
                    "title": "Throughput vs. Parameter Size",
                    "subtitle": "Καθαρή ταχύτητα παραγωγής κειμένου (tokens/sec) μετά τη λήψη του πρώτου token, σε συνάρτηση με το μέγεθος παραμέτρων.",
                    "y_label": "Throughput (tok/s)",
                    "fmt": "%{text:.1f} t/s",
                    "badge": "Generation Speed"
                },
                {
                    "id": "ttft",
                    "metric_col": "avg_ttft_sec",
                    "title": "Time to First Token (TTFT)",
                    "subtitle": "Χρόνος αναμονής (ανάκτηση, reranking και επεξεργασία prompt) μέχρι την παραγωγή του 1ου token από το μοντέλο.",
                    "y_label": "Time to First Token (seconds)",
                    "fmt": "%{text:.2f} s",
                    "badge": "TTFT Latency"
                },
                {
                    "id": "total_time",
                    "metric_col": "avg_total_duration_sec",
                    "title": "Total Query Cycle Time",
                    "subtitle": "Συνολικός χρόνος κύκλου από την υποβολή του ερωτήματος μέχρι την πλήρη ολοκλήρωση της παραγωγής απάντησης.",
                    "y_label": "Total Latency (seconds)",
                    "fmt": "%{text:.1f} s",
                    "badge": "End-to-End Latency"
                }
            ]

            scaling_idx_key = "scaling_metric_mode_idx"
            if scaling_idx_key not in st.session_state:
                st.session_state[scaling_idx_key] = 0

            def _prev_scaling_mode():
                st.session_state[scaling_idx_key] = (st.session_state[scaling_idx_key] - 1) % len(SCALING_MODES)

            def _next_scaling_mode():
                st.session_state[scaling_idx_key] = (st.session_state[scaling_idx_key] + 1) % len(SCALING_MODES)

            curr_idx = st.session_state[scaling_idx_key] % len(SCALING_MODES)
            active_mode = SCALING_MODES[curr_idx]

            col_top_h, col_top_nav = st.columns([1.0, 0.12])
            with col_top_h:
                render_card_header(
                    title=active_mode["title"],
                    subtitle=active_mode["subtitle"],
                    badge=active_mode["badge"],
                    icon="bolt"
                )
            with col_top_nav:
                c_nav1, c_nav2 = st.columns(2)
                with c_nav1:
                    st.button(":material/chevron_left:", key="btn_prev_scaling_mode", on_click=_prev_scaling_mode)
                with c_nav2:
                    st.button(":material/chevron_right:", key="btn_next_scaling_mode", on_click=_next_scaling_mode)

            model_label_map = {
                "llama3.2:3b": "Llama 3.2 (3.2B)",
                "llama3.1:8b": "Llama 3.1 (8.0B)",
                "gemma4:latest": "Gemma 4 (8.0B)",
                "deepseek-r1:8b": "DeepSeek R1 (8.2B)"
            }
            plot_df["model_display"] = plot_df["model_name"].map(lambda x: model_label_map.get(str(x), str(x)))

            config_label_map = {
                "CFG-1": "CFG-1 (Dense)",
                "CFG-2": "CFG-2 (Hybrid)",
                "CFG-3": "CFG-3 (Rerank)",
                "CFG-4": "CFG-4 (HyDE Full)",
                "CFG-4-LITE": "CFG-4-LITE (HyDE Lite)"
            }
            plot_df["config_display"] = plot_df["config_id"].map(lambda x: config_label_map.get(str(x), str(x)))

            fig_scaling = px.bar(
                plot_df.sort_values(["param_size_num", "config_id"]),
                x="model_display",
                y=active_mode["metric_col"],
                color="config_display",
                barmode="group",
                text=active_mode["metric_col"],
                hover_name="experiment_name",
                hover_data=["config_name", "avg_ttft_sec", "avg_total_duration_sec", "avg_speed_tps"],
                labels={
                    "model_display": "Model & Parameter Tier",
                    active_mode["metric_col"]: active_mode["y_label"],
                    "config_display": "Pipeline Config"
                },
                color_discrete_map={
                    "CFG-1 (Dense)": "#38bdf8",
                    "CFG-2 (Hybrid)": "#818cf8",
                    "CFG-3 (Rerank)": "#fb7185",
                    "CFG-4 (HyDE Full)": "#34d399",
                    "CFG-4-LITE (HyDE Lite)": "#a7f3d0"
                }
            )
            fig_scaling.update_traces(
                texttemplate=active_mode["fmt"],
                textposition='outside',
                textfont=dict(size=10.5, color="#cbd5e1", family="'AHG', sans-serif"),
                marker=dict(line=dict(width=1, color="rgba(255, 255, 255, 0.15)"))
            )
            apply_clean_chart_theme(fig_scaling, height=420)
            fig_scaling.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.03,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=11, color="#cbd5e1"),
                    bgcolor="rgba(20,20,19,0.75)",
                    bordercolor="rgba(255,255,255,0.1)"
                ),
                margin=dict(l=50, r=25, t=55, b=45)
            )
            st.plotly_chart(fig_scaling, width="stretch")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        with st.container(border=True):
            render_card_header(
                title="Quality Profile",
                subtitle="Πολυαξονική αποτίμηση των μοντέλων στις 4 διαστάσεις ποιότητας Ragas: Faithfulness (πιστότητα), Relevancy (συνάφεια), Context Precision (ακρίβεια) και Context Recall (πληρότητα).",
                badge="Profile",
                icon="radar",
                is_judge_graph=True,
                peek_primary_key="radar_primary_chart",
                peek_compare_key="radar_compare_chart"
            )

            plot_source_df = exp_df.copy() if not exp_df.empty else scored_df
            plot_radar_df = pd.DataFrame(plot_source_df.drop_duplicates(subset=["model_name", "config_id"], keep="first"))

            edge_ids = []
            if not scored_df.empty:
                best_faith = scored_df.iloc[int(np.argmax(np.asarray(scored_df["avg_faithfulness"])))]
                best_prec = scored_df.iloc[int(np.argmax(np.asarray(scored_df["avg_context_precision"])))]
                best_rec = scored_df.iloc[int(np.argmax(np.asarray(scored_df["avg_context_recall"])))]
                fastest_exp = scored_df.iloc[int(np.argmax(np.asarray(scored_df["avg_speed_tps"])))]
                edge_ids = list({best_faith["id"], best_prec["id"], best_rec["id"], fastest_exp["id"]})

                l32_cfg1_list = [r["id"] for _, r in scored_df.iterrows() if "llama3.2" in str(r.get("model_name", "")).lower() and r.get("config_id") == "CFG-1"]
                if l32_cfg1_list and l32_cfg1_list[0] not in edge_ids:
                    edge_ids.append(l32_cfg1_list[0])

            if not edge_ids and not plot_radar_df.empty:
                edge_ids = plot_radar_df["id"].tolist()[:4]

            trace_label_map = {}
            for _, r in plot_radar_df.iterrows():
                m_clean = _clean_model_display(r.get("model_name", ""))
                c_clean = _fmt_cfg(r.get("config_id", "CFG"))
                trace_label_map[r["id"]] = f"{m_clean} ({c_clean})"

            all_opts = list(trace_label_map.keys())
            radar_sel_key = "radar_selected_model_ids"
            if radar_sel_key not in st.session_state:
                st.session_state[radar_sel_key] = [i for i in edge_ids if i in all_opts] if edge_ids else all_opts[:4]

            col_rd_h1, col_rd_h2 = st.columns([1.3, 0.7])
            with col_rd_h2:
                with st.popover("Επιλογή Μοντέλων", icon=":material/tune:", width="stretch"):
                    st.markdown("###### Προβολή Μοντέλων στο Radar")
                    col_b1, col_b2, col_b3 = st.columns(3)
                    with col_b1:
                        if st.button("Preselected", width="stretch", key="radar_p_btn_frontier"):
                            st.session_state[radar_sel_key] = [i for i in edge_ids if i in all_opts]
                            st.rerun()
                    with col_b2:
                        if st.button("Select All", width="stretch", key="radar_p_btn_all"):
                            st.session_state[radar_sel_key] = all_opts
                            st.rerun()
                    with col_b3:
                        if st.button("Clear", width="stretch", key="radar_p_btn_clear"):
                            st.session_state[radar_sel_key] = []
                            st.rerun()

                    current_sel = [i for i in st.session_state.get(radar_sel_key, []) if i in all_opts]
                    selected_ids = st.multiselect(
                        "Μοντέλα & Configurations:",
                        options=all_opts,
                        default=current_sel,
                        format_func=lambda x: trace_label_map.get(x, str(x)),
                        key="ms_radar_popover_sel"
                    )
                    if set(selected_ids) != set(current_sel):
                        st.session_state[radar_sel_key] = selected_ids
                        st.rerun()

            active_trace_ids = set(st.session_state.get(radar_sel_key, []))

            def _build_radar_fig(source_df: pd.DataFrame, trace_ids: set, labels_map: dict) -> go.Figure:
                f_rad = go.Figure()
                categories = ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"]
                r_cats = categories + [categories[0]]

                for _, row in source_df.iterrows():
                    rid = row["id"]
                    if rid not in trace_ids:
                        continue

                    f_val = row.get("avg_faithfulness")
                    a_val = row.get("avg_answer_relevancy")
                    p_val = row.get("avg_context_precision")
                    r_val = row.get("avg_context_recall")
                    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in [f_val, a_val, p_val, r_val]):
                        continue

                    m_raw = str(row.get("model_name", ""))
                    m_name = _clean_model_display(m_raw)
                    m_color = MODEL_COLORS.get(m_raw, "#94a3b8")

                    vals = [
                        float(f_val),
                        float(a_val),
                        float(p_val),
                        float(r_val)
                    ]
                    vals.append(vals[0])
                    lbl = labels_map.get(rid, f"{m_name} ({row.get('config_id', 'CFG')})")

                    f_rad.add_trace(go.Scatterpolar(
                        r=vals,
                        theta=r_cats,
                        fill="toself",
                        name=lbl,
                        mode="lines+markers",
                        marker=dict(size=7),
                        line=dict(color=m_color, width=2.5),
                        fillcolor=f"rgba({int(m_color[1:3],16)}, {int(m_color[3:5],16)}, {int(m_color[5:7],16)}, 0.08)" if len(m_color)==7 and m_color.startswith('#') else None,
                        visible=True
                    ))

                f_rad.update_layout(
                    polar=dict(
                        domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),
                        radialaxis=dict(
                            visible=True,
                            range=[0.4, 0.9],
                            dtick=0.1,
                            gridcolor="rgba(148,163,184,0.14)",
                            tickfont=dict(size=9.5, color="#94a3b8")
                        ),
                        angularaxis=dict(
                            gridcolor="rgba(148,163,184,0.18)",
                            tickfont=dict(size=12, color="#f8fafc", family="'AHG', 'Segoe UI', sans-serif"),
                            rotation=45
                        ),
                        bgcolor="rgba(0,0,0,0)"
                    ),
                    showlegend=False,
                    margin=dict(l=15, r=15, t=15, b=15)
                )
                apply_clean_chart_theme(f_rad, height=500)
                return f_rad

            fig_radar_primary = _build_radar_fig(plot_radar_df, active_trace_ids, trace_label_map)
            fig_radar_compare = _build_radar_fig(exp_compare_df, active_trace_ids, trace_label_map)

            st.plotly_chart(fig_radar_primary, width="stretch", key="radar_primary_chart")
            st.plotly_chart(fig_radar_compare, width="stretch", key="radar_compare_chart")

    with col_r2:
        with st.container(border=True):
            render_card_header(
                title="Latency Breakdown",
                subtitle="Ανάλυση του συνολικού χρόνου εκτέλεσης σε επιμέρους στάδια: προ-παραγωγή HyDE, διανυσματική ανάκτηση, συμπίεση Cross-Encoder και παραγωγή απάντησης.",
                badge="Latency",
                icon="timer"
            )
            lat_rows = []
            for _, r in exp_df.iterrows():
                cfg = str(r.get("config_id", "")).upper()
                m_clean = _clean_model_display(r.get("model_name", ""))
                c_clean = _fmt_cfg(cfg)
                exp_label = f"{m_clean} • {c_clean}"

                raw_ret = float(r.get("avg_retrieval_duration_sec") or 0.0)
                raw_rerank = float(r.get("avg_rerank_duration_sec") or 0.0)
                raw_gen = float(r.get("avg_generation_duration_sec") or 0.0)
                total_dur = float(r.get("avg_total_duration_sec") or r.get("avg_duration_sec") or 0.0)

                if "CFG-4-LITE" in cfg or "CFG-4-LITE" in str(r.get("experiment_name", "")):
                    pure_ret = raw_ret
                    hyde_s = max(total_dur - (pure_ret + raw_rerank + raw_gen), 0.0) if total_dur > 0 else 0.0
                elif "CFG-4" in cfg or "CFG4" in str(r.get("experiment_name", "")):
                    # In monolithic CFG-4, HyDE 8B ran inside the retrieval timer hook.
                    # Derive pure vector retrieval dynamically from the model's CFG-1 baseline.
                    m_base = exp_df[(exp_df["model_name"] == r.get("model_name")) & (exp_df["config_id"].str.upper() == "CFG-1")]["avg_retrieval_duration_sec"]
                    pure_ret = float(m_base.iloc[0]) if not m_base.empty and pd.notna(m_base.iloc[0]) else 0.287
                    pure_ret = min(pure_ret, raw_ret)
                    hyde_s = max(raw_ret - pure_ret, 0.0)
                else:
                    hyde_s = 0.0
                    pure_ret = raw_ret

                lat_rows.append({
                    "Architecture": exp_label,
                    "HyDE Pre-generation": round(hyde_s, 2),
                    "Pure Vector Retrieval": round(pure_ret, 3),
                    "Cross-Encoder Rerank": round(raw_rerank, 3),
                    "Final LLM Generation": round(raw_gen, 2),
                    "Total Cycle Time": round(hyde_s + pure_ret + raw_rerank + raw_gen, 2)
                })

            lat_df = pd.DataFrame(lat_rows).drop_duplicates(subset=["Architecture"], keep="first")
            lat_df = lat_df.sort_values("Total Cycle Time", ascending=True)

            fig_lat = px.bar(
                lat_df,
                x="Architecture",
                y=["HyDE Pre-generation", "Pure Vector Retrieval", "Cross-Encoder Rerank", "Final LLM Generation"],
                barmode="stack",
                labels={"value": "Χρόνος (δευτερόλεπτα)", "variable": "Στάδιο Pipeline", "Architecture": "Μοντέλο & Διαμόρφωση"},
                color_discrete_map={
                    "HyDE Pre-generation": "#a855f7",
                    "Pure Vector Retrieval": "#38bdf8",
                    "Cross-Encoder Rerank": "#fbbf24",
                    "Final LLM Generation": "#fb7185"
                }
            )
            apply_clean_chart_theme(fig_lat, height=542)
            fig_lat.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10, color="#cbd5e1"),
                    bgcolor="rgba(20,20,19,0.75)",
                    bordercolor="rgba(255,255,255,0.1)"
                ),
                margin=dict(l=50, r=25, t=40, b=55)
            )
            st.plotly_chart(fig_lat, width="stretch")

    with st.container(border=True):
        render_card_header(
            title="Leaderboard",
            subtitle="Αναλυτικός συγκριτικός πίνακας επιδόσεων, μετρικών ποιότητας Ragas και χρονικών καθυστερήσεων για όλες τις πειραματικές διατάξεις τοπικών μοντέλων (3.2B–8B).",
            badge=f"Judge: {active_provider.display_name}",
            icon="table_chart",
            is_judge_graph=False
        )

        def _prep_leaderboard_df(source_exp_df):
            d_df = source_exp_df.copy() if not source_exp_df.empty else pd.DataFrame()
            if not d_df.empty:
                d_df["scored_coverage"] = d_df.apply(
                    lambda r: f"{int(r.get('scored_questions_count', 0))}/{int(r.get('sample_size', 103))}" if r.get('sample_size') else "0/103",
                    axis=1
                )
                d_df["judge_status_disp"] = d_df["judge_status"].map(
                    lambda s: "✅ Completed" if s == "completed" else ("⏳ Partial" if s == "partial" else "⚪ Unscored")
                )
            return d_df

        disp_df_primary = _prep_leaderboard_df(exp_df)

        display_cols = [
            "id", "experiment_name", "model_name", "model_params", "config_name",
            "avg_faithfulness", "avg_answer_relevancy", "avg_context_precision", "avg_context_recall",
            "avg_speed_tps", "avg_ttft_sec", "avg_total_duration_sec", "scored_coverage", "judge_status_disp", "timestamp"
        ]
        existing_disp_cols_primary = [c for c in display_cols if c in disp_df_primary.columns]

        col_config = {
            "id": st.column_config.NumberColumn("ID", width="small"),
            "avg_faithfulness": st.column_config.ProgressColumn("Faithfulness", min_value=0.0, max_value=1.0, format="%.3f"),
            "avg_answer_relevancy": st.column_config.ProgressColumn("Relevancy", min_value=0.0, max_value=1.0, format="%.3f"),
            "avg_context_precision": st.column_config.ProgressColumn("Precision", min_value=0.0, max_value=1.0, format="%.3f"),
            "avg_context_recall": st.column_config.ProgressColumn("Recall", min_value=0.0, max_value=1.0, format="%.3f"),
            "avg_speed_tps": st.column_config.NumberColumn("Speed (t/s)", format="%.1f"),
            "avg_ttft_sec": st.column_config.NumberColumn("TTFT (s)", format="%.2f"),
            "avg_total_duration_sec": st.column_config.NumberColumn("Total (s)", format="%.2f"),
            "scored_coverage": st.column_config.TextColumn("Scored"),
            "judge_status_disp": st.column_config.TextColumn("Judge Status"),
        }

        st.dataframe(
            disp_df_primary[existing_disp_cols_primary],
            column_config=col_config,
            hide_index=True,
            width="stretch",
            key=f"leaderboard_table_{active_judge}"
        )

        col_bt1, col_bt2 = st.columns([1.6, 0.4])
        with col_bt1:
            csv_data = exp_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Export Leaderboard CSV",
                data=csv_data,
                file_name=f"thesis_rag_benchmark_leaderboard_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                icon=":material/download:"
            )
        with col_bt2:
            if not exp_df.empty:
                with st.popover("Διαγραφή Run", icon=":material/delete:", width="stretch"):
                    st.markdown("###### Διαγραφή Εκτέλεσης")
                    exp_options = exp_df["id"].tolist()
                    exp_labels = {
                        r["id"]: f"Run #{r['id']}: {_clean_model_display(r.get('model_name',''))} ({_fmt_cfg(r.get('config_id',''))})"
                        for _, r in exp_df.iterrows()
                    }
                    del_exp_id = st.selectbox(
                        "Επιλογή",
                        options=exp_options,
                        format_func=lambda x: exp_labels.get(x, f"Run #{x}"),
                        key="pop_del_exp_id"
                    )
                    if st.button("Οριστική Διαγραφή", type="primary", width="stretch", key="btn_confirm_delete"):
                        if delete_experiment_from_db(del_exp_id):
                            st.success(f"Διαγράφηκε η εκτέλεση #{del_exp_id}")
                            st.rerun(scope="app")
                        else:
                            st.error(f"Αποτυχία διαγραφής της εκτέλεσης #{del_exp_id}")
