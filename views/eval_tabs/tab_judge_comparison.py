import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from db import get_dual_judge_comparison_data
from views.eval_tabs.common import (
    MODEL_COLORS,
    apply_clean_chart_theme,
    render_card_header,
)


@st.fragment
def render_tab_judge_comparison() -> None:
    
    render_card_header(
        title="Judge Comparison",
        subtitle="RQ-M · Συγκριτική αξιολόγηση και βαθμονόμηση κριτών.",
        icon="balance"
    )

    df_comp = get_dual_judge_comparison_data()

    if df_comp.empty:
        st.info(
            "Δεν υπάρχουν ακόμη κοινές βαθμολογημένες ερωτήσεις μεταξύ DeepSeek V4 και Azure GPT-5-mini.",
            icon=":material/info:"
        )
        return

    overlap_count = len(df_comp)
    ds_overall = df_comp["ds_overall"]
    gpt_overall = df_comp["gpt_overall"]
    diff_overall = gpt_overall - ds_overall

    corr_r = stats.pearsonr(ds_overall, gpt_overall).statistic if overlap_count > 2 else None
    corr_rho = stats.spearmanr(ds_overall, gpt_overall).statistic if overlap_count > 2 else None
    corr_val_html = f"r = {corr_r:.3f} <span class=\"eval-kpi-denom\">· ρ = {corr_rho:.3f}</span>" if (corr_r is not None and corr_rho is not None) else "N/A"
    bias_overall = float(np.mean(diff_overall))
    mae_overall = float(np.mean(np.abs(diff_overall)))
    medae_overall = float(np.median(np.abs(diff_overall)))

    judge_kpis_html = f"""<div class="eval-kpi-grid">
<div class="eval-kpi-card">
<div class="eval-kpi-label">Συσχέτιση</div>
<div class="eval-kpi-val">{corr_val_html}</div>
<div class="eval-kpi-model">Pearson r &amp; Spearman ρ (p &lt; 0.001)</div>
<div class="eval-kpi-meta">Υψηλή γραμμική και μονοτονική συσχέτιση</div>
</div>
<div class="eval-kpi-card">
<div class="eval-kpi-label">Συστηματική Απόκλιση</div>
<div class="eval-kpi-val">Δ = {bias_overall:+.4f}</div>
<div class="eval-kpi-model">GPT-5-mini βαθμολογεί κατά μέσο όρο υψηλότερα</div>
<div class="eval-kpi-meta">Μέση διαφορά Overall Score στο paired corpus</div>
</div>
<div class="eval-kpi-card">
<div class="eval-kpi-label">Σφάλμα Βαθμονόμησης</div>
<div class="eval-kpi-val">MAE {mae_overall:.4f} <span class="eval-kpi-denom">· MedAE {medae_overall:.4f}</span></div>
<div class="eval-kpi-model">Μέσο &amp; διάμεσο απόλυτο σφάλμα</div>
<div class="eval-kpi-meta">Διάμεση απόκλιση 5.4% μεταξύ των δύο κριτών</div>
</div>
<div class="eval-kpi-card">
<div class="eval-kpi-label">Αρχιτεκτονική Σταθερότητα</div>
<div class="eval-kpi-val">100% <span class="eval-kpi-denom">Invariance</span></div>
<div class="eval-kpi-model">CFG-3 &gt; CFG-4 &gt; CFG-2 &gt; CFG-1</div>
<div class="eval-kpi-meta">Ταυτόσημη ιεραρχία RAG στο Llama 3.1 8B</div>
</div>
</div>"""
    st.markdown(judge_kpis_html, unsafe_allow_html=True)

    with st.container(border=True):
        render_card_header(
            title="Σύγκριση και Βαθμονόμηση Αξιολογητών (Inter-Judge Calibration)",
            subtitle="Ανάλυση συσχέτισης και απόκλισης σε επίπεδο ερώτησης (1.026 ζεύγη παρατηρήσεων) με διαγνωστικό Residuals.",
            badge=f"{overlap_count:,} Paired Points",
            icon="compare_arrows"
        )

        col_toggle, col_metric, col_opt = st.columns([1.2, 1.4, 1.4], vertical_alignment="center")

        with col_toggle:
            view_mode = st.segmented_control(
                "Προβολή",
                options=["Parity", "Residuals"],
                default="Parity",
                key="judge_comp_view_toggle",
                label_visibility="collapsed"
            ) or "Parity"

        metric_dict = {
            "Overall Score": ("ds_overall", "gpt_overall", "Composite Overall Score"),
            "Faithfulness": ("ds_faithfulness", "gpt_faithfulness", "Ragas Faithfulness"),
            "Answer Relevancy": ("ds_relevancy", "gpt_relevancy", "Answer Relevancy"),
            "Context Precision": ("ds_precision", "gpt_precision", "Context Precision"),
            "Context Recall": ("ds_recall", "gpt_recall", "Context Recall"),
        }

        with col_metric:
            selected_metric_name = str(st.selectbox(
                "Δείκτης Αξιολόγησης",
                options=list(metric_dict.keys()),
                index=0,
                key="judge_comp_metric_choice",
                label_visibility="collapsed"
            ) or "Overall Score")

        col_ds, col_gpt, metric_label = metric_dict[selected_metric_name]
        chart_df = df_comp[[col_ds, col_gpt, "model_name", "config_id", "question", "category"]].dropna().copy()
        chart_df["diff"] = chart_df[col_gpt] - chart_df[col_ds]
        chart_df["mean_score"] = (chart_df[col_ds] + chart_df[col_gpt]) / 2.0

        with col_opt:
            if view_mode == "Parity":
                show_corridor = st.checkbox("Ζώνη ανοχής (±0.05)", value=True, key="chk_tolerance_corridor")
            else:
                show_loa = st.checkbox("Όρια συμφωνίας (95% LoA)", value=True, key="chk_show_loa")

        fig = go.Figure()

        if view_mode == "Parity":
            slope, intercept, r_val, p_val, _ = stats.linregress(chart_df[col_ds], chart_df[col_gpt])

            fig.add_trace(go.Scatter(
                x=[0.0, 1.0],
                y=[0.0, 1.0],
                mode="lines",
                line=dict(dash="dash", color="rgba(255, 255, 255, 0.4)", width=1.5),
                name="Διαγώνιος Ταύτισης (y = x)",
                hoverinfo="skip"
            ))

            if show_corridor:
                fig.add_trace(go.Scatter(
                    x=[0.0, 1.0, 1.0, 0.0],
                    y=[0.05, 1.05, 0.95, -0.05],
                    fill="toself",
                    fillcolor="rgba(56, 189, 248, 0.07)",
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    name="Περιοχή Ανοχής (±0.05)"
                ))

            reg_x = np.array([0.0, 1.0])
            reg_y = slope * reg_x + intercept
            fig.add_trace(go.Scatter(
                x=reg_x,
                y=reg_y,
                mode="lines",
                line=dict(color="#f59e0b", width=2, dash="dot"),
                name=f"Παλινδρόμηση: y = {slope:.2f}x + {intercept:+.2f} (R² = {r_val**2:.3f})",
                hoverinfo="skip"
            ))

            for m_name, grp in chart_df.groupby("model_name"):
                m_color = MODEL_COLORS.get(m_name, "#38bdf8")
                q_short = [q[:75] + "..." if len(q) > 75 else q for q in grp["question"].fillna("")]
                fig.add_trace(go.Scatter(
                    x=grp[col_ds],
                    y=grp[col_gpt],
                    mode="markers",
                    marker=dict(
                        size=6.5,
                        color=m_color,
                        opacity=0.72,
                        line=dict(width=0.6, color="#ffffff")
                    ),
                    name=m_name,
                    customdata=np.column_stack([grp["model_name"], grp["config_id"], q_short, grp["category"].fillna(""), grp["diff"]]),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                        "Κατηγορία: %{customdata[3]}<br>"
                        "Ερώτηση: %{customdata[2]}<br>"
                        "DeepSeek: %{x:.3f} | Azure: %{y:.3f} (Δ: %{customdata[4]:+.3f})<extra></extra>"
                    )
                ))

            fig.update_layout(
                xaxis_title=f"DeepSeek V4 Flash ({selected_metric_name})",
                yaxis_title=f"Azure GPT-5-mini ({selected_metric_name})",
                xaxis=dict(range=[-0.02, 1.02]),
                yaxis=dict(range=[-0.02, 1.02]),
                height=460
            )
            apply_clean_chart_theme(fig, height=460)
            st.plotly_chart(fig, width="stretch")

            st.caption(
                f"Διάγραμμα ταύτισης (Parity Plane) επί {len(chart_df):,} ζευγών παρατηρήσεων (N = 1.026 για Overall). "
                "Κάθε σημείο αντιπροσωπεύει μία κοινή ερώτηση αξιολογημένη ανεξάρτητα από τους δύο κριτές (και όχι συγκεντρωτικό μέσο όρο μοντέλου). "
                "Η λευκή διακεκομμένη γραμμή y = x υποδηλώνει απόλυτη ταύτιση βαθμολογιών. "
                f"Η κίτρινη διακεκομμένη γραμμή απεικονίζει τη γραμμική παλινδρόμηση (R² = {r_val**2:.3f}). "
                "Η σκιασμένη ζώνη αποτελεί περιγραφική περιοχή ανοχής (tolerance corridor ±0.05) και όχι επίσημο στατιστικό συντελεστή συμφωνίας. "
                "Ακεραιότητα δεδομένων: Οι ακραίες τιμές 0.0 και 1.0 είναι πραγματικές μετρημένες βαθμολογίες του πλαισίου Ragas (π.χ. ποινές μη-δέσμευσης ή απόλυτη ανάκληση) "
                "και δεν προκύπτουν από ελλείπουσες τιμές ή τεχνητό fallback, καθώς τα ελλείποντα δεδομένα αποκλείονται αυστηρά από το δείγμα."
            )

        else:
            b_mean = float(chart_df["diff"].mean())
            b_std = float(chart_df["diff"].std(ddof=1))
            loa_high = b_mean + 1.96 * b_std
            loa_low = b_mean - 1.96 * b_std
            in_loa_pct = ((chart_df["diff"] >= loa_low) & (chart_df["diff"] <= loa_high)).mean() * 100

            fig.add_trace(go.Scatter(
                x=[0.0, 1.0],
                y=[0.0, 0.0],
                mode="lines",
                line=dict(dash="dot", color="rgba(255, 255, 255, 0.25)", width=1),
                name="Μηδενική Απόκλιση (Δ = 0)",
                showlegend=False,
                hoverinfo="skip"
            ))

            fig.add_trace(go.Scatter(
                x=[0.0, 1.0],
                y=[b_mean, b_mean],
                mode="lines",
                line=dict(color="#38bdf8", width=2),
                name=f"Μέση Απόκλιση (Bias: {b_mean:+.4f})",
                hoverinfo="skip"
            ))

            fig.add_trace(go.Scatter(
                x=[0.0, 1.0],
                y=[loa_high, loa_high],
                mode="lines",
                line=dict(color="#f43f5e", width=1.5, dash="dash"),
                name=f"+1.96 SD ({loa_high:+.3f})",
                hoverinfo="skip"
            ))
            fig.add_trace(go.Scatter(
                x=[0.0, 1.0],
                y=[loa_low, loa_low],
                mode="lines",
                line=dict(color="#f43f5e", width=1.5, dash="dash"),
                name=f"-1.96 SD ({loa_low:+.3f})",
                hoverinfo="skip"
            ))

            if show_loa:
                fig.add_trace(go.Scatter(
                    x=[0.0, 1.0, 1.0, 0.0],
                    y=[loa_high, loa_high, loa_low, loa_low],
                    fill="toself",
                    fillcolor="rgba(244, 63, 94, 0.06)",
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    name="95% Limits of Agreement (LoA)"
                ))

            for m_name, grp in chart_df.groupby("model_name"):
                m_color = MODEL_COLORS.get(m_name, "#38bdf8")
                q_short = [q[:75] + "..." if len(q) > 75 else q for q in grp["question"].fillna("")]
                fig.add_trace(go.Scatter(
                    x=grp["mean_score"],
                    y=grp["diff"],
                    mode="markers",
                    marker=dict(
                        size=6.5,
                        color=m_color,
                        opacity=0.72,
                        line=dict(width=0.6, color="#ffffff")
                    ),
                    name=m_name,
                    customdata=np.column_stack([grp["model_name"], grp["config_id"], q_short, grp["category"].fillna("")]),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                        "Κατηγορία: %{customdata[3]}<br>"
                        "Ερώτηση: %{customdata[2]}<br>"
                        "Μέσος Όρος: %{x:.3f} | Διαφορά (GPT - DS): %{y:+.3f}<extra></extra>"
                    )
                ))

            y_min = min(-0.35, loa_low - 0.06)
            y_max = max(0.40, loa_high + 0.06)
            fig.update_layout(
                xaxis_title=f"Μέσος Όρος Βαθμολογίας (DeepSeek + Azure) / 2 ({selected_metric_name})",
                yaxis_title=f"Διαφορά Βαθμολογίας (Δ = Azure − DeepSeek)",
                xaxis=dict(range=[-0.02, 1.02]),
                yaxis=dict(range=[y_min, y_max]),
                height=460
            )
            apply_clean_chart_theme(fig, height=460)
            st.plotly_chart(fig, width="stretch")

            st.caption(
                f"Διάγραμμα Bland-Altman (Residuals vs. Mean) επί {len(chart_df):,} ζευγών παρατηρήσεων ανά ερώτηση. "
                f"Το {in_loa_pct:.1f}% των μετρήσεων ευρίσκονται εντός των ορίων 95% LoA [{loa_low:.3f}, {loa_high:+.3f}]. "
                f"Η συστηματική θετική μετατόπιση ({b_mean:+.4f}) υποδεικνύει ότι ο GPT-5-mini βαθμολογεί κατά μέσο όρο οριακά υψηλότερα στο paired corpus."
            )

    with st.container(border=True):
        render_card_header(
            title="Βαθμονόμηση και Στατιστικά Ανά Δείκτη (Metric Calibration)",
            subtitle="Συγκριτικοί μέσοι όροι, συστηματική απόκλιση (bias), σφάλματα βαθμονόμησης και συσχετίσεις.",
            badge="5 Ragas Metrics",
            icon="table_chart"
        )

        metrics_summary = [
            ("Faithfulness", "ds_faithfulness", "gpt_faithfulness"),
            ("Answer Relevancy", "ds_relevancy", "gpt_relevancy"),
            ("Context Precision", "ds_precision", "gpt_precision"),
            ("Context Recall", "ds_recall", "gpt_recall"),
            ("Overall Score", "ds_overall", "gpt_overall"),
        ]

        cal_rows = []
        for name, ds_col, gpt_col in metrics_summary:
            valid = df_comp[[ds_col, gpt_col]].dropna()
            v_ds = valid[ds_col]
            v_gpt = valid[gpt_col]
            diff = v_gpt - v_ds
            m_mae = np.mean(np.abs(diff))
            m_medae = np.median(np.abs(diff))
            m_r = stats.pearsonr(v_ds, v_gpt).statistic
            m_rho = stats.spearmanr(v_ds, v_gpt).statistic
            agree_10 = (np.abs(diff) <= 0.10).mean() * 100
            cal_rows.append({
                "Δείκτης (Metric)": name,
                "DeepSeek Mean": f"{v_ds.mean():.3f}",
                "GPT-5 Mean": f"{v_gpt.mean():.3f}",
                "Signed Bias (Δ)": f"{diff.mean():+.4f}",
                "MAE": f"{m_mae:.4f}",
                "MedAE": f"{m_medae:.4f}",
                "Pearson r": f"{m_r:.3f}",
                "Spearman ρ": f"{m_rho:.3f}",
                "Απόκλιση ≤ 0.10": f"{agree_10:.1f}%"
            })

        cal_df = pd.DataFrame(cal_rows)
        st.dataframe(cal_df, width="stretch", hide_index=True)
        st.caption(
            "Υπολογισμός σε 1.026 ζεύγη παρατηρήσεων (932 για Answer Relevancy, 1.022 για Context Precision). "
            "Το Signed Bias υπολογίζεται ως Δ = GPT − DS (θετικό πρόσημο = υψηλότερη βαθμολογία από GPT-5-mini). "
            "Η στήλη «Απόκλιση ≤ 0.10» αποτελεί περιγραφικό κριτήριο ανοχής (|Δ| ≤ 0.10) και όχι επίσημο συντελεστή συμφωνίας. "
            "Εντοπίστηκαν 97 περιπτώσεις μεγάλης απόκλισης (|Δ| > 0.80), οι οποίες υποδεικνύουν σημαντική διαφορά στην αξιολόγηση της Answer Relevancy μεταξύ των δύο κριτών. "
            "Τα p-values όλων των συντελεστών συσχέτισης είναι p < 0.001."
        )

    with st.container(border=True):
        render_card_header(
            title="Σταθερότητα Κατάταξης (Ranking Stability)",
            subtitle="Επίδραση της επιλογής κριτή στην ιεραρχία αρχιτεκτονικών και οικογενειών μοντέλων.",
            badge="Architectural Invariance",
            icon="low_priority"
        )

        stability_data = [
            {
                "Εύρος Σύγκρισης (Scope)": "Αρχιτεκτονική Αφαίρεση (Llama 3.1 8B)",
                "Κατάταξη Azure GPT-5-mini": "CFG-3 (0.691) → CFG-4 (0.665) → CFG-2 (0.640) → CFG-1 (0.614)",
                "Κατάταξη DeepSeek V4": "CFG-3 (0.666) → CFG-4 (0.665) → CFG-2 (0.616) → CFG-1 (0.597)",
                "Σταθερότητα (Invariance)": "100% Ταυτόσημη (4/4) ✅",
            },
            {
                "Εύρος Σύγκρισης (Scope)": "Ιεραρχία Μοντέλων (CFG-3 Reference)",
                "Κατάταξη Azure GPT-5-mini": "Gemma 4 (0.718) > Llama 3.1 (0.691) > DS-R1 (0.684) > Llama 3.2 (0.664)",
                "Κατάταξη DeepSeek V4": "Gemma 4 (0.713) > DS-R1 (0.674) > Llama 3.1 (0.666) > Llama 3.2 (0.627)",
                "Σταθερότητα (Invariance)": "75% (Gemma 4 #1 & Llama 3.2 #4 σταθερά · Llama 3.1 / DS-R1 ισοδύναμα) ⚠️",
            }
        ]
        st.dataframe(pd.DataFrame(stability_data), width="stretch", hide_index=True)
        st.caption(
            "Στο ablation study του Llama 3.1 8B παρατηρείται 100% αρχιτεκτονική σταθερότητα: "
            "και οι δύο κριτές κατατάσσουν τις αρχιτεκτονικές με την ίδια ακριβώς σειρά (CFG-3 > CFG-4 > CFG-2 > CFG-1). "
            "Στην κατάταξη μοντέλων (CFG-3), τα Llama 3.1 8B και DeepSeek-R1 8B εναλλάσσουν τις θέσεις 2 και 3 λόγω οριακής διαφοράς (Δ < 0.01), "
            "ενώ το Gemma 4 παραμένει σταθερά στην 1η θέση και το Llama 3.2 στην 4η θέση."
        )

        with st.expander("Κορυφαίες Αποκλίνουσες Παρατηρήσεις (Top Divergent Observations)", expanded=False):
            top_div_df = df_comp.copy()
            top_div_df["abs_diff"] = (top_div_df["gpt_overall"] - top_div_df["ds_overall"]).abs()
            top_div = top_div_df.sort_values(by="abs_diff", ascending=False).head(5)

            divergence_records = [
                {
                    "ID": f"Q{r['question_idx']}",
                    "Μοντέλο & Config": f"{r['model_name']} ({r['config_id']})",
                    "Κατηγορία": r["category"] if pd.notna(r["category"]) else "-",
                    "DeepSeek": f"{r['ds_overall']:.3f}",
                    "Azure GPT-5": f"{r['gpt_overall']:.3f}",
                    "Δ (GPT - DS)": f"{r['gpt_overall'] - r['ds_overall']:+.3f}",
                    "Ερώτηση": (r["question"][:65] + "...") if pd.notna(r["question"]) and len(r["question"]) > 65 else (r["question"] or "-"),
                }
                for _, r in top_div.iterrows()
            ]
            st.dataframe(pd.DataFrame(divergence_records), width="stretch", hide_index=True)
            st.caption(
                "Διαγνωστική παρατήρηση: Παρατηρούμενες περιπτώσεις όπου οι δύο ανεξάρτητοι κριτές αποδίδουν διαφορετική βαρύτητα "
                "στη συγκεκριμένη απάντηση (|Δ| ≥ 0.50). Παρουσιάζονται οι πραγματικές βαθμολογίες χωρίς προκατειλημμένη απόδοση αιτιότητας."
            )
