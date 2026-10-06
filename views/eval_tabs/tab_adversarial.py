import json
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from db import get_adversarial_category_breakdown, get_adversarial_results_for_experiment
from views.eval_tabs.common import (
    EXP_DB_PATH,
    apply_clean_chart_theme,
    fmt_sec,
    fmt_tps,
    get_cached_adversarial_summary,
    render_card_header,
)


def _clean_adv_run_name(name: str) -> str:
    s = str(name or "")
    s = s.replace("ADV_", "")
    s = s.replace("_", " ")
    s = s.replace("Llama 3.1 8B CFG-4-LITE", "Llama 3.1 8B (CFG-4-Lite)")
    s = s.replace("CFG-1", "(CFG-1)")
    s = s.replace("CFG-3", "(CFG-3)")
    return s


@st.fragment
def render_tab_adversarial() -> None:
    
    render_card_header(
        title="Robustness",
        subtitle="RQ4 · Ανθεκτικότητα σε παραπλανητικά ερωτήματα και πρακτικοί περιορισμοί εκτέλεσης.",
        icon="security"
    )

    adv_intro_html = """<div style="margin: 6px 0 4px 0;">
<div style="font-weight: 700; font-size: 0.96rem; color: #f1f5f9; display: flex; align-items: center; gap: 8px;">
<span class="material-symbols-rounded" style="font-size: 20px; color: #38bdf8;">pest_control</span>
<span>Τύποι Παραπλανητικών Ερωτημάτων (Adversarial Traps)</span>
</div>
<div style="font-size: 0.84rem; color: #94a3b8; margin-top: 2px;">
Οι 5 ελεγχόμενες κατηγορίες παγίδων (20 ερωτήματα stress-test) που κατασκευάστηκαν για να ελέγξουν αν το RAG αρνείται να απαντήσει (Refusal) ή αν υποπίπτει σε παραισθήσεις (Hallucination):
</div>
</div>"""
    st.markdown(adv_intro_html, unsafe_allow_html=True)

    adv_tax_html = """<div class="adv-tax-grid-tiny" style="grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); margin: 8px 0 16px 0;">
<div class="adv-tax-card-tiny" style="border-left: 3.5px solid #38bdf8;">
<div class="adv-tax-title-tiny"><span class="material-symbols-rounded" style="font-size:16px;color:#38bdf8;">schedule</span> 1. Χρονικοί Αναχρονισμοί</div>
<div class="adv-tax-desc-tiny">Ερωτήματα για γεγονότα εκτός των ετών του corpus (π.χ. iPhone 18 το 2027, κανόνες SEC 2029).</div>
</div>
<div class="adv-tax-card-tiny" style="border-left: 3.5px solid #818cf8;">
<div class="adv-tax-title-tiny"><span class="material-symbols-rounded" style="font-size:16px;color:#818cf8;">domain</span> 2. Ενέσεις Ανταγωνιστών</div>
<div class="adv-tax-desc-tiny">Εισαγωγή στοιχείων άλλων εταιρειών ως δεδομένα Apple (π.χ. πωλήσεις Microsoft Surface, Android 10).</div>
</div>
<div class="adv-tax-card-tiny" style="border-left: 3.5px solid #f43f5e;">
<div class="adv-tax-title-tiny"><span class="material-symbols-rounded" style="font-size:16px;color:#f43f5e;">device_unknown</span> 3. Φανταστικά Προϊόντα</div>
<div class="adv-tax-desc-tiny">Αναφορές σε ανύπαρκτες συσκευές που δεν υπάρχουν στα 10-K (π.χ. Apple TV hardware, Apple Smart Ring).</div>
</div>
<div class="adv-tax-card-tiny" style="border-left: 3.5px solid #fbbf24;">
<div class="adv-tax-title-tiny"><span class="material-symbols-rounded" style="font-size:16px;color:#fbbf24;">manage_search</span> 4. Υπερβολική Εξειδίκευση</div>
<div class="adv-tax-desc-tiny">Απαιτήσεις μικρο-πληροφορίας που δεν καταγράφονται στα filings (π.χ. βίδες Apple Watch, κωδικός HEX χρώματος).</div>
</div>
<div class="adv-tax-card-tiny" style="border-left: 3.5px solid #ef4444;">
<div class="adv-tax-title-tiny"><span class="material-symbols-rounded" style="font-size:16px;color:#ef4444;">gpp_maybe</span> 5. Ψευδή Γεγονότα (Gaslighting)</div>
<div class="adv-tax-desc-tiny">Ερωτήματα βασισμένα σε κατασκευασμένες ψευδείς προκείμενες (π.χ. παραίτηση Tim Cook 2014, πτώχευση 2015).</div>
</div>
</div>"""
    st.markdown(adv_tax_html, unsafe_allow_html=True)

    df_adv_summary = get_cached_adversarial_summary(EXP_DB_PATH)
    cat_breakdown = get_adversarial_category_breakdown()

    if not df_adv_summary.empty:
        best_refusal = float(df_adv_summary["refusal_rate_pct"].max())
        best_runs = df_adv_summary[df_adv_summary["refusal_rate_pct"] == best_refusal]["experiment_name"].tolist()
        best_model_label = "Gemma 4 & DeepSeek-R1" if len(best_runs) > 1 else str(best_runs[0])

        category_greek_map = {
            "Temporal Anachronisms": "Χρονικοί Αναχρονισμοί",
            "Cross-Entity Contamination": "Ενέσεις Ανταγωνιστών",
            "Fictitious Product Lines": "Φανταστικά Προϊόντα",
            "Unreasonable Specificity": "Υπερβολική Εξειδίκευση",
            "False Premise (Gaslighting)": "Ψευδή Γεγονότα",
            "Temporal": "Χρονικοί Αναχρονισμοί",
            "Cross-Entity": "Ενέσεις Ανταγωνιστών",
            "Fictitious": "Φανταστικά Προϊόντα",
            "Specificity": "Υπερβολική Εξειδίκευση",
            "Gaslighting": "Ψευδή Γεγονότα"
        }

        if not cat_breakdown.empty:
            cat_means = cat_breakdown.groupby("category")["refusal_rate"].mean()
            hardest_cat_raw = str(cat_means.idxmin())
            hardest_cat = category_greek_map.get(hardest_cat_raw, hardest_cat_raw)
            hardest_rate = float(cat_means.min())
        else:
            hardest_cat = "Υπερβολική Εξειδίκευση"
            hardest_rate = 77.5

        cfg3_adv = df_adv_summary[df_adv_summary["config_id"] == "CFG-3"]["refusal_rate_pct"].mean()
        cfg1_adv = df_adv_summary[df_adv_summary["config_id"] == "CFG-1"]["refusal_rate_pct"].mean()
        rerank_defense_lift = float(cfg3_adv - cfg1_adv) if pd.notna(cfg3_adv) and pd.notna(cfg1_adv) else None
        lift_html = f"+{rerank_defense_lift:.1f} π.μ. <span class='eval-kpi-denom'>Διαφορά Άρνησης</span>" if rerank_defense_lift is not None else "N/A"

        total_runs_adv = len(df_adv_summary)
        total_stress_tests = int(df_adv_summary["total_questions"].sum()) if "total_questions" in df_adv_summary.columns else total_runs_adv * 20

        adv_kpis_html = f"""<div class="eval-kpi-grid">
<div class="eval-kpi-card">
<div class="eval-kpi-label">Κορυφαία Άμυνα Άρνησης</div>
<div class="eval-kpi-val">{best_refusal:.1f}% <span class="eval-kpi-denom">Refusal</span></div>
<div class="eval-kpi-model">Καλύτερη εκτέλεση: {best_model_label}</div>
<div class="eval-kpi-meta">20/20 ερωτήματα με άρνηση · 0% classified hallucination</div>
</div>
<div class="eval-kpi-card">
<div class="eval-kpi-label">Μεταβολή Άρνησης με Reranker</div>
<div class="eval-kpi-val">{lift_html}</div>
<div class="eval-kpi-model">CFG-3 έναντι CFG-1</div>
<div class="eval-kpi-meta">Μέση μεταβολή σε ποσοστιαίες μονάδες (RQ4)</div>
</div>
<div class="eval-kpi-card">
<div class="eval-kpi-label">Χαμηλότερη Άρνηση</div>
<div class="eval-kpi-val">{hardest_rate:.1f}% <span class="eval-kpi-denom">Refusal</span></div>
<div class="eval-kpi-model">{hardest_cat}</div>
<div class="eval-kpi-meta">Χαμηλότερο μετρημένο refusal rate μεταξύ των 5 κατηγοριών.</div>
</div>
<div class="eval-kpi-card">
<div class="eval-kpi-label">Adversarial Tests</div>
<div class="eval-kpi-val">{total_stress_tests} <span class="eval-kpi-denom">Δοκιμές</span></div>
<div class="eval-kpi-model">{total_runs_adv} runs × 20 traps</div>
<div class="eval-kpi-meta">5 κατηγορίες επιθέσεων (RQ4)</div>
</div>
</div>"""
        st.markdown(adv_kpis_html, unsafe_allow_html=True)
    else:
        st.info("Δεν υπάρχουν ακόμη καταγεγραμμένες δοκιμές ανθεκτικότητας. Μπορείτε να εκτελέσετε νέα Adversarial Tests από το research setup guide.", icon=":material/info:")

    if not df_adv_summary.empty:
        render_card_header(
            title="Διάσταση A: Adversarial Robustness",
            subtitle="Συγκεντρωτική συμπεριφορά και άμυνα σε παραπλανητικά ερωτήματα ανά κατηγορία επίθεσης και πειραματική εκτέλεση.",
            badge="Adversarial Defense",
            icon="security"
        )

        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            with st.container(border=True):
                render_card_header(
                    title="Refusal Rate by Attack Vector (%)",
                    subtitle="Σε ποιο είδος παγίδας παρατηρείται η μεγαλύτερη δυσκολία άρνησης.",
                    icon="analytics"
                )
                if not cat_breakdown.empty:
                    cat_plot = cat_breakdown.copy()
                    cat_plot["category_gr"] = cat_plot["category"].map(category_greek_map).fillna(cat_plot["category"])
                    cat_plot["clean_exp"] = cat_plot["experiment_name"].apply(_clean_adv_run_name)

                    fig_cat = px.bar(
                        cat_plot,
                        x="category_gr",
                        y="refusal_rate",
                        color="clean_exp",
                        barmode="group",
                        labels={"refusal_rate": "Ποσοστό Άρνησης (%)", "category_gr": "Διάνυσμα Επίθεσης", "clean_exp": "Εκτέλεση"},
                        color_discrete_sequence=["#38bdf8", "#34d399", "#818cf8", "#fbbf24", "#f43f5e"]
                    )
                    fig_cat.update_layout(
                        yaxis=dict(range=[0, 105]),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=9.5))
                    )
                    apply_clean_chart_theme(fig_cat, height=360)
                    st.plotly_chart(fig_cat, width="stretch", key="adv_cat_breakdown_bar")
                else:
                    st.caption("Δεν υπάρχουν διαθέσιμα δεδομένα ανά κατηγορία.")

        with col_chart2:
            with st.container(border=True):
                render_card_header(
                    title="Refusal Rate vs. Hallucination Rate (%)",
                    subtitle="Ποσοστά άρνησης και classified hallucination ανά run.",
                    icon="verified"
                )
                df_adv_plot = df_adv_summary.copy()
                df_adv_plot["display_name"] = df_adv_plot["experiment_name"].apply(_clean_adv_run_name)

                fig_adv_bar = go.Figure()
                fig_adv_bar.add_trace(go.Bar(
                    x=df_adv_plot["display_name"],
                    y=df_adv_plot["refusal_rate_pct"],
                    name="Άρνηση % (Υψηλότερο = Καλύτερο)",
                    marker_color="#10b981"
                ))
                fig_adv_bar.add_trace(go.Bar(
                    x=df_adv_plot["display_name"],
                    y=df_adv_plot["hallucination_rate_pct"],
                    name="Παραίσθηση % (Χαμηλότερο = Καλύτερο)",
                    marker_color="#ef4444"
                ))
                fig_adv_bar.update_layout(
                    barmode="group",
                    yaxis=dict(range=[0, 105], title="Ποσοστό (%)"),
                    xaxis_tickangle=-35,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                )
                apply_clean_chart_theme(fig_adv_bar, height=360)
                st.plotly_chart(fig_adv_bar, width="stretch", key="adv_refusal_hallucination_bar")

        with st.container(border=True):
            render_card_header(
                title="Ευπάθειες & Σφάλματα Αρχιτεκτονικής (Where Models Failed)",
                subtitle="Συγκριτική διάγνωση των κυριότερων αδυναμιών και αστοχιών άρνησης ανάμεσα σε RAG configs και παραμέτρους μοντέλων.",
                badge="Vulnerability Insights",
                icon="report_problem"
            )

            col_vuln1, col_vuln2, col_vuln3 = st.columns(3)
            with col_vuln1:
                st.markdown("""<div style="background:rgba(244,63,94,0.06); border:1px solid rgba(244,63,94,0.25); border-left:3.5px solid #f43f5e; border-radius:8px; padding:12px 14px; min-height:165px;">
<div style="font-weight:700; color:#f43f5e; font-size:0.90rem; margin-bottom:4px;">1. Συμπεριφορά CFG-4-LITE (65.0% Άρνηση · 35.0% Ασαφής · 0% Παραίσθηση)</div>
<div style="color:#e2e8f0; font-size:0.86rem; line-height:1.52;">
<b>Παρατηρούμενη Επίδοση:</b> Η CFG-4-LITE παρουσίασε ρητή άρνηση στο <b>65.0% (13/20 ερωτήματα)</b>. Οι υπόλοιπες <b>7/20 περιπτώσεις (35.0%) ταξινομήθηκαν ως UNCLEAR</b> (όπου το μοντέλο παρείχε επεξηγηματική αναδιατύπωση αντί τυπικής φόρμουλας άρνησης) και <b>0.0% παραισθήσεις</b> (καμία υιοθέτηση ψευδούς προκείμενης). Το αποτέλεσμα υποδηλώνει πιθανή ευαισθησία της lightweight HyDE διαδικασίας σε παραπλανητικές ερωτήσεις, χωρίς να τεκμηριώνεται αιτιακά από τα διαθέσιμα traces.
</div>
</div>""", unsafe_allow_html=True)

            with col_vuln2:
                st.markdown("""<div style="background:rgba(245,158,11,0.06); border:1px solid rgba(245,158,11,0.25); border-left:3.5px solid #f59e0b; border-radius:8px; padding:12px 14px; min-height:165px;">
<div style="font-weight:700; color:#f59e0b; font-size:0.90rem; margin-bottom:4px;">2. Μικρά Μοντέλα & Gaslighting (5.0% Hallucination)</div>
<div style="color:#e2e8f0; font-size:0.86rem; line-height:1.52;">
<b>Η Μοναδική Παραίσθηση:</b> Το Llama 3.2 3B στο Naive Dense (CFG-1) υπήρξε το <b>μοναδικό run</b> στο benchmark (1/180 δοκιμές) που παρήγαγε επιβεβαιωμένο hallucination (5.0%, 1/20), στην ερώτηση ψευδούς παραδοχής σχετικά με υποτιθέμενη χρεοκοπία Chapter 11 το 2015 (δοκιμή T17).
</div>
</div>""", unsafe_allow_html=True)

            with col_vuln3:
                st.markdown("""<div style="background:rgba(56,189,248,0.06); border:1px solid rgba(56,189,248,0.25); border-left:3.5px solid #38bdf8; border-radius:8px; padding:12px 14px; min-height:165px;">
<div style="font-weight:700; color:#38bdf8; font-size:0.90rem; margin-bottom:4px;">3. Μετάβαση CFG-1 → CFG-3 (+15 ποσοστιαίες μονάδες)</div>
<div style="color:#e2e8f0; font-size:0.86rem; line-height:1.52;">
<b>Σύγκριση Αρχιτεκτονικών:</b> Η μετάβαση από CFG-1 σε CFG-3 αύξησε το Refusal Rate του Llama 3.2 3B από <b>75.0% σε 90.0%</b> (+15 ποσοστιαίες μονάδες) και μείωσε το Hallucination Rate από 5.0% σε 0.0%. Το πείραμα συγκρίνει τις δύο συνολικές αρχιτεκτονικές και δεν απομονώνει αιτιακά τη συμβολή του Cross-Encoder από τις υπόλοιπες μεταβολές ανάκτησης.
</div>
</div>""", unsafe_allow_html=True)

        with st.container(border=True):
            render_card_header(
                title="Adversarial Benchmark Run Comparison",
                subtitle="Συγκριτικός πίνακας μετρικών άρνησης και παραίσθησης ανά πειραματική εκτέλεση.",
                icon="table_chart"
            )
            df_adv_table = df_adv_summary.copy()
            df_adv_table["clean_name"] = df_adv_table["experiment_name"].apply(_clean_adv_run_name)
            st.dataframe(
                df_adv_table[[
                    "clean_name", "model_name", "config_id", "total_questions",
                    "refused_count", "hallucinated_count", "refusal_rate_pct",
                    "hallucination_rate_pct", "avg_speed_tps", "avg_duration_sec"
                ]],
                column_config={
                    "clean_name": "Πείραμα",
                    "model_name": "Μοντέλο",
                    "config_id": "Διαμόρφωση",
                    "total_questions": "Παγίδες",
                    "refused_count": "Αρνήσεις",
                    "hallucinated_count": "Παραισθήσεις",
                    "refusal_rate_pct": st.column_config.ProgressColumn("Ποσοστό Άρνησης", format="%.1f%%", min_value=0, max_value=100),
                    "hallucination_rate_pct": st.column_config.ProgressColumn("Ποσοστό Παραίσθησης", format="%.1f%%", min_value=0, max_value=100),
                    "avg_speed_tps": st.column_config.NumberColumn("Ταχύτητα", format="%.2f tok/s"),
                    "avg_duration_sec": st.column_config.NumberColumn("Μέση Καθυστέρηση", format="%.2f s")
                },
                width="stretch",
                hide_index=True
            )

        with st.container(border=True):
            render_card_header(
                title="Διάσταση B: Hardware Feasibility (RQ4)",
                subtitle="Εμπειρικά όρια εκτέλεσης στο δοκιμασμένο Apple Silicon 16GB setup: συσχέτιση μεγέθους μοντέλου και throughput.",
                badge="Hardware Limits",
                icon="memory"
            )

            col_hw1, col_hw2 = st.columns([1.15, 0.85])
            with col_hw1:
                hw_feasibility_data = [
                    {"Model": "Llama 3.2 3B (3.2B)", "Throughput": 11.31, "Status": "Βιώσιμο (3.2B)"},
                    {"Model": "Llama 3.1 8B (8.0B)", "Throughput": 4.78, "Status": "Βιώσιμο (8.0B)"},
                    {"Model": "Gemma 4 8B (8.2B)", "Throughput": 2.25, "Status": "Βιώσιμο (8.2B)"},
                    {"Model": "DeepSeek-R1 8B (8.2B)", "Throughput": 1.49, "Status": "Βιώσιμο (8.2B)"},
                    {"Model": "Qwen 3.5 9B (9.7B)", "Throughput": 0.07, "Status": "Stress-test boundary"}
                ]
                df_hw_plot = pd.DataFrame(hw_feasibility_data)

                fig_hw_lim = px.bar(
                    df_hw_plot,
                    x="Throughput",
                    y="Model",
                    orientation="h",
                    color="Status",
                    text=[f"{v:.2f} tok/s" for v in df_hw_plot["Throughput"]],
                    labels={"Throughput": "Ταχύτητα Παραγωγής (tok/s)", "Model": "Μέγεθος Μοντέλου"},
                    color_discrete_map={
                        "Βιώσιμο (3.2B)": "#38bdf8",
                        "Βιώσιμο (8.0B)": "#818cf8",
                        "Βιώσιμο (8.2B)": "#34d399",
                        "Stress-test boundary": "#ef4444"
                    }
                )
                fig_hw_lim.add_annotation(
                    x=0.07,
                    y="Qwen 3.5 9B (9.7B)",
                    text="<b>Stress-test boundary observed</b>",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1.5,
                    arrowcolor="#ef4444",
                    ax=130,
                    ay=0,
                    font=dict(size=10, color="#fca5a5"),
                    bgcolor="rgba(239, 68, 68, 0.15)",
                    bordercolor="#ef4444",
                    borderwidth=1,
                    borderpad=3
                )
                fig_hw_lim.update_layout(
                    xaxis=dict(title="Ταχύτητα Παραγωγής (tok/s)"),
                    yaxis=dict(autorange="reversed"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=9.5)),
                    margin=dict(l=45, r=20, t=40, b=40)
                )
                apply_clean_chart_theme(fig_hw_lim, height=335)
                st.plotly_chart(fig_hw_lim, width="stretch", key="adv_hardware_feasibility_bar")

            with col_hw2:
                hw_findings_html = """<div style="padding-top: 2px;">
<div style="display:flex; align-items:center; gap:8px; font-weight:700; color:#ef4444; font-size:0.98rem; margin-bottom:12px;">
<span class="material-symbols-rounded" style="font-size:22px; color:#ef4444;">memory</span>
<span>Παρατηρούμενα Όρια στο Δοκιμασμένο Setup (16GB RAM)</span>
</div>

<div style="background:rgba(239,68,68,0.06); border:1px solid rgba(239,68,68,0.25); border-left:3.5px solid #ef4444; border-radius:8px; padding:10px 14px; margin-bottom:10px;">
<div style="font-weight:700; color:#ef4444; font-size:0.90rem; margin-bottom:3px;">Πίεση Μνήμης &amp; Swap</div>
<div style="color:#e2e8f0; font-size:0.86rem; line-height:1.52;">Στο συγκεκριμένο stress-test του Qwen 3.5 9B, η συνολική μνήμη έφτασε περίπου στα <b>15.6 GB</b> και η έντονη πίεση οδήγησε σε συνεχή SSD swap, με αποτέλεσμα δραματική πτώση του throughput από περίπου 11 tok/s σε 0.07 tok/s (<b>−99.4% στο συγκεκριμένο comparison</b>).</div>
</div>

<div style="background:rgba(245,158,11,0.06); border:1px solid rgba(245,158,11,0.25); border-left:3.5px solid #f59e0b; border-radius:8px; padding:10px 14px; margin-bottom:10px;">
<div style="font-weight:700; color:#f59e0b; font-size:0.90rem; margin-bottom:3px;">Παρατηρούμενο Εμπειρικό Όριο</div>
<div style="color:#e2e8f0; font-size:0.86rem; line-height:1.52;">Στο δοκιμασμένο setup των 16GB, τα μοντέλα έως περίπου <b>8.2B</b> παρέμειναν εντός της παρατηρούμενης περιοχής πρακτικής εκτέλεσης, ενώ το Qwen 3.5 9B αποτέλεσε stress-test failure.</div>
</div>

<div style="background:rgba(56,189,248,0.06); border:1px solid rgba(56,189,248,0.25); border-left:3.5px solid #38bdf8; border-radius:8px; padding:10px 14px;">
<div style="font-weight:700; color:#38bdf8; font-size:0.90rem; margin-bottom:3px;">Ανεξάρτητη Διάσταση RQ4</div>
<div style="color:#e2e8f0; font-size:0.86rem; line-height:1.52;">Η ανθεκτικότητα σε παραπλανητικές εισόδους (Adversarial Robustness) και η εφικτότητα υλικού (Hardware Feasibility) εξετάζονται ως δύο ξεχωριστές διαστάσεις αξιοπιστίας, χωρίς αιτιώδη συσχέτιση μεταξύ τους.</div>
</div>
</div>"""
                st.markdown(hw_findings_html, unsafe_allow_html=True)

        with st.expander("Trap Inspector", expanded=False, icon=":material/manage_search:"):
            st.caption("Αναλυτική εξέταση απαντήσεων, ανακτηθέντος θορύβου και αποτελεσμάτων ανά ερώτηση-παγίδα.")
            col_insp_exp, col_insp_cat = st.columns(2)
            with col_insp_exp:
                adv_exp_lookup = {r["id"]: str(r["experiment_name"]) for _, r in df_adv_summary.iterrows()}
                exp_options = [-1] + df_adv_summary["id"].tolist()
                selected_adv_exp = st.selectbox(
                    "Επιλογή Adversarial Εκτέλεσης",
                    exp_options,
                    format_func=lambda x: "★ Όλες οι Αστοχίες του Benchmark (20 περιπτώσεις)" if x == -1 else f"#{x} - {_clean_adv_run_name(adv_exp_lookup.get(x, f'Run {x}'))}"
                )
            with col_insp_cat:
                if selected_adv_exp == -1:
                    with sqlite3.connect(EXP_DB_PATH) as conn:
                        adv_detail_df = pd.read_sql_query("""
                            SELECT r.*, e.experiment_name, e.model_name, e.config_id
                            FROM adversarial_results r
                            JOIN adversarial_experiments e ON r.exp_id = e.id
                            WHERE r.is_refusal = 0 OR r.is_hallucination = 1
                            ORDER BY e.id, r.question_idx;
                        """, conn)
                    cats_avail = ["All"] + list(adv_detail_df["category"].dropna().unique()) if not adv_detail_df.empty else ["All"]
                    selected_adv_cat = st.selectbox("Φιλτράρισμα ανά Κατηγορία", cats_avail, key="adv_cat_filter", format_func=lambda x: category_greek_map.get(x, x))
                else:
                    adv_detail_df = get_adversarial_results_for_experiment(selected_adv_exp)
                    cats_avail = ["All"] + list(adv_detail_df["category"].dropna().unique()) if not adv_detail_df.empty else ["All"]
                    selected_adv_cat = st.selectbox("Φιλτράρισμα ανά Κατηγορία", cats_avail, key="adv_cat_filter", format_func=lambda x: category_greek_map.get(x, x))

            if selected_adv_exp != -1:
                show_failures_only = st.checkbox("Προβολή μόνο των αστοχιών (Μη-Άρνηση / Παραίσθηση) για τη συγκεκριμένη εκτέλεση", value=False)
                if show_failures_only and not adv_detail_df.empty:
                    adv_detail_df = adv_detail_df[(adv_detail_df["is_refusal"] == 0) | (adv_detail_df["is_hallucination"] == 1)]

            if not adv_detail_df.empty:
                if selected_adv_cat != "All":
                    adv_detail_df = adv_detail_df[adv_detail_df["category"] == selected_adv_cat]

                st.caption(f"Εμφάνιση **{len(adv_detail_df)}** ερωτήσεων-παγίδων:")
                with st.container(height=520):
                    for _, r_adv in adv_detail_df.iterrows():
                        is_ref = bool(r_adv.get("is_refusal", False))
                        is_hall = bool(r_adv.get("is_hallucination", False))
                        icon_e = ":material/shield:" if is_ref else (":material/warning:" if is_hall else ":material/help:")
                        status_str = "Ορθή Άρνηση (Refusal)" if is_ref else ("Παραίσθηση (Hallucination)" if is_hall else "Ασαφής / Αμφίσημη")

                        exp_prefix = f"[{_clean_adv_run_name(r_adv.get('experiment_name', ''))}] " if selected_adv_exp == -1 else ""
                        cat_label = category_greek_map.get(r_adv["category"], r_adv["category"])
                        expander_title = f"{exp_prefix}Παγίδα #{r_adv['question_idx']+1} · [{cat_label}] {r_adv['question'][:75]}..."

                        with st.expander(expander_title, expanded=False, icon=icon_e):
                            col_t1, col_t2 = st.columns([1.2, 0.8])
                            with col_t1:
                                st.markdown(f"**Trap Question:**\n{r_adv['question']}")
                                st.markdown(f"**Ground Truth Reference:**\n> {r_adv['reference']}")
                                st.markdown(f"**Model Answer:**\n{r_adv['generated_answer']}")

                            with col_t2:
                                with st.container(border=True):
                                    st.markdown("##### Ετυμηγορία")
                                    if is_ref:
                                        st.success(f"**{status_str}**", icon=":material/check_circle:")
                                    elif is_hall:
                                        st.error(f"**{status_str}**", icon=":material/cancel:")
                                    else:
                                        st.warning(f"**{status_str}**", icon=":material/help:")

                                    st.metric("Βεβαιότητα Άρνησης", f"{r_adv.get('refusal_confidence', 0.0):.2f}", border=True)
                                    st.caption(f"Ταχύτητα: `{fmt_tps(r_adv.get('speed_tps'))}` · Χρόνος: `{fmt_sec(r_adv.get('total_duration_sec'))}`")

                            try:
                                raw_adv_ctxs = r_adv.get("retrieved_contexts")
                                if raw_adv_ctxs:
                                    if isinstance(raw_adv_ctxs, str):
                                        try:
                                            adv_ctxs = json.loads(raw_adv_ctxs)
                                        except Exception:
                                            adv_ctxs = [raw_adv_ctxs]
                                    elif isinstance(raw_adv_ctxs, list):
                                        adv_ctxs = raw_adv_ctxs
                                    else:
                                        adv_ctxs = [str(raw_adv_ctxs)]

                                    if adv_ctxs and isinstance(adv_ctxs, list):
                                        with st.expander(f"Ανακτηθέντα Chunks Θορύβου από 10-K ({len(adv_ctxs)} chunks)", expanded=False, icon=":material/description:"):
                                            for idx_ac, actx in enumerate(adv_ctxs):
                                                st.markdown(f"**Noise Chunk #{idx_ac+1}:**")
                                                st.text(actx)
                                    elif adv_ctxs:
                                        with st.expander("Ανακτηθέντα Chunks Θορύβου από 10-K", expanded=False, icon=":material/description:"):
                                            st.text(str(adv_ctxs[0]))
                            except Exception:
                                pass
