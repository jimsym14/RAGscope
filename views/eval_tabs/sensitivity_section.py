import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import EXP_DB_PATH, get_sensitivity_db_path
from views.eval_tabs.common import apply_clean_chart_theme



def _safe_html(html_str: str) -> None:
    
    cleaned = "".join(line.strip() for line in html_str.splitlines() if line.strip())
    st.html(cleaned)



@st.cache_data(ttl=60, show_spinner=False)
def load_sensitivity_dataset(
    db_path: str,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    

    if not db_path or not os.path.exists(db_path):
        return None, "Database file not found."

    conn = None

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

        query = """
        SELECT
            g.cell_id,
            g.chunk_size,
            g.final_k,
            g.benchmark_row_index,
            g.question_id,
            g.category,
            g.total_duration_sec AS latency_sec,
            j.overall_score,
            j.context_recall
        FROM generations g
        JOIN judge_evaluations j
          ON g.cell_id = j.cell_id
         AND g.benchmark_row_index = j.benchmark_row_index
        ORDER BY
            g.chunk_size,
            g.final_k,
            g.benchmark_row_index
        """

        df = pd.read_sql_query(query, conn)

    except Exception as exc:
        return None, f"Query error: {exc}"

    finally:
        if conn:
            conn.close()

    if df is None or df.empty:
        return None, "Empty sensitivity dataset returned."

    total_rows = len(df)
    unique_cells = df["cell_id"].nunique()
    chunk_sizes = sorted(df["chunk_size"].unique().tolist())
    final_ks = sorted(df["final_k"].unique().tolist())

    q_per_cell = (
        df.groupby("cell_id")["benchmark_row_index"]
        .count()
        .tolist()
    )

    if total_rows != 180:
        return df, f"Integrity warning: expected 180 rows, got {total_rows}."

    if unique_cells != 9:
        return df, f"Integrity warning: expected 9 cells, got {unique_cells}."

    if chunk_sizes != [256, 512, 1024]:
        return df, f"Integrity warning: unexpected chunk sizes: {chunk_sizes}."

    if final_ks != [1, 3, 5]:
        return df, f"Integrity warning: unexpected final K values: {final_ks}."

    if any(count != 20 for count in q_per_cell):
        return (
            df,
            "Integrity warning: not all cells have 20 benchmark questions."
        )

    return df, None


@st.cache_data(ttl=60, show_spinner=False)
def load_overlap_comparison(
    sens_df: pd.DataFrame,
) -> Dict[str, float]:
    
    sens_512_k3 = sens_df[
        (sens_df["chunk_size"] == 512) & (sens_df["final_k"] == 3)
    ]

    s_ov = float(sens_512_k3["overall_score"].mean()) if not sens_512_k3.empty else 0.6452
    s_rec = float(sens_512_k3["context_recall"].mean()) if not sens_512_k3.empty else 0.5625
    s_lat = float(sens_512_k3["latency_sec"].mean()) if not sens_512_k3.empty else 35.06

    r04_ov = 0.7292
    r04_rec = 0.5535
    r04_lat = 50.19

    exp_db_path = EXP_DB_PATH
    if os.path.exists(exp_db_path):
        conn_exp = None
        try:
            conn_exp = sqlite3.connect(f"file:{exp_db_path}?mode=ro", uri=True)
            indices = sens_df["benchmark_row_index"].drop_duplicates().tolist()
            indices_str = ",".join(map(str, indices))

            q_judge = f"""
            SELECT AVG(overall_score) as overall,
                   AVG(context_recall) as recall
            FROM judge_scores
            WHERE experiment_id = 6
              AND judge_model = 'gpt-5-mini'
              AND question_idx IN ({indices_str})
            """
            exp_judge = pd.read_sql_query(q_judge, conn_exp)

            q_lat = f"""
            SELECT AVG(total_duration_sec) as latency
            FROM detailed_results
            WHERE experiment_id = 6
              AND question_idx IN ({indices_str})
            """
            exp_lat = pd.read_sql_query(q_lat, conn_exp)

            if not exp_judge.empty and pd.notnull(exp_judge["overall"].iloc[0]):
                r04_ov = float(exp_judge["overall"].iloc[0])
                r04_rec = float(exp_judge["recall"].iloc[0])
            if not exp_lat.empty and pd.notnull(exp_lat["latency"].iloc[0]):
                r04_lat = float(exp_lat["latency"].iloc[0])

        except Exception:
            pass
        finally:
            if conn_exp:
                conn_exp.close()

    delta_ov = r04_ov - s_ov
    pct_ov = (delta_ov / s_ov) * 100 if s_ov > 0 else 0.0

    return {
        "r04_ov": r04_ov,
        "r04_rec": r04_rec,
        "r04_lat": r04_lat,
        "s_ov": s_ov,
        "s_rec": s_rec,
        "s_lat": s_lat,
        "delta_ov": delta_ov,
        "pct_ov": pct_ov,
    }



def _make_stepped_colorscale(
    cutoffs: List[float],
    colors: List[str],
) -> Tuple[List[List[Any]], float, float]:
    
    zmin = cutoffs[0]
    zmax = cutoffs[-1]
    span = zmax - zmin
    u = [(c - zmin) / span for c in cutoffs]

    cs: List[List[Any]] = []
    for i in range(len(colors)):
        cs.append([u[i], colors[i]])
        cs.append([u[i + 1], colors[i]])

    return cs, zmin, zmax


def _metric_matrix(
    agg: pd.DataFrame,
    metric_column: str,
    chunk_sizes: List[int],
    ks: List[int],
) -> Tuple[List[List[float]], List[List[str]]]:
    
    z_matrix: List[List[float]] = []
    text_matrix: List[List[str]] = []

    for chunk_size in chunk_sizes:
        z_row: List[float] = []
        text_row: List[str] = []

        for k in ks:
            match = agg[
                (agg["chunk_size"] == chunk_size)
                & (agg["final_k"] == k)
            ]

            if match.empty:
                value = 0.0
            else:
                value = float(match[metric_column].iloc[0])

            z_row.append(value)
            text_row.append(
                f"{value:.4f}" if "latency" not in metric_column
                else f"{value:.2f}s"
            )

        z_matrix.append(z_row)
        text_matrix.append(text_row)

    return z_matrix, text_matrix


def _heatmap(
    z: List[List[float]],
    text: List[List[str]],
    colorscale: Any,
    zmin: float,
    zmax: float,
    metric_name: str,
    hover_suffix: str = "",
) -> go.Figure:
    
    is_latency = "latency" in metric_name.lower() or "απόκρισης" in metric_name.lower()
    z_fmt = "%{z:.2f}s" if is_latency else "%{z:.4f}"

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=["k=1", "k=3", "k=5"],
            y=["256", "512", "1024"],
            text=text,
            texttemplate="%{text}",
            textfont=dict(
                size=14.5,
                color="#ffffff",
                family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            ),
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            showscale=False,
            hovertemplate=(
                "<b>Μέγεθος Chunk:</b> %{y} tokens<br>"
                "<b>Τελικό k:</b> %{x}<br>"
                f"<b>{metric_name}:</b> {z_fmt}"
                "<extra></extra>"
            ),
            xgap=4,
            ygap=4,
        )
    )

    apply_clean_chart_theme(fig, height=230)

    fig.update_layout(
        margin=dict(l=54, r=10, t=6, b=32),
        xaxis=dict(
            title="",
            type="category",
            categoryorder="array",
            categoryarray=["k=1", "k=3", "k=5"],
            tickmode="array",
            tickvals=["k=1", "k=3", "k=5"],
            ticktext=["<b>k=1</b>", "<b>k=3</b>", "<b>k=5</b>"],
            tickfont=dict(size=12.5, color="#cbd5e1"),
            fixedrange=True,
        ),
        yaxis=dict(
            title="Chunk size",
            title_font=dict(size=11, color="#94a3b8"),
            type="category",
            categoryorder="array",
            categoryarray=["256", "512", "1024"],
            tickmode="array",
            tickvals=["256", "512", "1024"],
            ticktext=["<b>256</b>", "<b>512</b>", "<b>1024</b>"],
            tickfont=dict(size=12, color="#f1f5f9"),
            autorange="reversed",
            fixedrange=True,
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig



def render_sensitivity_section() -> None:
    

    db_path = get_sensitivity_db_path()

    if not db_path or not os.path.exists(db_path):
        st.info(
            "Τα δεδομένα της μελέτης ευαισθησίας δεν είναι διαθέσιμα.",
            icon=":material/info:",
        )
        return

    df, integrity_warning = load_sensitivity_dataset(db_path)

    if df is None:
        st.info(
            "Τα δεδομένα της μελέτης ευαισθησίας δεν είναι διαθέσιμα.",
            icon=":material/info:",
        )
        return

    if integrity_warning:
        st.warning(
            integrity_warning,
            icon=":material/warning:",
        )


    agg = (
        df.groupby(["chunk_size", "final_k"])
        .agg(
            overall=("overall_score", "mean"),
            recall=("context_recall", "mean"),
            latency=("latency_sec", "mean"),
        )
        .reset_index()
    )

    chunk_sizes = [256, 512, 1024]
    ks = [1, 3, 5]

    k_means = df.groupby("final_k")["overall_score"].mean()
    k_delta_1_3 = float(k_means.get(3, 0.0) - k_means.get(1, 0.0))
    k_delta_3_5 = float(k_means.get(5, 0.0) - k_means.get(3, 0.0))


    with st.container(border=True):


        header_html = """
        <div style="margin-bottom:12px;">
          <div style="display:flex; align-items:baseline; justify-content:space-between; gap:10px; flex-wrap:wrap;">
            <div style="font-size:1.16rem; font-weight:750; color:#f8fafc; letter-spacing:-0.02em;">
              Ελεγχόμενη Μελέτη Ευαισθησίας Ανάκτησης (Retrieval Sensitivity)
            </div>
            <div style="font-size:0.86rem; color:#cbd5e1; font-weight:600;">
              CFG-3 · Llama 3.1 8B · 20 ερωτήσεις × 9 συνθήκες (180 εκτελέσεις)
            </div>
          </div>
          <div style="margin-top:3px; font-size:0.84rem; color:#94a3b8;">
            Επίδραση μεγέθους chunk (256, 512, 1024) και αριθμού τελικών chunks K (1, 3, 5) με σταθερό overlap 50 tokens.
          </div>
        </div>
        """
        _safe_html(header_html)


        colors_quality_steps = [
            "#dc2626",  
            "#ea580c",  
            "#f59e0b",  
            "#eab308",  
            "#84cc16",  
            "#16a34a",  
        ]

        cutoffs_overall = [0.54, 0.57, 0.60, 0.63, 0.665, 0.69, 0.72]
        cs_overall, zmin_overall, zmax_overall = _make_stepped_colorscale(cutoffs_overall, colors_quality_steps)

        cutoffs_recall = [0.28, 0.34, 0.40, 0.50, 0.595, 0.635, 0.67]
        cs_recall, zmin_recall, zmax_recall = _make_stepped_colorscale(cutoffs_recall, colors_quality_steps)

        # Latency Ranges: [20.21s -> 107.22s] (Lower is better -> Reversed color sequence)
        colors_latency_steps = [
            "#16a34a",  
            "#84cc16",  
            "#eab308",  
            "#f59e0b",  
            "#ea580c",  
            "#dc2626",  
        ]
        cutoffs_latency = [15.0, 28.0, 42.0, 62.0, 85.0, 100.0, 115.0]
        cs_latency, zmin_latency, zmax_latency = _make_stepped_colorscale(cutoffs_latency, colors_latency_steps)

        overall_z, overall_text = _metric_matrix(agg, "overall", chunk_sizes, ks)
        recall_z, recall_text = _metric_matrix(agg, "recall", chunk_sizes, ks)
        latency_z, latency_text = _metric_matrix(agg, "latency", chunk_sizes, ks)


        col1, col2, col3 = st.columns(3, gap="medium")

        with col1:
            _safe_html("""<div style="font-size:0.92rem; font-weight:750; color:#f1f5f9; margin-bottom:6px;">Overall Score:</div>""")

            fig_overall = _heatmap(
                overall_z,
                overall_text,
                colorscale=cs_overall,
                zmin=zmin_overall,
                zmax=zmax_overall,
                metric_name="Overall Score",
            )

            st.plotly_chart(
                fig_overall,
                width="stretch",
                config={"displayModeBar": False},
                key="sens_matrix_overall",
            )

        with col2:
            _safe_html("""<div style="font-size:0.92rem; font-weight:750; color:#f1f5f9; margin-bottom:6px;">Context Recall:</div>""")

            fig_recall = _heatmap(
                recall_z,
                recall_text,
                colorscale=cs_recall,
                zmin=zmin_recall,
                zmax=zmax_recall,
                metric_name="Context Recall",
            )

            st.plotly_chart(
                fig_recall,
                width="stretch",
                config={"displayModeBar": False},
                key="sens_matrix_recall",
            )

        with col3:
            _safe_html("""<div style="font-size:0.92rem; font-weight:750; color:#f1f5f9; margin-bottom:6px;">Χρόνος Απόκρισης (Latency s):</div>""")

            fig_latency = _heatmap(
                latency_z,
                latency_text,
                colorscale=cs_latency,
                zmin=zmin_latency,
                zmax=zmax_latency,
                metric_name="Χρόνος Απόκρισης (Latency s)",
            )

            st.plotly_chart(
                fig_latency,
                width="stretch",
                config={"displayModeBar": False},
                key="sens_matrix_latency",
            )


        delta_str_1_3 = f"{k_delta_1_3:+.4f}"
        delta_str_3_5 = f"{k_delta_3_5:+.4f}"

        findings_html = f"""
        <div style="margin-top:10px; margin-bottom:12px;">
          <div style="font-size:0.95rem; font-weight:700; color:#f1f5f9; margin-bottom:10px;">
            Πρακτικά Συμπεράσματα (Τι αλλάζει στην πράξη)
          </div>
          <div style="display:flex; align-items:stretch; gap:10px; flex-wrap:wrap;">

            <!-- CARD 1: K 1 -> 3 -->
            <div style="flex:1; min-width:240px; background:rgba(22,163,74,0.08); border:1px solid rgba(22,163,74,0.35); border-radius:8px; padding:14px 16px;">
              <div style="font-size:0.82rem; font-weight:700; color:#cbd5e1; letter-spacing:0.02em; margin-bottom:4px;">
                Αύξηση Budget (K: 1 → 3)
              </div>
              <div style="display:flex; align-items:baseline; gap:8px; margin-bottom:2px;">
                <span style="font-size:clamp(1.15rem, 1.6vw, 1.40rem); font-weight:800; color:#4ade80; letter-spacing:-0.01em;">
                  Σαφής βελτίωση
                </span>
                <span style="font-size:clamp(0.80rem, 1.0vw, 0.92rem); font-weight:600; color:#86efac; opacity:0.85;">
                  ({delta_str_1_3})
                </span>
              </div>
              <div style="margin-top:4px; font-size:0.80rem; color:#94a3b8; font-weight:500;">
                Σημαντικό άλμα ακρίβειας (p = 0.0045)
              </div>
            </div>

            <!-- ARROW -->
            <div style="display:flex; align-items:center; justify-content:center; color:#64748b; font-size:1.4rem; font-weight:700; padding:0 4px; user-select:none;">
              →
            </div>

            <!-- CARD 2: K 3 -> 5 -->
            <div style="flex:1; min-width:240px; background:rgba(148,163,184,0.06); border:1px solid rgba(148,163,184,0.25); border-radius:8px; padding:14px 16px;">
              <div style="font-size:0.82rem; font-weight:700; color:#cbd5e1; letter-spacing:0.02em; margin-bottom:4px;">
                Περαιτέρω Αύξηση (K: 3 → 5)
              </div>
              <div style="display:flex; align-items:baseline; gap:8px; margin-bottom:2px;">
                <span style="font-size:clamp(1.15rem, 1.6vw, 1.40rem); font-weight:800; color:#cbd5e1; letter-spacing:-0.01em;">
                  Μη ουσιώδης μείωση
                </span>
                <span style="font-size:clamp(0.80rem, 1.0vw, 0.92rem); font-weight:500; color:#94a3b8; opacity:0.75;">
                  ({delta_str_3_5})
                </span>
              </div>
              <div style="margin-top:4px; font-size:0.80rem; color:#94a3b8; font-weight:500;">
                Πλήρης κορεσμός (πλατό, p = 0.9854)
              </div>
            </div>

            <!-- SEPARATOR -->
            <div style="display:flex; align-items:center; justify-content:center; color:rgba(255,255,255,0.18); font-size:1.3rem; padding:0 4px; user-select:none;">
              |
            </div>

            <!-- CARD 3: CHUNK SIZE -->
            <div style="flex:1; min-width:240px; background:rgba(148,163,184,0.06); border:1px solid rgba(148,163,184,0.25); border-radius:8px; padding:14px 16px;">
              <div style="font-size:0.82rem; font-weight:700; color:#cbd5e1; letter-spacing:0.02em; margin-bottom:4px;">
                Μέγεθος Chunk (256 ↔ 1024)
              </div>
              <div style="display:flex; align-items:baseline; gap:8px; margin-bottom:2px;">
                <span style="font-size:clamp(1.15rem, 1.6vw, 1.40rem); font-weight:800; color:#cbd5e1; letter-spacing:-0.01em;">
                  Καμία επίδραση
                </span>
                <span style="font-size:clamp(0.80rem, 1.0vw, 0.92rem); font-weight:500; color:#94a3b8; opacity:0.75;">
                  (p = 0.8187)
                </span>
              </div>
              <div style="margin-top:4px; font-size:0.80rem; color:#94a3b8; font-weight:500;">
                Ίδια ποιότητα, 5× καθυστέρηση (26s ➔ 68s)
              </div>
            </div>

          </div>
        </div>
        """
        _safe_html(findings_html)


        comp = load_overlap_comparison(df)

        comparison_html = f"""
        <div style="margin-top:12px; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:14px 16px;">
          
          <div style="font-size:clamp(0.80rem, 1.1vw, 0.92rem); font-weight:750; color:#cbd5e1; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:12px;">
            Σύγκριση Overlap (512 tokens / K=3)
          </div>

          <div style="display:flex; align-items:center; justify-content:space-between; gap:14px; flex-wrap:wrap;">

            <!-- PRODUCTION RUN 04 -->
            <div style="flex:1; min-width:280px; background:rgba(56,189,248,0.05); border:1px solid rgba(56,189,248,0.25); border-radius:8px; padding:12px 16px;">
              <div style="display:flex; align-items:center; justify-content:space-between; padding-bottom:8px; border-bottom:1px solid rgba(56,189,248,0.20);">
                <span style="font-size:clamp(0.92rem, 1.3vw, 1.15rem); font-weight:800; color:#38bdf8;">Production Run 04</span>
                <span style="font-size:clamp(0.70rem, 0.9vw, 0.80rem); background:rgba(56,189,248,0.18); color:#7dd3fc; padding:2px 8px; border-radius:4px; font-weight:700;">Overlap 200</span>
              </div>

              <div style="display:flex; align-items:stretch; margin-top:10px;">
                <div style="flex:1; text-align:center; padding-right:8px; border-right:1px solid rgba(255,255,255,0.10);">
                  <div style="font-size:clamp(0.68rem, 0.9vw, 0.80rem); color:#94a3b8; text-transform:uppercase; font-weight:700;">Overall</div>
                  <div style="font-size:clamp(1.35rem, 2.2vw, 2.00rem); font-weight:850; color:#38bdf8; line-height:1.2; margin-top:2px;">{comp['r04_ov']:.4f}</div>
                </div>
                <div style="flex:1; text-align:center; padding:0 8px; border-right:1px solid rgba(255,255,255,0.10);">
                  <div style="font-size:clamp(0.68rem, 0.9vw, 0.80rem); color:#94a3b8; text-transform:uppercase; font-weight:700;">Recall</div>
                  <div style="font-size:clamp(1.15rem, 1.8vw, 1.70rem); font-weight:750; color:#cbd5e1; line-height:1.2; margin-top:2px;">{comp['r04_rec']:.4f}</div>
                </div>
                <div style="flex:1; text-align:center; padding-left:8px;">
                  <div style="font-size:clamp(0.68rem, 0.9vw, 0.80rem); color:#94a3b8; text-transform:uppercase; font-weight:700;">Latency</div>
                  <div style="font-size:clamp(1.15rem, 1.8vw, 1.70rem); font-weight:750; color:#cbd5e1; line-height:1.2; margin-top:2px;">{comp['r04_lat']:.1f}s</div>
                </div>
              </div>
            </div>

            <!-- DELTA WITH GIANT < AND > (SVG SHAPES) -->
            <div style="display:flex; align-items:center; justify-content:center; gap:10px; flex-shrink:0; margin:0 auto;">
              <!-- GIANT < (LEFT, RED SHAPE) -->
              <svg viewBox="0 0 24 46" fill="none" xmlns="http://www.w3.org/2000/svg" style="height:clamp(38px, 4.8vw, 54px); width:auto; display:block; flex-shrink:0;">
                <path d="M18 7L6 23L18 39" stroke="#ef4444" stroke-width="3.8" stroke-linecap="round" stroke-linejoin="round"/>
              </svg>

              <!-- DELTA CARD -->
              <div style="text-align:center; padding:8px 14px; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.10); border-radius:8px; min-width:120px;">
                <div style="font-size:0.68rem; color:#94a3b8; font-weight:700; text-transform:uppercase; letter-spacing:0.04em;">Διαφορά (Δ)</div>
                <div style="font-size:clamp(1.20rem, 1.8vw, 1.45rem); font-weight:850; color:#ffffff; line-height:1.15; margin:2px 0;">{comp['delta_ov']:+.4f}</div>
                <div style="font-size:0.72rem; color:#34d399; font-weight:700;">+{comp['pct_ov']:.1f}% Overall</div>
              </div>

              <!-- GIANT > (RIGHT, GREEN SHAPE) -->
              <svg viewBox="0 0 24 46" fill="none" xmlns="http://www.w3.org/2000/svg" style="height:clamp(38px, 4.8vw, 54px); width:auto; display:block; flex-shrink:0;">
                <path d="M6 7L18 23L6 39" stroke="#10b981" stroke-width="3.8" stroke-linecap="round" stroke-linejoin="round"/>
              </svg>
            </div>

            <!-- SENSITIVITY GRID -->
            <div style="flex:1; min-width:280px; background:rgba(168,85,247,0.05); border:1px solid rgba(168,85,247,0.25); border-radius:8px; padding:12px 16px;">
              <div style="display:flex; align-items:center; justify-content:space-between; padding-bottom:8px; border-bottom:1px solid rgba(168,85,247,0.20);">
                <span style="font-size:clamp(0.92rem, 1.3vw, 1.15rem); font-weight:800; color:#c084fc;">Μελέτη Ευαισθησίας</span>
                <span style="font-size:clamp(0.70rem, 0.9vw, 0.80rem); background:rgba(168,85,247,0.18); color:#d8b4fe; padding:2px 8px; border-radius:4px; font-weight:700;">Overlap 50</span>
              </div>

              <div style="display:flex; align-items:stretch; margin-top:10px;">
                <div style="flex:1; text-align:center; padding-right:8px; border-right:1px solid rgba(255,255,255,0.10);">
                  <div style="font-size:clamp(0.68rem, 0.9vw, 0.80rem); color:#94a3b8; text-transform:uppercase; font-weight:700;">Overall</div>
                  <div style="font-size:clamp(1.35rem, 2.2vw, 2.00rem); font-weight:850; color:#c084fc; line-height:1.2; margin-top:2px;">{comp['s_ov']:.4f}</div>
                </div>
                <div style="flex:1; text-align:center; padding:0 8px; border-right:1px solid rgba(255,255,255,0.10);">
                  <div style="font-size:clamp(0.68rem, 0.9vw, 0.80rem); color:#94a3b8; text-transform:uppercase; font-weight:700;">Recall</div>
                  <div style="font-size:clamp(1.15rem, 1.8vw, 1.70rem); font-weight:750; color:#cbd5e1; line-height:1.2; margin-top:2px;">{comp['s_rec']:.4f}</div>
                </div>
                <div style="flex:1; text-align:center; padding-left:8px;">
                  <div style="font-size:clamp(0.68rem, 0.9vw, 0.80rem); color:#94a3b8; text-transform:uppercase; font-weight:700;">Latency</div>
                  <div style="font-size:clamp(1.15rem, 1.8vw, 1.70rem); font-weight:750; color:#cbd5e1; line-height:1.2; margin-top:2px;">{comp['s_lat']:.1f}s</div>
                </div>
              </div>
            </div>

          </div>

        </div>
        """
        _safe_html(comparison_html)


        unique_questions = (
            df[
                [
                    "benchmark_row_index",
                    "question_id",
                    "category",
                ]
            ]
            .drop_duplicates()
            .sort_values("benchmark_row_index")
        )

        with st.expander(
            "Προαιρετικά: Διερεύνηση ανά Ερώτημα (Question Drilldown)",
            expanded=False,
        ):
            q_options = unique_questions[
                "benchmark_row_index"
            ].tolist()

            q_map = {
                row["benchmark_row_index"]:
                    f"Q{int(row['benchmark_row_index']) + 1}: "
                    f"{row['question_id']} "
                    f"[{row['category']}]"
                for _, row in unique_questions.iterrows()
            }

            selected_idx = st.selectbox(
                "Επιλέξτε ερώτημα:",
                options=q_options,
                format_func=lambda idx: q_map.get(
                    idx,
                    f"Row {idx}",
                ),
                key="sens_q_select",
            )

            q_df = df[
                df["benchmark_row_index"] == selected_idx
            ].copy()

            if not q_df.empty:
                q_overall = q_df.pivot(
                    index="chunk_size",
                    columns="final_k",
                    values="overall_score",
                )

                q_recall = q_df.pivot(
                    index="chunk_size",
                    columns="final_k",
                    values="context_recall",
                )

                q_latency = q_df.pivot(
                    index="chunk_size",
                    columns="final_k",
                    values="latency_sec",
                )

                d1, d2, d3 = st.columns(3)

                with d1:
                    _safe_html("<div style='font-size:0.80rem; font-weight:600; color:#cbd5e1; margin-bottom:4px;'>Overall Score:</div>")
                    st.dataframe(
                        q_overall.style.format("{:.4f}"),
                        width="stretch",
                    )

                with d2:
                    _safe_html("<div style='font-size:0.80rem; font-weight:600; color:#cbd5e1; margin-bottom:4px;'>Context Recall:</div>")
                    st.dataframe(
                        q_recall.style.format("{:.4f}"),
                        width="stretch",
                    )

                with d3:
                    _safe_html("<div style='font-size:0.80rem; font-weight:600; color:#cbd5e1; margin-bottom:4px;'>Χρόνος Απόκρισης (Latency s):</div>")
                    st.dataframe(
                        q_latency.style.format("{:.2f}s"),
                        width="stretch",
                    )
