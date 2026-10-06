from typing import Any, Dict
import numpy as np
import pandas as pd
import streamlit as st
import dashboard_data as dashboard_data
from config import MODEL_REGISTRY
from ui.icons import get_judge_icon_svg
from db import EXP_DB_PATH

MODEL_COLORS = {k: v["color"] for k, v in MODEL_REGISTRY.items()}
MODEL_COLOR_SEQ = [v["color"] for v in MODEL_REGISTRY.values()]


@st.cache_data(ttl=5, show_spinner=False)
def load_cached_experiments(db_path: str, judge_provider: str) -> pd.DataFrame:
    
    return dashboard_data.load_evaluation_experiments(db_path, judge_provider)


@st.cache_data(ttl=60, show_spinner=False)
def get_cached_system_health() -> Dict[str, Any]:
    
    return dashboard_data.check_system_health()


@st.cache_data(ttl=15, show_spinner=False)
def get_cached_adversarial_summary(db_path: str) -> pd.DataFrame:
    
    return dashboard_data.load_adversarial_summary(db_path)


def fmt_score(val: Any, default: str = "N/A") -> str:
    
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    try:
        return f"{float(val):.3f}"
    except (ValueError, TypeError):
        return default


def fmt_tps(val: Any, default: str = "N/A") -> str:
    
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    try:
        return f"{float(val):.1f} t/s"
    except (ValueError, TypeError):
        return default


def fmt_sec(val: Any, default: str = "N/A") -> str:
    
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    try:
        return f"{float(val):.2f}s"
    except (ValueError, TypeError):
        return default


def apply_clean_chart_theme(fig: Any, height: int = 380) -> Any:
    
    layout_updates: dict[str, Any] = dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif", size=14, color="#e2e8f0"),
        legend=dict(
            bgcolor="rgba(20, 20, 19, 0.8)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            borderwidth=1,
            font=dict(size=14, color="#cbd5e1"),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        hoverlabel=dict(
            bgcolor="rgba(20, 20, 19, 0.95)",
            bordercolor="rgba(225, 29, 72, 0.4)",
            font=dict(family="-apple-system, BlinkMacSystemFont, 'Inter', sans-serif", size=14, color="#f8fafc")
        ),
        margin=dict(l=50, r=30, t=40, b=45),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.06)",
            zerolinecolor="rgba(255, 255, 255, 0.12)",
            tickfont=dict(size=13.5, color="#9c9c94"),
            title_font=dict(size=15, color="#cbd5e1"),
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.06)",
            zerolinecolor="rgba(255, 255, 255, 0.12)",
            tickfont=dict(size=13.5, color="#9c9c94"),
            title_font=dict(size=15, color="#cbd5e1"),
        ),
        title=dict(text=""),
    )
    if height:
        layout_updates["height"] = int(height)
    fig.update_layout(**layout_updates)
    if hasattr(fig, "layout") and hasattr(fig.layout, "title") and fig.layout.title is not None:
        fig.layout.title.text = ""

    for trace in fig.data:
        if hasattr(trace, 'textposition'):
            trace.textposition = None
        if hasattr(trace, 'mode') and trace.mode:
            cleaned_mode = trace.mode.replace('+text', '').replace('text+', '').replace('text', 'markers')
            if cleaned_mode != trace.mode:
                trace.mode = cleaned_mode if cleaned_mode else 'markers'
    return fig


def render_card_header(
    title: str, 
    subtitle: str = "", 
    badge: str = "", 
    icon: str = "", 
    is_judge_graph: bool = False,
    peek_primary_key: str = "",
    peek_compare_key: str = ""
) -> None:
    
    active_judge = st.session_state.get("active_judge_provider", "azure_gpt5_mini")

    if (is_judge_graph or bool(peek_primary_key)) and peek_primary_key and peek_compare_key:
        other_judge = "deepseek_v4" if active_judge == "azure_gpt5_mini" else "azure_gpt5_mini"
        other_name = "DeepSeek V4" if active_judge == "azure_gpt5_mini" else "GPT-5 Mini"
        primary_svg = get_judge_icon_svg(active_judge, size=18, opacity=0.45, grayscale=True)
        compare_svg = get_judge_icon_svg(other_judge, size=18, opacity=1.0, grayscale=False)
        right_html = f'<div class="eval-card-header-right"><div class="judge-peek-btn" tabindex="0" data-primary="{peek_primary_key}" data-compare="{peek_compare_key}" title="Πατήστε παρατεταμένα για σύγκριση με {other_name}"><span class="judge-icon-normal">{primary_svg}</span><span class="judge-icon-active">{compare_svg}</span></div></div>'
    else:
        right_html = ''

    if icon:
        clean_icon = icon.replace(":material/", "").replace(":", "").strip()
        icon_html = f'<span class="material-symbols-rounded eval-card-icon">{clean_icon}</span>'
    else:
        icon_html = ''

    sub_html = f'<div class="eval-card-sub">{subtitle}</div>' if subtitle else ''
    html = f'<div class="eval-card-header"><div class="eval-card-header-top"><div class="eval-card-title">{icon_html}<span>{title}</span></div>{right_html}</div>{sub_html}</div>'
    st.markdown(html, unsafe_allow_html=True)


_JUDGE_PEEK_CONTROLLER_HTML = """
<script>
(function() {
    function setupJudgePeek() {
        const rootDoc = (window.parent && window.parent.document) ? window.parent.document : document;

        // Handle Hold-to-Peek buttons (Charts)
        const btns = rootDoc.querySelectorAll('.judge-peek-btn:not([data-peek-bound="true"])');
        
        btns.forEach(btn => {
            btn.setAttribute('data-peek-bound', 'true');
            const pKey = btn.getAttribute('data-primary');
            const cKey = btn.getAttribute('data-compare');
            if (!pKey || !cKey) return;
            
            const card = btn.closest('[data-testid="stVerticalBlockBorderWrapper"]') || btn.closest('.stVerticalBlock') || rootDoc;
            
            function findTarget(key) {
                if (!key) return null;
                return card.querySelector(`[class*="${key}"], [data-testid*="${key}"], .st-key-${key}`) ||
                       card.querySelector(`div:has(> .st-key-${key})`) ||
                       rootDoc.querySelector(`[class*="${key}"], .st-key-${key}`) ||
                       rootDoc.querySelector(`div:has(> .st-key-${key})`);
            }
            
            let isHolding = false;
            
            function activateCompare(e) {
                if (e) {
                    try { e.preventDefault(); } catch(err) {}
                }
                if (isHolding) return;
                isHolding = true;
                btn.classList.add('active-peek');
                const primaryEl = findTarget(pKey);
                const compareEl = findTarget(cKey);
                
                if (primaryEl) {
                    primaryEl.style.setProperty('display', 'none', 'important');
                }
                if (compareEl) {
                    compareEl.style.setProperty('display', 'block', 'important');
                }
                window.dispatchEvent(new Event('resize'));
            }
            
            function deactivateCompare() {
                if (!isHolding) return;
                isHolding = false;
                btn.classList.remove('active-peek');
                const primaryEl = findTarget(pKey);
                const compareEl = findTarget(cKey);
                
                if (compareEl) {
                    compareEl.style.setProperty('display', 'none', 'important');
                }
                if (primaryEl) {
                    primaryEl.style.setProperty('display', 'block', 'important');
                }
                window.dispatchEvent(new Event('resize'));
            }
            
            btn.addEventListener('mousedown', activateCompare);
            btn.addEventListener('touchstart', activateCompare, { passive: false });
            
            btn.addEventListener('mouseup', deactivateCompare);
            btn.addEventListener('mouseleave', deactivateCompare);
            btn.addEventListener('touchend', deactivateCompare);
            btn.addEventListener('touchcancel', deactivateCompare);
            
            btn.addEventListener('keydown', (e) => {
                if (e.key === ' ' || e.key === 'Enter') {
                    activateCompare(e);
                }
            });
            btn.addEventListener('keyup', (e) => {
                if (e.key === ' ' || e.key === 'Enter') {
                    deactivateCompare();
                }
            });
            
            const initialCompare = findTarget(cKey);
            if (initialCompare) {
                initialCompare.style.setProperty('display', 'none', 'important');
            }
        });
    }
    
    setupJudgePeek();
    const targetDoc = (window.parent && window.parent.document) ? window.parent.document.body : document.body;
    const observer = new MutationObserver(() => setupJudgePeek());
    observer.observe(targetDoc, { childList: true, subtree: true });
})();
</script>
"""
