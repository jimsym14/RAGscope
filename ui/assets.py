import base64
import html
import os
import re
from typing import Any, Dict, List, Optional
import streamlit as st
from ui.icons import (
    DEEPSEEK_PATH,
    OPENAI_PATH,
    get_database_icon,
    get_deepseek_icon,
    get_file_text_icon,
    get_judge_icon_svg,
    get_openai_icon,
    get_scales_icon,
    get_server_icon,
)

UI_DIR = os.path.dirname(os.path.abspath(__file__))


def b64(path: str) -> str:
    
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


f_ahg_b = b64(os.path.join(UI_DIR, "AlteHaasGroteskBold.ttf"))
f_ahg_r = b64(os.path.join(UI_DIR, "AlteHaasGroteskRegular.ttf"))
f_ag_r = b64(os.path.join(UI_DIR, "AppleGaramond.ttf"))
f_ag_b = b64(os.path.join(UI_DIR, "AppleGaramond-Bold.ttf"))
f_ag_i = b64(os.path.join(UI_DIR, "AppleGaramond-Italic.ttf"))
f_ag_bi = b64(os.path.join(UI_DIR, "AppleGaramond-BoldItalic.ttf"))
f_ahg_fallback = b64(os.path.join(UI_DIR, "IBMPlexSans-Variable.ttf"))
f_ag_fallback = b64(os.path.join(UI_DIR, "EBGaramond-Variable.ttf"))

title_svg_path = os.path.join(UI_DIR, "RAGscope_title.svg")
title_svg_b64 = b64(title_svg_path)
favicon_path = os.path.join(UI_DIR, "RAGscope_transparent.png")
logo_svg_path = os.path.join(UI_DIR, "RAGscope_logo.svg")
logo_svg_b64 = b64(logo_svg_path)
logo_b64 = logo_svg_b64


def load_css() -> None:
    
    css_path = os.path.join(UI_DIR, "styles.css")
    styles_content = ""
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            styles_content = f.read()

    font_css = f"""
    <style>
    @font-face {{ font-family: 'AHG'; src: url('data:font/ttf;base64,{f_ahg_b}') format('truetype'); font-weight: 700; font-style: normal; }}
    @font-face {{ font-family: 'AHG'; src: url('data:font/ttf;base64,{f_ahg_r}') format('truetype'); font-weight: 400; font-style: normal; }}
    @font-face {{ font-family: 'AG'; src: url('data:font/ttf;base64,{f_ag_r}') format('truetype'); font-weight: 400; font-style: normal; }}
    @font-face {{ font-family: 'AG'; src: url('data:font/ttf;base64,{f_ag_b}') format('truetype'); font-weight: 700; font-style: normal; }}
    @font-face {{ font-family: 'AG'; src: url('data:font/ttf;base64,{f_ag_i}') format('truetype'); font-weight: 400; font-style: italic; }}
    @font-face {{ font-family: 'AG'; src: url('data:font/ttf;base64,{f_ag_bi}') format('truetype'); font-weight: 700; font-style: italic; }}
    @font-face {{ font-family: 'AHG Fallback'; src: url('data:font/ttf;base64,{f_ahg_fallback}') format('truetype'); font-weight: 100 700; font-style: normal; }}
    @font-face {{ font-family: 'AG Fallback'; src: url('data:font/ttf;base64,{f_ag_fallback}') format('truetype'); font-weight: 400 800; font-style: normal; }}
    </style>
    """
    st.markdown(font_css, unsafe_allow_html=True)
    if styles_content:
        st.markdown(f"<style>{styles_content}</style>", unsafe_allow_html=True)


def render_loading_screen(title: str, subtitle: str) -> None:
    
    html = f"""
    <div style="
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: #0b0f19;
        z-index: 999999;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 20px;
    ">
        <div style="
            width: 44px;
            height: 44px;
            border-radius: 50%;
            border: 3px solid rgba(255, 255, 255, 0.12);
            border-top-color: #f1f5f9;
            border-right-color: #94a3b8;
            animation: classicWheelSpin 0.75s linear infinite;
            margin-bottom: 22px;
        "></div>
        <h2 style="
            font-family: 'AHG', 'AHG Fallback', 'SF Pro Display', -apple-system, sans-serif;
            font-size: 20px;
            font-weight: 600;
            color: #f8fafc;
            letter-spacing: -0.02em;
            margin: 0 0 8px 0;
        ">{title}</h2>
        <p style="
            font-family: 'AHG', 'AHG Fallback', 'SF Pro Text', -apple-system, sans-serif;
            font-size: 13.5px;
            color: #94a3b8;
            margin: 0;
            max-width: 420px;
            line-height: 1.5;
        ">{subtitle}</p>
    </div>
    <style>
        @keyframes classicWheelSpin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
    </style>
    """
    st.markdown(html, unsafe_allow_html=True)


def highlight_financial_values(text: str) -> str:
    
    if not text:
        return ""
    number = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    scale = r"(?:trillion|billion|million|thousand|[TBMK])"
    amount = rf"\$\s*{number}\s*(?:{scale}\b)?"
    bare_scaled = rf"{number}\s*{scale}\b"
    percent = rf"[+\-−]?\s*{number}\s*%"

    financial_pattern = re.compile(
        rf"(?<![\w$])(?:"
        rf"(?P<range_percent>[+\-−]?\s*{number}\s*%?\s*(?:to|[-–—])\s*[+\-−]?\s*{number}\s*%)"
        rf"|(?P<range_amount>{amount}|{bare_scaled})\s*(?:to|[-–—])\s*(?:{amount}|{bare_scaled})"
        rf"|(?P<per_share>{amount}\s+per\s+share)"
        rf"|(?P<amount>{amount})"
        rf"|(?P<percent>{percent})"
        rf"|(?P<quantity>{bare_scaled}(?:\s+shares?)?)"
        rf")(?![\w%])",
        re.IGNORECASE,
    )
    token_pattern = re.compile(r"```[\s\S]*?```|`[^`\n]*`|</?[A-Za-z][^>]*>")
    text = _unwrap_existing_financial_badges(text)

    parts = []
    cursor = 0
    code_depth = 0
    for match in token_pattern.finditer(text):
        segment = text[cursor:match.start()]
        parts.append(segment if code_depth else _badge_financial_text(segment, financial_pattern))
        token = match.group(0)
        parts.append(token)
        if token.startswith("<"):
            tag = re.match(r"<\s*(/?)\s*(code|pre)\b", token, flags=re.I)
            if tag:
                code_depth = max(0, code_depth + (-1 if tag.group(1) else 1))
        elif token.startswith("```") or token.startswith("`"):
            pass
        cursor = match.end()
    tail = text[cursor:]
    parts.append(tail if code_depth else _badge_financial_text(tail, financial_pattern))
    return "".join(parts)


def _unwrap_existing_financial_badges(text: str) -> str:
    
    span_token = re.compile(r"</?span\b[^>]*>", re.I)
    badge_class = re.compile(r"\bclass\s*=\s*(['\"])[^'\"]*\bfin-num-badge\b[^'\"]*\1", re.I)
    stack = []
    saw_badge = bool(badge_class.search(text))
    output = []
    cursor = 0
    for match in span_token.finditer(text):
        output.append(text[cursor:match.start()])
        token = match.group(0)
        if re.match(r"</", token):
            if stack:
                is_badge = stack.pop()
                if not is_badge:
                    output.append(token)
            elif not saw_badge:
                output.append(token)
        else:
            is_badge = bool(badge_class.search(token))
            saw_badge = saw_badge or is_badge
            stack.append(is_badge)
            if not is_badge:
                output.append(token)
        cursor = match.end()
    output.append(text[cursor:])
    return "".join(output)


def _badge_financial_text(text: str, pattern: re.Pattern) -> str:
    
    type_by_group = {
        "range_percent": "range",
        "range_amount": "range",
        "per_share": "per-share",
        "amount": "amount",
        "percent": "percent",
        "quantity": "quantity",
    }

    def replace(match: re.Match) -> str:
        value = match.group(0)
        value_type = next((kind for group, kind in type_by_group.items() if match.group(group)), "amount")
        return f'<span class="fin-num-badge" data-type="{value_type}">{value}</span>'

    return pattern.sub(replace, text)


def render_assistant_message(
    content: str,
    thinking: Optional[str] = None,
    sources: Optional[List[Dict[str, Any]]] = None,
    sig: Optional[str] = "",
    rag_trace: Optional[Dict[str, Any]] = None,
    is_live_streaming: bool = False,
    animate_rag_trace: bool = False,
) -> str:
    
    clean_content = content or ""
    thinking_text = thinking

    if thinking_text is None and clean_content:
        if "<think>" in clean_content and "</think>" in clean_content:
            parts = clean_content.split("</think>", 1)
            thinking_text = parts[0].replace("<think>", "").strip()
            clean_content = parts[1].strip()
        elif "<think>" in clean_content:
            thinking_text = clean_content.split("<think>", 1)[1].strip()
            clean_content = ""
        elif '<div class="thought-body">' in clean_content:
            m = re.search(r'<div class="thought-body">(.*?)</div>', clean_content, flags=re.DOTALL)
            if m:
                thinking_text = m.group(1).strip()
                clean_content = re.sub(r'<details.*?</details>', '', clean_content, flags=re.DOTALL).strip()

    clean_content = re.sub(r'\\\$(\d[\d\,\.]*)', r'$\1', clean_content)
    clean_content = re.sub(r'\n{3,}', '\n\n', clean_content)

    thinking_html = ""
    if thinking_text:
        thinking_text = re.sub(r'\n{3,}', '\n\n', thinking_text)
        highlighted_thinking = highlight_financial_values(thinking_text)
        if is_live_streaming and not clean_content:
            thinking_html = (
                f'<details class="thought-accordion is-thinking" open>'
                f'<summary>'
                f'<span class="thought-summary-inner">'
                f'<span class="thought-chevron">⌄</span>'
                f'<span class="thought-label-text">Thinking</span>'
                f'</span>'
                f'</summary>'
                f'<div class="thought-body">\n\n{highlighted_thinking}\n\n</div>'
                f'</details>'
            )
        else:
            thinking_html = (
                f'<details class="thought-accordion">'
                f'<summary>'
                f'<span class="thought-summary-inner">'
                f'<span class="thought-chevron">⌄</span>'
                f'<span class="thought-label-text">Thinking</span>'
                f'</span>'
                f'</summary>'
                f'<div class="thought-body">\n\n{highlighted_thinking}\n\n</div>'
                f'</details>'
            )

    highlighted_content = clean_content if is_live_streaming else highlight_financial_values(clean_content)
    sources_html = format_sources_component(sources) if sources else ""
    trace_html = format_rag_trace(rag_trace, animate=animate_rag_trace) if rag_trace else ""
    sig_html = sig or ""

    parts = []
    if trace_html:
        parts.append(trace_html)
    if thinking_html:
        parts.append(thinking_html)
    if highlighted_content:
        parts.append(highlighted_content)
    if sources_html:
        parts.append(f'<div class="post-generation-item post-generation-sources">{sources_html}</div>')
    if sig_html:
        parts.append(f'<div class="post-generation-item post-generation-telemetry">{sig_html}</div>')

    return "\n\n".join(parts)


def format_rag_trace(trace: Dict[str, Any], *, live: bool = False, animate: bool = False) -> str:
    
    if not trace.get("stages"):
        return ""
    process_html = "".join(
        f'<div class="rag-process-step" style="--trace-index:{index}">' + html.escape(str(step)) + "</div>"
        for index, step in enumerate(trace.get("process", []))
    )
    techniques = trace.get("techniques", [])
    title = "RAG trace" + (" · " + " · ".join(techniques) if techniques else "")
    pipeline_steps = trace.get("pipeline", [])
    pipeline_html = ""
    if pipeline_steps:
        pipeline_html = '<div class="rag-pipeline-steps" aria-label="Retrieval pipeline">'
        pipeline_html += '<span class="rag-pipeline-caption">PIPELINE</span>'
        for index, step in enumerate(pipeline_steps):
            if index:
                pipeline_html += '<span class="rag-pipeline-arrow" aria-hidden="true">›</span>'
            pipeline_html += '<span class="rag-pipeline-step">' + html.escape(str(step)) + '</span>'
        pipeline_html += '</div>'
    if live:
        state_attrs = ' rag-trace-live" open'
    elif animate:
        state_attrs = ' rag-trace-final rag-trace-collapse-after" open'
    else:
        state_attrs = ' rag-trace-final"'
    return (
        '<details class="rag-trace-details' + state_attrs + ' data-no-copy="true">'
        '<summary class="rag-trace-summary"><span class="rag-trace-chevron"></span>' + html.escape(title) + (' · in progress' if live else '') + '</summary>'
        '<div class="rag-trace-content">' + pipeline_html + process_html
        + "</div></details>"
    )


def format_rag_response(full_text: str, is_live_streaming: bool = False) -> str:
    
    return render_assistant_message(content=full_text, is_live_streaming=is_live_streaming)


def render_stage_indicator(stage_text: str) -> str:
    
    return (
        f'<div class="rag-stage-indicator">'
        f'<span class="rag-stage-text">{html.escape(stage_text)}</span>'
        f'</div>'
    )


def render_live_process(completed: List[str], active: str, *, streaming: bool = False) -> str:
    
    rows = []
    for index, label in enumerate(completed):
        rows.append(
            f'<div class="rag-process-step" style="--trace-index:{index}">{html.escape(label)}</div>'
        )
    if active:
        rows.append(
            f'<div class="rag-process-active" style="--trace-index:{len(completed)}">'
            f'<span class="rag-process-dot">●</span><span class="rag-stage-text">{html.escape(active)}</span></div>'
        )
    stream_class = " rag-trace-streaming" if streaming else ""
    return (
        '<details class="rag-trace-details rag-trace-live' + stream_class + '" open data-no-copy="true">'
        '<summary class="rag-trace-summary"><span class="rag-trace-chevron"></span>RAG trace · in progress</summary>'
        '<div class="rag-process-list rag-trace-content" style="display:flex;flex-direction:column;gap:2px;margin:8px 0 12px">'
        + "".join(rows) + "</div></details>"
    )


def format_sources_component(sources: List[Dict[str, Any]]) -> str:
    
    if not sources:
        return ""
    count = len(sources)
    cards_html = []
    for s in sources:
        fn = html.escape(str(s.get("file_name", "Document")), quote=True)
        fy = html.escape(str(s.get("fiscal_year", "")), quote=True)
        score = s.get("score")
        score_str = f"Score {score:.4f}" if isinstance(score, (int, float)) and score > 0 else (f"Score {score}" if score else "")
        score_str = html.escape(str(score_str), quote=True)
        snippet = html.escape(str(s.get("text_snippet", "")), quote=True)
        chunk_id = html.escape(str(s.get("chunk_id", "")), quote=True)

        fy_badge = f'<span class="rag-source-badge rag-source-badge-fy">{fy}</span>' if fy else ''
        score_badge = f'<span class="rag-source-badge rag-source-badge-score">{score_str}</span>' if score_str else ''
        title_attr = f' title="Chunk ID: {chunk_id}"' if chunk_id else ''

        card = (
            f'<div class="rag-source-card"{title_attr}>'
            f'<div class="rag-source-header">'
            f'<span class="rag-source-filename">{fn}</span>'
            f'{fy_badge}'
            f'{score_badge}'
            f'</div>'
            f'<div class="rag-source-snippet">"{snippet}"</div>'
            f'</div>'
        )
        cards_html.append(card)

    cards_joined = "\n".join(cards_html)
    return (
        f'<div class="rag-sources-container" data-no-copy="true">'
        f'<details class="rag-sources-accordion">'
        f'<summary>'
        f'<span class="rag-sources-chevron" aria-hidden="true">›</span>'
        f'<span class="rag-sources-title-wrap">Sources · {count}</span>'
        f'</summary>'
        f'<div class="rag-sources-grid">\n{cards_joined}\n</div>'
        f'</details>'
        f'</div>'
    )


def build_relevant_snippet(text: str, query: str, max_chars: int = 220) -> str:
    
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(clean) <= max_chars:
        return clean

    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "for", "from",
        "how", "in", "is", "it", "of", "on", "or", "the", "to", "was", "were", "what",
        "when", "which", "why", "with", "across", "compare", "compared", "during", "about",
        "their", "there", "this", "that", "than", "into", "over", "under", "between",
    }
    token_pattern = re.compile(r"fy\s*(?:20)?\d{2}|r&d|[a-z]+|20\d{2}|\d+(?:\.\d+)?%?", re.IGNORECASE)

    def terms(value: str) -> set[str]:
        found = {re.sub(r"\s+", "", token.casefold()) for token in token_pattern.findall(value)}
        found = {term for term in found if term not in stop_words and len(term) > 1}
        if "r&d" in found or {"research", "development"}.issubset(found):
            found.update({"r&d", "research", "development"})
        for term in tuple(found):
            if term.startswith("fy") and term[2:].isdigit():
                year = term[2:]
                found.add("20" + year if len(year) == 2 else year)
            elif term.isdigit() and len(term) == 4 and term.startswith("20"):
                found.add("fy" + term[2:])
        return found

    query_terms = terms(query)
    query_years = set(re.findall(r"\b20\d{2}\b", query))
    query_years.update("20" + year for year in re.findall(r"\bFY\s*(\d{2})\b", query, re.IGNORECASE))
    if not query_terms:
        return clean[:max_chars].rstrip() + "…"

    text_tokens = [(m.group(0).casefold(), m.start(), m.end()) for m in token_pattern.finditer(clean)]
    candidates = set()
    half = max_chars // 2
    for token, start, _ in text_tokens:
        if token in query_terms:
            candidates.add(max(0, min(start - half, len(clean) - max_chars)))
    if not candidates:
        return clean[:max_chars].rstrip() + "…"

    best_start = 0
    best_score = -1
    for start in candidates:
        end = min(start + max_chars, len(clean))
        window = clean[start:end]
        overlap = len(terms(window) & query_terms)
        lowered = window.casefold()
        phrase_bonus = 4 if "research and development" in lowered else 0
        phrase_bonus += 4 if "r&d" in lowered else 0
        window_terms = terms(window)
        if ("r&d" in window_terms or {"research", "development"}.issubset(window_terms)) and window_terms.intersection({"expense", "expenses"}):
            phrase_bonus += 6
        if "percentage" in window_terms and {"total", "net", "sales"}.issubset(window_terms):
            phrase_bonus += 5
        if "operating" in window_terms and "expenses" in window_terms:
            phrase_bonus += 2
        phrase_bonus += 2 * len(query_years.intersection(window_terms))
        score = overlap + phrase_bonus
        if score > best_score:
            best_start, best_score = start, score

    end = min(best_start + max_chars, len(clean))
    if best_start > 0:
        previous_space = clean.rfind(" ", 0, best_start)
        if previous_space >= 0 and best_start - previous_space < 24:
            best_start = previous_space + 1
            end = min(best_start + max_chars, len(clean))
    if end < len(clean):
        last_space = clean.rfind(" ", best_start, end)
        if last_space > best_start + max_chars * 0.75:
            end = last_space
    snippet = clean[best_start:end].strip()
    if best_start > 0:
        snippet = "…" + snippet
    if end < len(clean):
        snippet = snippet.rstrip() + "…"
    return snippet


__all__ = [
    "UI_DIR",
    "b64",
    "f_ahg_b",
    "f_ahg_r",
    "f_ag_r",
    "f_ag_b",
    "f_ag_i",
    "f_ag_bi",
    "title_svg_path",
    "title_svg_b64",
    "favicon_path",
    "logo_svg_path",
    "logo_svg_b64",
    "logo_b64",
    "load_css",
    "render_loading_screen",
    "render_assistant_message",
    "highlight_financial_values",
    "format_rag_response",
    "render_stage_indicator",
    "render_live_process",
    "format_sources_component",
    "build_relevant_snippet",
    "get_openai_icon",
    "get_deepseek_icon",
    "get_judge_icon_svg",
    "get_server_icon",
    "get_database_icon",
    "get_file_text_icon",
    "get_scales_icon",
    "OPENAI_PATH",
    "DEEPSEEK_PATH",
]
