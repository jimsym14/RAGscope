import io
import os
import shutil
from importlib.util import find_spec
from pathlib import Path

import streamlit as st

from config import CHROMA_PATH, FILINGS_DIR, MODEL_BENCHMARK_PRESETS, get_available_ollama_models
from ui.assets import favicon_path, load_css, logo_b64

st.set_page_config(
    page_title="RAGscope — Apple 10-K Intelligence",
    page_icon=favicon_path if os.path.exists(favicon_path) else "🍏",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css()

def render_navigation(current_page: str) -> None:
    with st.sidebar:
        if logo_b64:
            st.markdown(
                f'<div style="margin-bottom:12px; text-align:center;"><img src="data:image/svg+xml;base64,{logo_b64}" style="width:140px; filter:invert(1);" alt="RAGscope"/></div>',
                unsafe_allow_html=True,
            )
        if find_spec("chromadb") is None or find_spec("llama_index") is None:
            st.markdown(
                '<div class="sidebar-setup-hint">Chat and Vector Visualizer need the RAG dependencies.<br><code>pip install -r requirements-rag.txt</code><br>Evaluation works without them.</div>',
                unsafe_allow_html=True,
            )
        for page, label, icon in (
            ("evaluation", "Evaluation", ":material/leaderboard:"),
            ("chat", "Chat", ":material/chat:"),
            ("visualizer", "Vector Visualizer", ":material/view_in_ar:"),
        ):
            if st.button(label, icon=icon, width="stretch",
                         type="primary" if current_page == page else "secondary",
                         key=f"nav_{page}"):
                st.session_state.current_page = page
                st.query_params["page"] = page
                st.rerun()

valid_pages = {"chat", "evaluation", "visualizer"}
requested_page = st.query_params.get("page")
if "current_page" not in st.session_state:
    st.session_state.current_page = requested_page if requested_page in valid_pages else "evaluation"
current_page = st.session_state.current_page
if current_page not in valid_pages:
    current_page = "evaluation"
    st.session_state.current_page = current_page
if st.query_params.get("page") != current_page:
    st.query_params["page"] = current_page

models = get_available_ollama_models() if current_page == "chat" else None
if current_page != "chat" or not models:
    render_navigation(current_page)

if current_page == "evaluation":
    from views.evaluation_view import render_evaluation_view
    render_evaluation_view(None)
elif current_page == "visualizer":
    try:
        from views.visualizer_view import render_visualizer_view
        render_visualizer_view()
    except ImportError as exc:
        st.error("The Vector Visualizer dependencies are unavailable. Install the RAG extras to use this feature.")
        with st.expander("Technical details"):
            st.code(str(exc))
else:
    if not models:
        st.warning(
            "RAG Chat needs Ollama and at least one installed model. Start Ollama with "
            "`ollama serve`, then install a model with `ollama pull llama3.1:8b`. "
            "The Evaluation dashboard remains available without Ollama."
        )
    else:
        try:
            from document_pipeline import (
                get_chroma_collection,
                get_chatbot_collection,
                load_engine,
                process_and_index_files,
            )
            from prompts import load_test_prompts
            from views.chat_view import render_chat_view
            from views.sidebar_view import render_sidebar
            from db import load_chats_from_db
        except ImportError as exc:
            st.error("RAG Chat dependencies are unavailable. Install the RAG extras to use this feature.")
            with st.expander("Technical details"):
                st.code(str(exc))
            st.stop()

        default_model = "llama3.1:8b" if "llama3.1:8b" in models else models[0]
        if "selected_model" not in st.session_state or st.session_state.selected_model not in models:
            st.session_state.selected_model = default_model
            st.session_state._last_configured_model = default_model
            for key, value in MODEL_BENCHMARK_PRESETS.get(default_model, {}).items():
                st.session_state[key] = value
        for key, value in MODEL_BENCHMARK_PRESETS.get(st.session_state.selected_model, {}).items():
            st.session_state.setdefault(key, value)
        for key, value in {
            "active_id": None, "trigger_prompt": None, "rag_mode": "Advanced RAG",
            "sidebar_uploader_key": 0,
        }.items():
            st.session_state.setdefault(key, value)
        if "chats" not in st.session_state:
            try:
                st.session_state.chats = load_chats_from_db()
            except Exception as exc:
                st.session_state.chats = []
                st.warning("Chat history is unavailable. New Chat sessions can still be started.")
                with st.expander("Technical details"):
                    st.code(str(exc))

        def on_model_change() -> None:
            new_model = st.session_state.get("selected_model")
            if new_model and new_model != st.session_state.get("_last_configured_model"):
                for key, value in MODEL_BENCHMARK_PRESETS.get(new_model, {}).items():
                    st.session_state[key] = value
                st.session_state._last_configured_model = new_model
            if str(new_model or "").startswith("llama3.2:3b"):
                st.session_state["use_hyde_lite"] = False

        try:
            collection = get_chatbot_collection()
        except Exception:
            if os.environ.get("RAGSCOPE_VECTOR_STORE"):
                raise
            shutil.rmtree(CHROMA_PATH, ignore_errors=True)
            st.cache_resource.clear()
            collection = get_chatbot_collection()
        if collection.count() == 0:
            demo_paths = sorted(Path(FILINGS_DIR).glob("Apple_10K_*.pdf"))
            demo_files = []
            try:
                demo_files = [io.BytesIO(path.read_bytes()) for path in demo_paths]
                for path, demo_file in zip(demo_paths, demo_files):
                    demo_file.name = path.name
                if demo_files:
                    process_and_index_files(demo_files, None, collection=collection)
            finally:
                for demo_file in demo_files:
                    demo_file.close()

        render_sidebar(models, load_test_prompts(), on_model_change, get_chroma_collection, current_page)
        context_window = st.session_state.get("context_window", 8192)
        timeout = st.session_state.get("request_timeout", 0)
        enable_timeout = st.session_state.get("enable_timeout", False)
        try:
            collection = get_chatbot_collection()
            if collection.count() > 0:
                index, template = load_engine(st.session_state.selected_model, context_window, timeout, enable_timeout)
                render_chat_view(index, template)
            else:
                st.error("The bundled Apple 10-K demo corpus could not be indexed. Check the technical details below.")
                with st.expander("Technical details"):
                    st.code("No documents were added to the local Chroma collection.")
        except Exception as exc:
            st.error("RAG Chat could not open its local vector index. The Evaluation dashboard remains available.")
            with st.expander("Technical details"):
                st.code(str(exc))
