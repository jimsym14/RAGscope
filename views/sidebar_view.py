import os
import time
from typing import Any, Callable, Dict, List
import streamlit as st
from db import delete_chat_from_db
from document_pipeline import (
    delete_document_from_index,
    get_chatbot_collection,
    get_indexed_files_summary,
    process_and_index_files,
)
from config import MODEL_REGISTRY
from ui.assets import logo_b64

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@st.dialog("Environment Configuration (.env)", width="large")
def edit_env_dialog() -> None:
    
    st.caption("Επεξεργασία κλειδιών API (Azure OpenAI, DeepSeek) και μεταβλητών περιβάλλοντος.")
    env_path = os.path.join(BASE_DIR, ".env")

    current_content = ""
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            current_content = f.read()

    new_content = st.text_area(
        ".env File Content",
        value=current_content,
        height=240,
        help="Μπορείτε να επεξεργαστείτε άμεσα τις γραμμές του αρχείου .env (KEY=VALUE)."
    )

    col_s1, col_s2 = st.columns([1, 1])
    with col_s1:
        if st.button("Save & Reload", type="primary", width="stretch", icon=":material/save:"):
            with open(env_path, "w", encoding="utf-8") as f:
                f.write(new_content.strip() + "\n")
            try:
                from dotenv import load_dotenv
                load_dotenv(env_path, override=True)
            except Exception:
                pass
            st.success("Το αρχείο .env αποθηκεύτηκε και φορτώθηκε επιτυχώς!")
            time.sleep(0.5)
            st.rerun()
    with col_s2:
        if st.button("Close", type="secondary", width="stretch", icon=":material/close:"):
            st.rerun()


def render_sidebar(
    models: List[str],
    test_prompts: List[Dict[str, Any]],
    on_model_change: Callable[[], None],
    get_chroma_collection: Callable[[], Any],
    current_page: str
) -> None:
    
    with st.sidebar:
        if logo_b64:
            st.markdown(
                f'<div style="margin-bottom:12px; text-align:center;"><img src="data:image/svg+xml;base64,{logo_b64}" style="width:140px; filter:invert(1);" alt="RAGscope"/></div>',
                unsafe_allow_html=True
            )
        st.markdown('<div class="sidebar-section-title">Navigation</div>', unsafe_allow_html=True)
        col_nav1, col_nav2 = st.columns(2)
        with col_nav1:
            if st.button("Chat", icon=":material/chat:", key="nav_btn_chat", width="stretch", type="primary" if current_page == "chat" else "secondary"):
                if st.session_state.current_page != "chat":
                    st.session_state.current_page = "chat"
                    st.query_params["page"] = "chat"
                    st.rerun()
        with col_nav2:
            if st.button("Evaluation", icon=":material/leaderboard:", key="nav_btn_eval", width="stretch", type="primary" if current_page == "evaluation" else "secondary"):
                if st.session_state.current_page != "evaluation":
                    st.session_state.current_page = "evaluation"
                    st.query_params["page"] = "evaluation"
                    st.rerun()

        if st.button("New Conversation", icon=":material/add:", type="primary", width="stretch"):
            st.session_state.active_id = None
            if st.session_state.current_page != "chat":
                st.session_state.current_page = "chat"
                st.query_params["page"] = "chat"
            st.rerun()

        st.markdown('<div class="sidebar-section-title">Conversations</div>', unsafe_allow_html=True)
        chats_list = st.session_state.get("chats", [])
        if chats_list:
            chat_container = st.container(height=220) if len(chats_list) > 5 else st.container()
            with chat_container:
                for c in reversed(chats_list):
                    col1, col2 = st.columns([0.82, 0.18], gap="small", vertical_alignment="center")
                    is_active = c["id"] == st.session_state.get("active_id") and current_page == "chat"
                    display_title = c.get("title", "Conversation")

                    with col1:
                        if st.button(
                            display_title,
                            key=f"chat_nav_{c['id']}",
                            icon=":material/chat_bubble:",
                            width="stretch",
                            type="primary" if is_active else "secondary"
                        ):
                            st.session_state.active_id = c["id"]
                            if st.session_state.current_page != "chat":
                                st.session_state.current_page = "chat"
                                st.query_params["page"] = "chat"
                            st.rerun()
                    with col2:
                        if st.button("", key=f"chat_del_{c['id']}", icon=":material/delete:", width="stretch"):
                            delete_chat_from_db(c["id"])
                            st.session_state.chats = [chat for chat in chats_list if chat["id"] != c["id"]]
                            if st.session_state.get("active_id") == c["id"]:
                                st.session_state.active_id = None
                            st.rerun()
        else:
            st.caption("No past conversations.")

        st.markdown('<div class="sidebar-section-title">RAG Configuration</div>', unsafe_allow_html=True)
        with st.expander("General Settings", expanded=True):
            st.selectbox("LLM Model", models, key="selected_model", on_change=on_model_change)

            cur_model = st.session_state.get("selected_model", models[0] if models else "")
            m_meta = MODEL_REGISTRY.get(cur_model, {})
            has_thinking = m_meta.get("has_thinking", False)

            if has_thinking:
                THINKING_BUDGET_OPTIONS = [512, 768, 1024]
                THINKING_BUDGET_LABELS = {
                    512: "Short (512)",
                    768: "Balanced (768)",
                    1024: "Extended (1024)"
                }
                st.radio(
                    "Maximum thinking",
                    options=THINKING_BUDGET_OPTIONS,
                    format_func=lambda x: THINKING_BUDGET_LABELS.get(x, str(x)),
                    horizontal=True,
                    key="max_thinking",
                    help="Safety token budget for reasoning & generation."
                )

            st.slider("Context Window", 2048, 32768, step=1024, key="context_window", help="Lower this if heavy models hang or run out of memory.")

            def _on_timeout_toggle():
                if not st.session_state.get("enable_timeout"):
                    st.session_state["request_timeout"] = 0

            enable_to = st.toggle("Enable Request Timeout", key="enable_timeout", on_change=_on_timeout_toggle)
            if enable_to:
                st.slider("Timeout Limit (seconds)", 0, 600, 120, step=10, key="request_timeout", help="Max time to wait for model response. 0 = 0 seconds.")

        with st.expander("Retrieval", expanded=True):
            st.markdown('<div class="retrieval-section-label">Candidates</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="retrieval-candidate-summary"><span>Top candidates</span>'
                '<span class="retrieval-candidate-badge">15 · RRF</span></div>',
                unsafe_allow_html=True,
            )

            st.markdown('<div class="retrieval-section-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="retrieval-section-label">Hybrid search</div>', unsafe_allow_html=True)
            st.toggle(
                "BM25",
                key="use_bm25",
                help="Add keyword results to vector search and combine both lists with RRF.",
            )

            st.markdown('<div class="retrieval-section-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="retrieval-section-label">Answer context</div>', unsafe_allow_html=True)
            st.toggle(
                "Cross-Encoder Reranker",
                key="use_reranker",
                help="Rerank the candidate set before choosing the final context.",
            )
            context_options = [3, 5, 8, 10, 15]
            current_context = st.session_state.get("top_k", 3)
            try:
                current_context = int(current_context)
            except (TypeError, ValueError):
                current_context = 3
            if current_context not in context_options:
                st.session_state["top_k"] = min(context_options, key=lambda value: abs(value - current_context))
            st.selectbox(
                "Final context",
                options=context_options,
                key="top_k",
                help="Chunks passed to the answer model.",
            )

            st.markdown('<div class="retrieval-section-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="retrieval-section-label">Query transformation</div>', unsafe_allow_html=True)
            def _on_hyde_toggle():
                if st.session_state.get("use_hyde"):
                    st.session_state["use_hyde_lite"] = False

            def _on_hyde_lite_toggle():
                if st.session_state.get("use_hyde_lite"):
                    st.session_state["use_hyde"] = False

            st.toggle("HyDE", key="use_hyde", on_change=_on_hyde_toggle, help="Write a hypothetical query with the selected model before retrieval.")
            if not str(st.session_state.get("selected_model", "")).startswith("llama3.2:3b"):
                st.toggle("HyDE Lite · 3B", key="use_hyde_lite", on_change=_on_hyde_lite_toggle, help="Write the hypothetical query with Llama 3.2 3B; the selected model still answers.")
            else:
                st.session_state["use_hyde_lite"] = False

        st.markdown('<div class="sidebar-section-title">Quick Prompts</div>', unsafe_allow_html=True)
        with st.expander("Quick Prompts", expanded=False, icon=":material/bolt:"):
            st.caption("Select a prompt to quickly test against active model:")
            for tp in test_prompts:
                if st.button(f"{tp['label']}", key=f"prompt_btn_{tp['id']}", width="stretch"):
                    st.session_state.trigger_prompt = tp["prompt"]
                    if st.session_state.current_page != "chat":
                        st.session_state.current_page = "chat"
                        st.query_params["page"] = "chat"
                    st.rerun()

        with st.expander("Indexed Documents & Targeting", expanded=True):
            try:
                coll = get_chatbot_collection()
                file_counts = get_indexed_files_summary(coll)
                all_docs = list(file_counts.keys())

                st.markdown('<div class="sidebar-section-title">Document Processing</div>', unsafe_allow_html=True)
                with st.container(border=True):
                    st.slider("Chunk Size (Tokens)", 128, 1024, step=64, key="chunk_size", help="Controls how large the text pieces are when indexing PDFs. To apply a new size, delete and re-upload the document.")

                st.markdown('<div class="sidebar-section-title">Targeting & Filtering</div>', unsafe_allow_html=True)
                with st.container(border=True):
                    st.multiselect("Target Specific Files (@file)", all_docs, key="ui_file_includes", help="Only search within these documents.")
                    st.multiselect("Exclude Specific Files (@without)", all_docs, key="ui_file_excludes", help="Never search within these documents.")

                st.markdown('<div class="sidebar-section-title">Database</div>', unsafe_allow_html=True)
                uploader_key = f"sidebar_files_uploader_{st.session_state.get('sidebar_uploader_key', 0)}"
                sidebar_files = st.file_uploader(
                    "Upload PDF Filings",
                    type=["pdf"],
                    accept_multiple_files=True,
                    key=uploader_key,
                    help="Upload PDF filings to index and vectorize into ChromaDB."
                )
                if sidebar_files:
                    if st.button("Index Uploaded Files", type="primary", icon=":material/upload_file:", key="btn_sidebar_index", width="stretch"):
                        process_and_index_files(sidebar_files, None, collection=coll)
                        st.session_state.sidebar_uploader_key = st.session_state.get("sidebar_uploader_key", 0) + 1
                        time.sleep(0.5)
                        st.rerun()

                if file_counts:
                    st.markdown('<div style="font-size: 11px; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color: #a1a1aa; margin: 12px 0 6px 0;">INDEXED FILINGS</div>', unsafe_allow_html=True)
                    doc_container = st.container(height=210) if len(file_counts) > 4 else st.container()
                    with doc_container:
                        for fname, count in sorted(file_counts.items()):
                            col1, col2 = st.columns([0.82, 0.18], gap="small", vertical_alignment="center")
                            with col1:
                                st.markdown(
                                    f"<div class='indexed-doc-title' title='{fname}'>"
                                    f"<span class='indexed-doc-icon'>📄</span>"
                                    f"<span class='indexed-doc-name'>{fname}</span>"
                                    f"<span class='indexed-doc-badge'>{count}</span>"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )
                            with col2:
                                if st.button("", key=f"del_doc_{fname}", icon=":material/delete:", help=f"Remove {fname} from ChromaDB", width="stretch"):
                                    delete_document_from_index(fname, coll)
                                    st.rerun()

                    st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
                    if st.button("Explore 3D Vector Space", icon=":material/view_in_ar:", key="btn_open_3d_vis", width="stretch", type="primary" if current_page == "visualizer" else "secondary"):
                        if st.session_state.current_page != "visualizer":
                            st.session_state.current_page = "visualizer"
                            st.query_params["page"] = "visualizer"
                            st.rerun()
                else:
                    st.caption("No documents indexed.")
                    st.markdown('<div style="margin-top: 8px;"></div>', unsafe_allow_html=True)
                    if st.button("Explore 3D Vector Space", icon=":material/view_in_ar:", key="btn_open_3d_vis_empty", width="stretch", type="primary" if current_page == "visualizer" else "secondary"):
                        if st.session_state.current_page != "visualizer":
                            st.session_state.current_page = "visualizer"
                            st.query_params["page"] = "visualizer"
                            st.rerun()

            except Exception:
                st.caption("Database not available.")

        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
        if st.button("Edit .env Keys", icon=":material/key:", width="stretch", key="sidebar_edit_env_btn", help="Επεξεργασία API keys και μεταβλητών περιβάλλοντος (.env)"):
            edit_env_dialog()
