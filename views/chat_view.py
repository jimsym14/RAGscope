import base64
import json
import os
import re
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
import streamlit.components.v2 as st_components_v2
import tiktoken
from llama_index.core import Document, Settings
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer

from config import MODEL_REGISTRY
from db import (
    create_chat,
    get_active_chat,
    log_analytics_to_db,
    save_chat_to_db,
)
from document_pipeline import (
    DirectStreamChatEngine,
    load_reranker,
    process_and_index_files,
    show_custom_toast,
)
from ui.assets import (
    build_relevant_snippet,
    format_rag_response,
    format_sources_component,
    render_assistant_message,
    render_live_process,
    title_svg_b64,
)
from ui.chat_scripts import (
    AGENT_RESPONSE_SUFFIX_STREAM as _AGENT_SUFFIX_STREAM,
    render_agent_response_prefix,
    DISABLE_SIDEBAR_STYLE as _DISABLE_SIDEBAR_STYLE,
    GLOBAL_NAV_ARROWS_HTML as _GLOBAL_UI_HTML,
    USER_CARD_HTML as _USER_CARD_HTML,
    USER_CARD_JS as _USER_CARD_JS,
)

user_card_component = st_components_v2.component(
    "user_card_component",
    html=_USER_CARD_HTML,
    js=_USER_CARD_JS,
)


def _split_partial_response(response_text: str) -> Tuple[str, Optional[str]]:
    
    partial_answer = response_text.strip()
    partial_thinking = None
    if "<think>" in response_text and "</think>" in response_text:
        answer_parts = response_text.split("</think>", 1)
        partial_thinking = answer_parts[0].replace("<think>", "").strip()
        partial_answer = answer_parts[1].strip()
    elif "<think>" in response_text:
        partial_thinking = response_text.split("<think>", 1)[1].strip()
        partial_answer = ""
    return partial_answer, partial_thinking


def _record_stopped_generation(generation_id: str, response_text: str = "") -> None:
    
    chat = get_active_chat()
    if chat and chat.get("title"):
        first_user_message = next(
            (
                str(message.get("content", ""))
                for message in chat.get("messages", [])
                if message.get("role") == "user"
            ),
            "",
        )
        legacy_auto_title = first_user_message[:45] + "…"
        if len(first_user_message) > 45 and chat["title"] == legacy_auto_title:
            chat["title"] = first_user_message
            save_chat_to_db(chat)
    if not chat:
        return

    partial_answer, partial_thinking = _split_partial_response(response_text)
    stopped_message = {
        "role": "assistant",
        "content": partial_answer,
        "thinking": partial_thinking,
        "sources": [],
        "sig": "",
        "rag_trace": None,
        "stopped": True,
        "generation_id": generation_id,
    }
    messages = chat["messages"]
    existing = next(
        (m for m in messages if m.get("generation_id") == generation_id),
        None,
    )
    if existing is None:
        messages.append(stopped_message)
    else:
        existing.update(stopped_message)
    save_chat_to_db(chat)


def _stop_active_generation() -> None:
    
    active = st.session_state.get("_active_generation")
    if not active or not active.get("running"):
        return
    active["stop_requested"] = True
    _record_stopped_generation(
        active["id"],
        active.get("response_text", ""),
    )
    st.session_state._active_generation = None


def render_chat_view(idx, tmpl):
    trigger = st.session_state.trigger_prompt
    if trigger:
        st.session_state.trigger_prompt = None

    user_input = st.chat_input("Ask me anything about Apple's 10-K financial filings...", accept_file="multiple")

    user_input_text = ""
    uploaded_files = []

    if user_input:
        if isinstance(user_input, dict):
            user_input_text = str(user_input.get("text", ""))
            uploaded_files = user_input.get("files", [])
        elif hasattr(user_input, "text") and hasattr(user_input, "files"):
            user_input_text = str(getattr(user_input, "text", ""))
            uploaded_files = getattr(user_input, "files", [])
        else:
            user_input_text = str(user_input)

    prompt: str = str(trigger or user_input_text)

    st.iframe(_GLOBAL_UI_HTML, height="content")

    chat = get_active_chat()
    if prompt.strip() and chat is None:
        chat = create_chat(prompt)
        
    messages = chat["messages"] if chat else []

    if not messages and not prompt:
        st.markdown(
            f'<div class="welcome-container">'
            f'<img src="data:image/svg+xml;base64,{title_svg_b64}" class="welcome-logo" alt="RAGscope"/>'
            f'<div class="welcome-subtitle">Financial Audit &amp; Intelligence Suite for Corporate 10-K Filings</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        if chat is not None:
            active_chat = chat
            def update_chat_title():
                active_chat["title"] = st.session_state.get(f"chat_title_input_{active_chat['id']}", active_chat["title"])
                save_chat_to_db(active_chat)
                
            st.markdown(
                """
                <style>
                body:has([data-testid="stChatInput"]) header[data-testid="stHeader"] {
                    background: rgba(14, 15, 17, 0.78) !important;
                    backdrop-filter: blur(16px) !important;
                    -webkit-backdrop-filter: blur(16px) !important;
                    border-bottom: 1px solid rgba(143, 138, 130, 0.15) !important;
                    height: 70px !important;
                }
                body:has([data-testid="stChatInput"]) .main .block-container,
                body:has([data-testid="stChatInput"]) [data-testid="stMainBlockContainer"] {
                    padding-top: 96px !important;
                }
                div[data-testid="stElementContainer"]:has(input[aria-label="Chat Title"]) {
                    position: fixed !important;
                    top: 15px !important;
                    left: 50vw !important;
                    transform: translateX(-50%) !important;
                    z-index: 999999 !important;
                    width: min(1100px, calc(100vw - 48px)) !important;
                    box-sizing: border-box !important;
                    display: flex !important;
                    justify-content: center !important;
                    align-items: center !important;
                    height: 40px !important;
                    pointer-events: auto !important;
                }
                div[data-testid="stElementContainer"]:has(input[aria-label="Chat Title"]) div[data-testid="stTextInput"],
                div[data-testid="stElementContainer"]:has(input[aria-label="Chat Title"]) div[data-testid="stTextInput"] div,
                div[data-testid="stElementContainer"]:has(input[aria-label="Chat Title"]) div[data-testid="stTextInput"] > div {
                    background: transparent !important;
                    background-color: transparent !important;
                    border: none !important;
                    box-shadow: none !important;
                    height: 100% !important;
                    min-height: 40px !important;
                    width: 100% !important;
                    display: flex !important;
                    justify-content: center !important;
                    align-items: center !important;
                }
                div[data-testid="stElementContainer"]:has(input[aria-label="Chat Title"]) input[aria-label="Chat Title"] {
                    font-size: 26px !important;
                    font-weight: 700 !important;
                    font-family: 'AG', Georgia, serif !important;
                    color: var(--st-text-color, #f8fafc) !important;
                    background: transparent !important;
                    background-color: transparent !important;
                    border: none !important;
                    border-bottom: 2px solid transparent !important;
                    padding: 0 18px !important;
                    margin: 0 auto !important;
                    box-shadow: none !important;
                    border-radius: 0 !important;
                    text-align: center !important;
                    text-overflow: ellipsis !important;
                    white-space: nowrap !important;
                    overflow: hidden !important;
                    line-height: 40px !important;
                    height: 40px !important;
                    min-height: 40px !important;
                    letter-spacing: -0.015em !important;
                    transition: none !important;
                    width: 100% !important;
                    max-width: none !important;
                    min-width: 0 !important;
                    flex: 1 1 auto !important;
                    box-sizing: border-box !important;
                    cursor: text !important;
                }
                div[data-testid="stElementContainer"]:has(input[aria-label="Chat Title"]) input[aria-label="Chat Title"]:hover,
                div[data-testid="stElementContainer"]:has(input[aria-label="Chat Title"]) input[aria-label="Chat Title"]:focus {
                    border-bottom: 2px solid rgba(143, 138, 130, 0.7) !important;
                    background-color: rgba(255, 255, 255, 0.03) !important;
                    outline: none !important;
                }
                body:has([data-testid="stSidebar"][aria-expanded="true"]) div[data-testid="stElementContainer"]:has(input[aria-label="Chat Title"]) {
                    left: calc(50vw + 11rem) !important;
                    width: min(1100px, calc(100vw - 360px)) !important;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.text_input(
                "Chat Title",
                value=chat["title"],
                key=f"chat_title_input_{chat['id']}",
                on_change=update_chat_title,
                label_visibility="collapsed"
            )
            
        for m_idx, msg in enumerate(messages):
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.markdown(f'<div class="user-request-anchor" id="user-req-{m_idx}"></div>', unsafe_allow_html=True)
                    user_card_component(data={"text": msg["content"], "files": msg.get("files", [])}, key=f"user_msg_{m_idx}_{msg.get('content', '')[:10]}")
                else:
                    stopped = bool(msg.get("stopped"))
                    has_stopped_output = bool(
                        (msg.get("content") or "").strip()
                        or (msg.get("thinking") or "").strip()
                    )
                    if stopped and not has_stopped_output:
                        st.markdown(
                            '<div class="stopped-query-notice" role="status">'
                            '<span class="stopped-query-icon" aria-hidden="true"></span>'
                            '<span>Stopped by user</span></div>',
                            unsafe_allow_html=True,
                        )
                        continue

                    rendered_msg = render_assistant_message(
                        content=msg.get("content", ""),
                        thinking=msg.get("thinking", None),
                        sources=msg.get("sources", []),
                        sig=msg.get("sig", ""),
                        rag_trace=None if stopped else msg.get("rag_trace"),
                        is_live_streaming=False
                    )
                    if stopped:
                        rendered_msg += (
                            '<div class="stopped-inline-status" role="status">'
                            '<span class="stopped-query-icon" aria-hidden="true"></span>'
                            '<span>Stopped by user</span></div>'
                        )
                    st.markdown(
                        render_agent_response_prefix(
                            worked_for_seconds=None if stopped else msg.get("work_duration_sec")
                        )
                        + rendered_msg + _AGENT_SUFFIX_STREAM,
                        unsafe_allow_html=True,
                    )

    if uploaded_files:
        newly_idx, already_idx, failed_idx = process_and_index_files(uploaded_files, idx)
        
        if not prompt or not prompt.strip():
            time.sleep(0.5)
            st.rerun()

    if prompt and prompt.strip():
        current_pdf_files = [f.name for f in uploaded_files if f.name.lower().endswith(".pdf")] if uploaded_files else []

        if chat is None:
            chat = create_chat(prompt)
            messages = chat["messages"]

        with st.chat_message("user"):
            st.markdown(f'<div class="user-request-anchor" id="user-req-{len(messages)}"></div>', unsafe_allow_html=True)
            user_card_component(data={"text": prompt, "files": current_pdf_files}, key=f"user_msg_{len(messages)}_new")
        messages.append({"role": "user", "content": prompt, "files": current_pdf_files})

        with st.chat_message("assistant"):
            start_time = time.time()
            generation_id = uuid.uuid4().hex
            st.session_state._active_generation = {
                "id": generation_id,
                "running": True,
                "response_text": "",
            }
            live_agent_prefix = render_agent_response_prefix(started_at_ms=round(start_time * 1000))
            placeholder = st.empty()
            live_process_steps = []
            live_process_active = "Preparing query"

            def update_live_stage(label: str, complete_previous: bool = True) -> None:
                nonlocal live_process_active
                if complete_previous and live_process_active:
                    live_process_steps.append(live_process_active)
                live_process_active = label
                placeholder.markdown(
                    live_agent_prefix + render_live_process(live_process_steps, live_process_active) + _AGENT_SUFFIX_STREAM,
                    unsafe_allow_html=True,
                )

            update_live_stage("Preparing query", complete_previous=False)
            
            stop_btn_placeholder = st.empty()
            with stop_btn_placeholder.container():
                stop_btn = st.button(
                    "Stop generating",
                    type="secondary",
                    help="Stop generating",
                    key="stop_generating_btn",
                    on_click=_stop_active_generation,
                )
            
            from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator, FilterCondition
            
            prompt_str = str(prompt or "")
            inline_includes = re.findall(r'@file:([^\s]+)', prompt_str)
            inline_excludes = re.findall(r'@without:([^\s]+)', prompt_str)
            
            combined_includes = list(set(inline_includes + st.session_state.get("ui_file_includes", [])))
            combined_excludes = list(set(inline_excludes + st.session_state.get("ui_file_excludes", [])))
            
            clean_prompt = re.sub(r'@file:[^\s]+', '', prompt_str)
            clean_prompt = re.sub(r'@without:[^\s]+', '', clean_prompt)
            clean_prompt = clean_prompt.strip() or prompt_str
            
            filter_list = []
            if combined_includes:
                filter_list.append(MetadataFilter(key="file_name", value=combined_includes, operator=FilterOperator.IN))
            if combined_excludes:
                filter_list.append(MetadataFilter(key="file_name", value=combined_excludes, operator=FilterOperator.NIN))
                
            filters = None
            if filter_list:
                filters = MetadataFilters(filters=filter_list, condition=FilterCondition.AND)

            first_token_time = None
            full_response_text = ""
            chat_response = None
            
            try:
                retriever_top_k = st.session_state.get("top_k", 5)
                use_rerank = st.session_state.get("use_reranker", False)
                use_hybrid = st.session_state.get("use_bm25", False)
                use_hypo = st.session_state.get("use_hyde", False)
                use_hyde_lite = st.session_state.get("use_hyde_lite", False)
                retrieval_k = 15

                selected_model_name = st.session_state.get("selected_model", "llama3.2:3b")
                c_win = st.session_state.get("context_window", 8192)
                enable_timeout = st.session_state.get("enable_timeout", False)
                timeout = st.session_state.get("request_timeout", 0)
                actual_timeout = float(timeout) if (enable_timeout and timeout > 0) else 86400.0
                m_meta = MODEL_REGISTRY.get(selected_model_name, {})
                model_has_thinking = m_meta.get("has_thinking", False)
                num_predict = int(st.session_state.get("max_thinking", 768)) if model_has_thinking else int(m_meta.get("num_predict", 1024))
                from llama_index.llms.ollama import Ollama
                active_llm = Ollama(model=selected_model_name, request_timeout=actual_timeout, temperature=0.0,
                                    context_window=c_win, additional_kwargs={"num_predict": num_predict})
                Settings.llm = active_llm
                if use_hypo:
                    update_live_stage("Preparing retrievers")
                elif use_hyde_lite:
                    update_live_stage("Preparing retrievers")
                else:
                    update_live_stage("Preparing retrievers")
                rag_trace = {"stages": [{"label": "Query", "detail": "Cleaned user query submitted for retrieval."}]}
                bm25_error = None
                rrf_active = False
                
                base_retriever = None
                from llama_index.core.retrievers import QueryFusionRetriever
                from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
                vector_retriever = idx.as_retriever(similarity_top_k=retrieval_k, filters=filters)
                if use_hybrid:
                    try:
                        from llama_index.core.schema import TextNode
                        from llama_index.retrievers.bm25 import BM25Retriever
                        
                        all_data = idx.vector_store.client.get(include=["metadatas", "documents"])
                        nodes = []
                        if all_data and "documents" in all_data and "metadatas" in all_data:
                            docs_list = all_data.get("documents") or []
                            metas_list = all_data.get("metadatas") or []
                            for doc, meta in zip(docs_list, metas_list):
                                fn = meta.get("file_name", "") if meta else ""
                                if combined_includes and fn not in combined_includes:
                                    continue
                                if combined_excludes and fn in combined_excludes:
                                    continue
                                if doc and doc.strip():
                                    nodes.append(TextNode(text=doc, metadata=meta))
                        
                        if nodes:
                            bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=retrieval_k)
                            base_retriever = QueryFusionRetriever(
                                [vector_retriever, bm25_retriever],
                                similarity_top_k=retrieval_k,
                                num_queries=1,
                                mode=FUSION_MODES.RECIPROCAL_RANK,
                                use_async=False,
                                llm=Settings.llm
                            )
                            rrf_active = True
                    except Exception as exc:
                        bm25_error = f"BM25 setup/retrieval unavailable: {type(exc).__name__}: {exc}"
                
                if base_retriever is None:
                    if use_hybrid:
                        base_retriever = vector_retriever
                        rag_trace["stages"].append({"label": "BM25 fallback", "status": "fallback", "detail": bm25_error or "BM25 could not initialize (empty eligible corpus); dense vector retrieval used."})
                    else:
                        try:
                            base_retriever = QueryFusionRetriever(
                                [vector_retriever],
                                similarity_top_k=retrieval_k,
                                num_queries=1,
                                mode=FUSION_MODES.RECIPROCAL_RANK,
                                use_async=False,
                                llm=Settings.llm,
                            )
                            rrf_active = True
                        except Exception:
                            base_retriever = vector_retriever
                    
                final_retriever = base_retriever
                hyde_transform = None
                if use_hypo:
                    from llama_index.core.indices.query.query_transform import HyDEQueryTransform
                    hyde = HyDEQueryTransform(include_original=True, llm=active_llm)
                    hyde_transform = hyde
                    rag_trace["stages"].append({"label": "HyDE", "detail": f"Hypothetical query generated with selected model {selected_model_name}; original query included."})
                elif use_hyde_lite:
                    from llama_index.core.indices.query.query_transform import HyDEQueryTransform
                    lite_llm = Ollama(model="llama3.2:3b", request_timeout=actual_timeout, temperature=0.0,
                                      context_window=min(c_win, 8192), additional_kwargs={"num_predict": 128})
                    hyde = HyDEQueryTransform(include_original=True, llm=lite_llm)
                    hyde_transform = hyde
                    rag_trace["stages"].append({"label": "HyDE Lite · 3B", "detail": "Hypothetical query generated with llama3.2:3b (128 token cap); selected model generates the final answer."})
                else:
                    rag_trace["stages"].append({"label": "HyDE off", "detail": "No hypothetical query transform executed."})

                if use_hybrid and not bm25_error:
                    rag_trace["stages"].append({"label": f"Vector search ≤{retrieval_k}", "detail": f"Dense retriever candidate limit is {retrieval_k}; actual returned count is recorded after retrieval."})
                    rag_trace["stages"].append({"label": f"BM25 search ≤{retrieval_k}", "detail": f"BM25 candidate limit is {retrieval_k}; this is a configured maximum, not a returned count."})
                else:
                    rag_trace["stages"].append({"label": f"Vector ≤{retrieval_k}", "detail": f"Dense vector retrieval candidate limit is {retrieval_k}."})
                
                node_postprocessors = []
                if use_rerank:
                    reranker_model = load_reranker()
                    reranker_model.top_n = retriever_top_k
                    node_postprocessors.append(reranker_model)

                history = []
                for m in messages[:-1]:
                    if m["role"] == "user":
                        history.append(ChatMessage(role=MessageRole.USER, content=m["content"]))
                    else:
                        raw_c = m.get("content", "")
                        clean_c = re.sub(r'<think>.*?</think>', '', raw_c, flags=re.DOTALL)
                        clean_c = re.sub(r'<details.*?</details>', '', clean_c, flags=re.DOTALL).strip()
                        if clean_c:
                            history.append(ChatMessage(role=MessageRole.ASSISTANT, content=clean_c))
                memory = ChatMemoryBuffer.from_defaults(chat_history=history, token_limit=4096)

                chat_engine = DirectStreamChatEngine(
                    retriever=final_retriever,
                    memory=memory,
                    system_prompt_template=tmpl.template,
                    node_postprocessors=node_postprocessors,
                    llm=active_llm,
                )

                transformed_query = None
                try:
                    if hyde_transform is not None and transformed_query is None:
                        from llama_index.core.schema import QueryBundle
                        hyde_model_name = "llama3.2:3b" if use_hyde_lite else selected_model_name
                        update_live_stage(f"Writing hypothetical query · {hyde_model_name}")
                        transformed_query = hyde_transform.run(QueryBundle(query_str=clean_prompt), metadata=None)
                        update_live_stage("Searching · " + ("Vector + BM25" if use_hybrid and not bm25_error else "Vector"))
                        nodes = base_retriever.retrieve(transformed_query)
                    else:
                        update_live_stage("Searching · " + ("Vector + BM25" if use_hybrid and not bm25_error else "Vector"))
                        nodes = chat_engine.retrieve_nodes(clean_prompt)
                except Exception as retrieval_exc:
                    if hyde_transform is not None:
                        for stage in rag_trace["stages"]:
                            if stage.get("label") in {"HyDE", "HyDE Lite · 3B"}:
                                stage["status"] = "fallback"
                                stage["label"] = "HyDE failed" if transformed_query is None else "HyDE retrieval fallback"
                                stage["detail"] = f"Query transform/retrieval failed: {type(retrieval_exc).__name__}: {retrieval_exc}. Original-query retrieval was attempted."
                        transformed_query = None
                    if use_hybrid:
                        bm25_error = f"Hybrid retrieval failed: {type(retrieval_exc).__name__}: {retrieval_exc}"
                        rrf_active = False
                        fallback_retriever = idx.as_retriever(similarity_top_k=retrieval_k, filters=filters)
                        chat_engine._retriever = fallback_retriever
                        nodes = fallback_retriever.retrieve(clean_prompt)
                        rag_trace["stages"].append({"label": "BM25 fallback", "status": "fallback", "detail": bm25_error + "; dense vector retrieval completed."})
                    elif use_hypo or use_hyde_lite:
                        nodes = base_retriever.retrieve(clean_prompt)
                    elif not rrf_active:
                        nodes = vector_retriever.retrieve(clean_prompt)
                    else:
                        rrf_active = False
                        nodes = vector_retriever.retrieve(clean_prompt)
                        rag_trace["stages"].append({"label": "RRF fallback", "status": "fallback", "detail": f"Single-vector fusion failed ({type(retrieval_exc).__name__}); dense retrieval completed."})
                if use_hybrid and not bm25_error:
                    rag_trace["stages"].append({"label": f"RRF → {len(nodes)}", "detail": f"QueryFusionRetriever completed; {len(nodes)} fused nodes were actually returned."})
                elif rrf_active:
                    rag_trace["stages"].append({"label": f"RRF → {len(nodes)}", "detail": f"Single-retriever vector fusion completed with a candidate limit of {retrieval_k}; {len(nodes)} nodes were returned."})
                else:
                    rag_trace["stages"].append({"label": f"Vector out {len(nodes)}", "detail": f"Vector retriever actually returned {len(nodes)} nodes."})
                rrf_output_count = len(nodes)
                if chat_engine.has_reranker:
                    rerank_input_count = len(nodes)
                    update_live_stage(f"Reranking · {rerank_input_count} → {retriever_top_k}")
                    nodes = chat_engine.rerank_nodes(nodes, clean_prompt)[:retriever_top_k]
                    rag_trace["stages"].append({"label": f"Rerank {rerank_input_count} → {len(nodes)}", "detail": f"BAAI/bge-reranker-base received {rerank_input_count} nodes and returned {len(nodes)} (top_k={retriever_top_k})."})
                else:
                    nodes = nodes[:retriever_top_k]
                    rag_trace["stages"].append({"label": "Rerank off", "detail": "No cross-encoder reranking executed."})

                update_live_stage("Preparing context")
                llm_messages = chat_engine.assemble_context(nodes, clean_prompt)
                rag_trace["stages"].append({"label": f"Context → {len(nodes)} chunks", "detail": f"{len(nodes)} nodes assembled into the answer context."})

                pipeline = []
                if transformed_query is not None:
                    pipeline.append("HyDE Lite" if use_hyde_lite else "HyDE")
                pipeline.append(f"Vector ≤{retrieval_k}")
                if use_hybrid and not bm25_error:
                    pipeline.append(f"BM25 ≤{retrieval_k}")
                elif bm25_error:
                    pipeline.append("BM25 fallback")
                if rrf_active and not bm25_error:
                    pipeline.append(f"RRF {rrf_output_count}")
                if chat_engine.has_reranker:
                    pipeline.append(f"Rerank {rerank_input_count}→{len(nodes)}")
                pipeline.append(f"Context {len(nodes)}")
                rag_trace["pipeline"] = pipeline

                techniques = []
                if transformed_query is not None:
                    techniques.append("HyDE")
                if use_hybrid and not bm25_error:
                    techniques.append("BM25")
                if chat_engine.has_reranker:
                    techniques.append("Reranking")
                rag_trace["techniques"] = techniques

                update_live_stage(f"Generating answer · {selected_model_name} · waiting for first token")
                generation_start_time = time.time()
                chat_response = chat_engine.stream_from_messages(llm_messages, nodes, clean_prompt)
                rag_trace["stages"].append({"label": f"Answer · {selected_model_name}", "detail": f"Final response generated by {selected_model_name}."})

                last_render_time = 0.0
                for chunk in chat_response.response_gen:
                    if first_token_time is None:
                        first_token_time = time.time()
                        
                    full_response_text += chunk
                    active_generation = st.session_state.get("_active_generation")
                    if active_generation and active_generation.get("id") == generation_id:
                        active_generation["response_text"] = full_response_text
                    now = time.time()
                    if now - last_render_time >= 0.04:
                        rendered_msg = format_rag_response(full_response_text, is_live_streaming=True)
                        live_trace = render_live_process(live_process_steps, live_process_active, streaming=True)
                        placeholder.markdown(live_agent_prefix + _DISABLE_SIDEBAR_STYLE + live_trace + rendered_msg + _AGENT_SUFFIX_STREAM, unsafe_allow_html=True)
                        last_render_time = now
                    time.sleep(0.008)

                rendered_msg = format_rag_response(full_response_text, is_live_streaming=True)
                live_trace = render_live_process(live_process_steps, live_process_active, streaming=True)
                placeholder.markdown(live_agent_prefix + _DISABLE_SIDEBAR_STYLE + live_trace + rendered_msg + _AGENT_SUFFIX_STREAM, unsafe_allow_html=True)

                stop_btn_placeholder.empty()
                duration = max(time.time() - start_time, 0.1)
                rag_duration = max(generation_start_time - start_time, 0.0)
                generation_duration = max(time.time() - generation_start_time, 0.1)
                ttft = (first_token_time - generation_start_time) if first_token_time else generation_duration
                
                enc = tiktoken.get_encoding("cl100k_base")
                total_tokens = len(enc.encode(full_response_text))
                tps = total_tokens / max(generation_duration - ttft, 0.1)
                actual_model_display = getattr(active_llm, "model", selected_model_name)
                rag_trace["process"] = [*live_process_steps, live_process_active]
                rag_trace["summary"] = {
                    "rag_time": rag_duration, "ttft": ttft, "generation": generation_duration, "speed": tps,
                    "tokens": total_tokens, "model": actual_model_display,
                }
                rag_trace["process"][-1] = f"Generated answer · {selected_model_name}"
                
                sig = f"""
<div class='rag-telemetry-container' data-no-copy='true' style='display: flex; justify-content: flex-start; align-items: center; margin-top: 10px;'>
    <div class='copy-btn' onclick='if(window.copyAgentText){{window.copyAgentText(this);}}' title='Copy to clipboard' style='display: flex; align-items: center; justify-content: center; width: 32px; height: 32px; border-radius: 8px; background-color: rgba(143, 138, 130, 0.1); border: 1px solid rgba(143, 138, 130, 0.2); color: #8f8a82; cursor: pointer; transition: all 0.2s;'>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
    </div>
    <div style='display: inline-flex; align-items: center; gap: 12px; font-size: 11px; opacity: 0.6; font-family: monospace; background-color: rgba(143, 138, 130, 0.1); padding: 8px 16px; border-radius: 20px; border: 1px solid rgba(143, 138, 130, 0.2); margin-left: auto;'>
        <span title='Time spent on query preparation, retrieval, reranking and context assembly'>RAG time: {rag_duration:.1f}s</span>
        <span title='Time to first generated answer token'>LLM TTFT: {ttft:.1f}s</span>
        <span title='Answer generation time'>Gen: {generation_duration:.1f}s</span>
        <span title='Tokens per second after first token'>Speed: {tps:.1f} t/s</span>
        <span title='Total tokens generated'>Tokens: {total_tokens}</span>
        <span style='opacity: 0.3;'>|</span>
        <span>Generated by {actual_model_display}</span>
    </div>
</div>
"""
                msg_sources = []
                retrieved_files = []
                retrieved_chunks = []
                if hasattr(chat_response, 'source_nodes') and chat_response.source_nodes:
                    for node in chat_response.source_nodes:
                        file_name = node.metadata.get('file_name', 'Unknown') if (hasattr(node, 'metadata') and node.metadata) else 'Unknown'
                        if file_name not in retrieved_files:
                            retrieved_files.append(file_name)
                        
                        raw_text = node.text if hasattr(node, 'text') else str(node)
                        snip_preview = build_relevant_snippet(raw_text, clean_prompt, max_chars=220)
                        
                        raw_score = getattr(node, 'score', None)
                        parsed_score = round(float(raw_score), 4) if isinstance(raw_score, (int, float)) else 0.0
                        
                        fy_match = re.search(r'(20\d\d)', file_name)
                        fy = f"FY{fy_match.group(1)}" if fy_match else ""
                        
                        node_id = getattr(getattr(node, 'node', None), 'id_', '') or getattr(node, 'id_', '')
                        
                        msg_sources.append({
                            "file_name": file_name,
                            "fiscal_year": fy,
                            "score": parsed_score,
                            "text_snippet": snip_preview,
                            "chunk_id": str(node_id)
                        })
                        retrieved_chunks.append({
                            "file_name": file_name,
                            "score": parsed_score,
                            "text": raw_text
                        })

                final_answer = full_response_text.strip()
                thinking_text = None
                has_thinking = False

                if "<think>" in full_response_text and "</think>" in full_response_text:
                    parts = full_response_text.split("</think>", 1)
                    thinking_text = parts[0].replace("<think>", "").strip()
                    final_answer = parts[1].strip()
                    has_thinking = True
                elif "<think>" in full_response_text:
                    thinking_text = full_response_text.split("<think>", 1)[1].strip()
                    final_answer = ""
                    has_thinking = True

                messages.append({
                    "role": "assistant",
                    "content": final_answer,
                    "thinking": thinking_text,
                    "sources": msg_sources,
                    "sig": sig,
                    "rag_trace": rag_trace,
                    "work_duration_sec": duration,
                })
                st.session_state._active_generation = None
                save_chat_to_db(chat)

                final_rendered = render_assistant_message(
                    content=final_answer,
                    thinking=thinking_text,
                    sources=msg_sources,
                    sig=sig,
                    rag_trace=rag_trace,
                    is_live_streaming=False,
                    animate_rag_trace=True,
                )
                placeholder.markdown(
                    render_agent_response_prefix(worked_for_seconds=duration)
                    + final_rendered + _AGENT_SUFFIX_STREAM,
                    unsafe_allow_html=True,
                )
                
                analytics_data = {
                    "chat_id": chat["id"],
                    "timestamp": datetime.now().isoformat(),
                    "model_name": actual_model_display,
                    "query_text": clean_prompt,
                    "response_text": final_answer,
                    "has_thinking": has_thinking,
                    "thinking_text": thinking_text,
                    "ttft_sec": ttft,
                    "total_duration_sec": duration,
                    "speed_tps": tps,
                    "total_tokens": total_tokens,
                    "advanced_rag_enabled": bool(use_rerank or use_hybrid or use_hypo or use_hyde_lite),
                    "use_reranker": st.session_state.get("use_reranker", False),
                    "use_bm25": st.session_state.get("use_bm25", False),
                    "use_hyde": st.session_state.get("use_hyde", False),
                    "chunk_size": st.session_state.get("chunk_size", 512),
                    "top_k": st.session_state.get("top_k", 3),
                    "context_window": st.session_state.get("context_window", 8192),
                    "retrieved_files": json.dumps(retrieved_files),
                    "retrieved_chunks": json.dumps(retrieved_chunks),
                    "ragas_faithfulness": None,
                    "ragas_answer_relevancy": None,
                    "ragas_context_precision": None,
                    "ragas_context_recall": None,
                    "ragas_overall": None
                }
                log_analytics_to_db(analytics_data)
                
            except BaseException as e:
                if "RerunException" in type(e).__name__ or "StopException" in type(e).__name__:
                    _record_stopped_generation(generation_id, full_response_text)
                    st.session_state._active_generation = None
                else:
                    failed_duration = max(time.time() - start_time, 0.1)
                    err_msg = str(e).strip() or f"{type(e).__name__}"
                    err_html = f"\n\n*Error generating response: {err_msg}*"
                    if messages and messages[-1]["role"] == "assistant":
                        messages[-1]["content"] += err_html
                        messages[-1]["work_duration_sec"] = failed_duration
                        placeholder.error(messages[-1]["content"])
                    else:
                        messages.append({"role": "assistant", "content": err_html, "thinking": None, "sources": [], "sig": "", "work_duration_sec": failed_duration})
                        placeholder.error(err_html)
                    save_chat_to_db(chat)
                
            stop_btn_placeholder.empty()
            st.rerun()
