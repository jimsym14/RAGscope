import base64
import hashlib
import json
import os
import re
import time
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

import chromadb
import pypdf
import streamlit as st
from llama_index.core import Document, PromptTemplate, Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
if os.path.exists(os.path.expanduser("~/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5")):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

from config import (
    BASE_DIR,
    CHROMA_CANDIDATE_PATHS,
    CHROMA_PATH,
    DATA_DIR,
    get_chroma_path,
)
from prompts import (
    FINANCIAL_AUDITOR_TEMPLATE,
    STREAMING_SYSTEM_PROMPT_TEMPLATE,
)


CHATBOT_COLLECTION_NAME = "apple_financials_chat_clean"


@st.cache_resource(show_spinner=False)
def get_chroma_client(path: Optional[str] = None) -> chromadb.PersistentClient:
    
    target_path = path or get_chroma_path()
    return chromadb.PersistentClient(path=target_path)


@st.cache_resource(show_spinner=False)
def get_chroma_collection(collection_name: str = "apple_financials", path: Optional[str] = None) -> Any:
    
    client = get_chroma_client(path)
    return client.get_or_create_collection(collection_name)


@st.cache_resource(show_spinner=False)
def get_chatbot_collection(path: Optional[str] = None) -> Any:
    'The original collection stays available to evaluation. Exact duplicates are collapsed only when text, source file, and source offsets all match.'
    client = get_chroma_client(path)
    source = client.get_or_create_collection("apple_financials")
    try:
        target = client.get_collection(CHATBOT_COLLECTION_NAME)
    except Exception:
        kwargs = {"name": CHATBOT_COLLECTION_NAME}
        if source.metadata is not None:
            kwargs["metadata"] = source.metadata
        target = client.create_collection(**kwargs)

    if target.count() == 0 and source.count() > 0:
        data = source.get(include=["documents", "metadatas", "embeddings"])
        seen = set()
        records = []
        source_embeddings = data.get("embeddings")
        if source_embeddings is None:
            source_embeddings = []
        for record_id, document, metadata, embedding in zip(
            data.get("ids") or [],
            data.get("documents") or [],
            data.get("metadatas") or [],
            source_embeddings,
        ):
            meta = metadata or {}
            try:
                node_payload = json.loads(meta.get("_node_content", "{}"))
            except (TypeError, ValueError):
                node_payload = {}
            normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", document or "")).casefold().strip()
            # File and offsets are part of identity to protect repeated passages
            # and identical boilerplate that belongs to distinct filings.
            identity = (
                hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
                str(meta.get("file_name") or meta.get("filename") or ""),
                str(meta.get("start_char_idx", node_payload.get("start_char_idx", ""))),
                str(meta.get("end_char_idx", node_payload.get("end_char_idx", ""))),
            )
            if not normalized or identity in seen:
                continue
            seen.add(identity)
            records.append((record_id, document, meta, embedding))

        for offset in range(0, len(records), 400):
            batch = records[offset:offset + 400]
            target.add(
                ids=[row[0] for row in batch],
                documents=[row[1] for row in batch],
                metadatas=[row[2] for row in batch],
                embeddings=[row[3].tolist() if hasattr(row[3], "tolist") else list(row[3]) for row in batch],
            )
        print(
            f"Chatbot index initialized from apple_financials: {source.count()} source records, "
            f"{len(records)} retained records, {source.count() - len(records)} exact duplicate ingestions removed."
        )
    return target


def get_indexed_files_summary(collection: Optional[Any] = None) -> Dict[str, int]:
    
    try:
        coll = collection or get_chroma_collection()
        data = coll.get(include=["metadatas"])
        raw_meta = data.get("metadatas") or []
        file_counts: Dict[str, int] = {}
        for m in raw_meta:
            if m:
                fn = m.get("file_name") or m.get("filename", "Unknown")
                file_counts[fn] = file_counts.get(fn, 0) + 1
        return dict(sorted(file_counts.items()))
    except Exception as e:
        print(f"Error fetching indexed files summary: {e}")
        return {}


def delete_document_from_index(filename: str, collection: Optional[Any] = None) -> bool:
    
    try:
        coll = collection or get_chroma_collection()
        try:
            coll.delete(where={"file_name": filename})
        except Exception:
            pass
        try:
            coll.delete(where={"filename": filename})
        except Exception:
            pass
        return True
    except Exception as e:
        print(f"Error deleting document '{filename}': {e}")
        return False


@st.cache_resource(show_spinner=False)
def get_shared_embedding_model(model_name: str = "BAAI/bge-small-en-v1.5") -> HuggingFaceEmbedding:
    
    return HuggingFaceEmbedding(model_name=model_name)


def patch_ollama_for_thinking() -> None:
    
    orig_stream_chat = Ollama.stream_chat
    if not hasattr(Ollama, "_patched_for_thinking"):
        def patched_stream_chat(self, messages, **kwargs):
            chat_gen = orig_stream_chat.__get__(self, Ollama)(messages, **kwargs)

            def thinking_generator():
                in_think = False
                for resp in chat_gen:
                    thinking_delta = resp.additional_kwargs.get("thinking_delta") if resp.additional_kwargs else None
                    if thinking_delta:
                        if not in_think:
                            resp.delta = "<think>\n" + thinking_delta
                            in_think = True
                        else:
                            resp.delta = thinking_delta
                    else:
                        if in_think:
                            resp.delta = "\n</think>\n\n" + (resp.delta or "")
                            in_think = False
                    yield resp
            return thinking_generator()

        setattr(Ollama, "stream_chat", patched_stream_chat)
        setattr(Ollama, "_patched_for_thinking", True)


@st.cache_resource(show_spinner=False)
def load_engine(
    model_name: str,
    context_window: int,
    request_timeout: float,
    enable_timeout: bool = False,
    collection_name: str = CHATBOT_COLLECTION_NAME,
    path: Optional[str] = None
) -> Tuple[VectorStoreIndex, PromptTemplate]:
    
    embed = get_shared_embedding_model()
    Settings.embed_model = embed
    actual_timeout = float(request_timeout) if (enable_timeout and request_timeout > 0) else 86400.0

    patch_ollama_for_thinking()

    from config import MODEL_REGISTRY
    m_meta = MODEL_REGISTRY.get(model_name, {})
    try:
        import streamlit as st
        num_predict = int(st.session_state.get("max_thinking", 768)) if m_meta.get("has_thinking") else int(m_meta.get("num_predict", 1024))
    except Exception:
        num_predict = int(m_meta.get("num_predict", 1024))

    Settings.llm = Ollama(
        model=model_name,
        request_timeout=actual_timeout,
        temperature=0.0,
        context_window=context_window,
        additional_kwargs={"num_predict": num_predict}
    )
    Settings.chunk_size = st.session_state.get("chunk_size", 512)

    coll = get_chroma_collection(collection_name=collection_name, path=path)
    vs = ChromaVectorStore(chroma_collection=coll)
    idx = VectorStoreIndex.from_vector_store(vs, embed_model=embed)

    tmpl = PromptTemplate(
        "You are a strict Expert Financial Auditor.\n"
        "---------------------\n{context_str}\n---------------------\n"
        "Given the context information, answer the query.\n"
        "Query: {query_str}\nAnswer: "
    )
    return idx, tmpl


def show_custom_toast(placeholder: Any, message: str, status: str = "indexing", delay: float = 0.0) -> None:
    
    styles = {
        "indexing": {
            "border": "#3b82f6",
            "bg": "rgba(30, 58, 138, 0.45)",
            "text": "#93c5fd",
            "badge_bg": "#1d4ed8",
            "badge_color": "#eff6ff",
            "badge_text": "INDEXING"
        },
        "success": {
            "border": "#10b981",
            "bg": "rgba(6, 78, 59, 0.45)",
            "text": "#6ee7b7",
            "badge_bg": "#047857",
            "badge_color": "#ecfdf5",
            "badge_text": "DONE"
        },
        "skip": {
            "border": "#f59e0b",
            "bg": "rgba(120, 53, 15, 0.45)",
            "text": "#fcd34d",
            "badge_bg": "#b45309",
            "badge_color": "#fffbeb",
            "badge_text": "EXISTS"
        },
        "error": {
            "border": "#ef4444",
            "bg": "rgba(127, 29, 29, 0.45)",
            "text": "#fca5a5",
            "badge_bg": "#b91c1c",
            "badge_color": "#fef2f2",
            "badge_text": "ERROR"
        }
    }
    st_style = styles.get(status, styles["indexing"])

    html = f"""
    <div style="
        display: flex;
        align-items: center;
        gap: 8px;
        background: {st_style['bg']};
        border-left: 3px solid {st_style['border']};
        color: {st_style['text']};
        padding: 6px 10px;
        border-radius: 4px;
        font-family: 'AHG', 'Segoe UI', sans-serif;
        font-size: 11.5px;
        margin-bottom: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    ">
        <span style="
            background: {st_style['badge_bg']};
            color: {st_style['badge_color']};
            font-size: 9px;
            font-weight: 700;
            padding: 2px 7px;
            border-radius: 6px;
            flex-shrink: 0;
        ">{st_style['badge_text']}</span>
        <div style="flex-grow: 1; line-height: 1.35; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
            {message}
        </div>
    </div>
    """
    placeholder.markdown(html, unsafe_allow_html=True)
    if delay > 0:
        time.sleep(delay)


def extract_text_from_pdf(file_obj: Any) -> str:
    
    reader = pypdf.PdfReader(file_obj)
    return "\n".join([page.extract_text() or "" for page in reader.pages])


def process_and_index_files(
    files: List[Any],
    idx: Optional[VectorStoreIndex] = None,
    collection: Optional[Any] = None
) -> Tuple[List[str], List[str], List[str]]:
    
    if not files:
        return [], [], []

    toast_placeholder = st.empty()

    pdf_files = [f for f in files if getattr(f, "name", "").lower().endswith(".pdf")]
    non_pdf_files = [getattr(f, "name", "unknown") for f in files if not getattr(f, "name", "").lower().endswith(".pdf")]

    if non_pdf_files:
        show_custom_toast(toast_placeholder, f"Ignored non-PDF files: {', '.join(non_pdf_files)}", "error", delay=1.0)

    if not pdf_files:
        toast_placeholder.empty()
        return [], [], non_pdf_files

    existing_files: Set[str] = set()
    try:
        coll = collection or get_chroma_collection()
        coll_data = coll.get(include=["metadatas"])
        for m in coll_data.get("metadatas", []):
            if m:
                fn = m.get("file_name") or m.get("filename")
                if fn:
                    existing_files.add(fn)
    except Exception as e:
        print(f"Error checking ChromaDB files: {e}")

    target_idx = idx
    if target_idx is None:
        coll = collection or get_chroma_collection()
        embed = get_shared_embedding_model()
        vs = ChromaVectorStore(chroma_collection=coll)
        target_idx = VectorStoreIndex.from_vector_store(vs, embed_model=embed)

    newly_indexed: List[str] = []
    already_indexed: List[str] = []
    failed: List[str] = []

    seen_in_batch: Set[str] = set()
    files_to_index: List[Any] = []

    for f in pdf_files:
        fname = getattr(f, "name", "")
        if fname in existing_files:
            already_indexed.append(fname)
            show_custom_toast(toast_placeholder, f"{fname} (already in database)", "skip", delay=0.8)
        elif fname in seen_in_batch:
            already_indexed.append(fname)
            show_custom_toast(toast_placeholder, f"{fname} (duplicate in batch)", "skip", delay=0.8)
        else:
            seen_in_batch.add(fname)
            files_to_index.append(f)

    if files_to_index:
        for i, f in enumerate(files_to_index, start=1):
            fname = getattr(f, "name", "")
            show_custom_toast(toast_placeholder, f"{fname} ({i}/{len(files_to_index)})...", "indexing", delay=0.2)
            try:
                text = extract_text_from_pdf(f)
                if text.strip():
                    doc = Document(text=text, metadata={"file_name": fname})
                    target_idx.insert(doc)
                    newly_indexed.append(fname)
                    show_custom_toast(toast_placeholder, f"{fname} indexed & vectorized!", "success", delay=0.9)
                else:
                    failed.append(fname)
                    show_custom_toast(toast_placeholder, f"No text found in {fname}", "error", delay=1.0)
            except Exception as e:
                failed.append(fname)
                show_custom_toast(toast_placeholder, f"Error: {fname} ({e})", "error", delay=1.0)

    time.sleep(0.3)
    toast_placeholder.empty()
    return newly_indexed, already_indexed, failed


@st.cache_resource
def load_reranker(top_k: int = 5):
    
    from llama_index.core.postprocessor import SentenceTransformerRerank
    return SentenceTransformerRerank(model="BAAI/bge-reranker-base", top_n=top_k)


class StreamingResult:
    
    def __init__(self, gen, source_nodes, memory, user_msg):
        self.source_nodes = source_nodes
        self._gen = gen
        self._memory = memory
        self._user_msg = user_msg

    @property
    def response_gen(self):
        full_text = ""
        for chunk in self._gen:
            delta = chunk.delta or ""
            full_text += delta
            yield delta
        if self._memory is not None:
            import re
            from llama_index.core.llms import ChatMessage, MessageRole
            clean_text = re.sub(r'<think>.*?</think>', '', full_text, flags=re.DOTALL).strip()
            self._memory.put(ChatMessage(role=MessageRole.USER, content=self._user_msg))
            self._memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=clean_text))


class DirectStreamChatEngine:
    
    def __init__(
        self,
        retriever,
        memory,
        system_prompt_template: Optional[str] = None,
        node_postprocessors=None,
        llm: Optional[Any] = None
    ):
        self._retriever = retriever
        self._memory = memory
        self._system_prompt_template = system_prompt_template or STREAMING_SYSTEM_PROMPT_TEMPLATE
        self._node_postprocessors = node_postprocessors or []
        self._llm = llm

    @property
    def has_reranker(self) -> bool:
        
        return bool(self._node_postprocessors)

    def retrieve_nodes(self, message: str) -> List[Any]:
        
        return self._retriever.retrieve(message)

    def rerank_nodes(self, nodes: List[Any], message: str) -> List[Any]:
        
        from llama_index.core.schema import QueryBundle
        for postprocessor in self._node_postprocessors:
            nodes = postprocessor.postprocess_nodes(nodes, query_bundle=QueryBundle(message))
        return nodes

    def assemble_context(self, nodes: List[Any], message: str) -> List[Any]:
        
        from llama_index.core.schema import MetadataMode
        from llama_index.core.llms import ChatMessage, MessageRole

        context_str = "\n\n".join([
            n.node.get_content(metadata_mode=MetadataMode.LLM) if hasattr(n, "node") else str(n.text)
            for n in nodes
        ])

        if "{context_str}" in self._system_prompt_template:
            formatted_system = self._system_prompt_template.replace("{context_str}", context_str)
        else:
            formatted_system = (
                f"You are a strict Expert Financial Auditor.\n"
                f"---------------------\n{context_str}\n---------------------\n"
                f"Given the context information, answer the query accurately.\n"
                f"Formatting guidelines:\n"
                f"- For mathematical calculations, write standard clean equations (e.g., (136,700 - 155,041) / 155,041 = -11.83%) rather than raw unrendered bracketed LaTeX codes.\n"
                f"- State both the exact calculated figure and any reported/rounded figure mentioned in the 10-K filings.\n"
            )

        chat_history = self._memory.get_all() if hasattr(self._memory, "get_all") else []
        llm_messages = [ChatMessage(role=MessageRole.SYSTEM, content=formatted_system)]
        llm_messages.extend(chat_history)
        llm_messages.append(ChatMessage(role=MessageRole.USER, content=message))
        return llm_messages

    def stream_from_messages(self, llm_messages: List[Any], nodes: List[Any], message: str) -> StreamingResult:
        
        from llama_index.core import Settings
        active_llm = self._llm or Settings.llm
        stream_generator = active_llm.stream_chat(llm_messages)
        return StreamingResult(stream_generator, nodes, self._memory, message)

    def stream_chat(self, message: str) -> StreamingResult:
        
        nodes = self.retrieve_nodes(message)
        nodes = self.rerank_nodes(nodes, message)
        llm_messages = self.assemble_context(nodes, message)
        return self.stream_from_messages(llm_messages, nodes, message)
