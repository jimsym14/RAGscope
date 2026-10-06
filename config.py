import json
import os
import urllib.request
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.expanduser(os.environ.get("RAGSCOPE_DATA_DIR", os.path.join(BASE_DIR, "data"))))
UI_DIR = os.path.join(BASE_DIR, "ui")
FILINGS_DIR = os.path.join(BASE_DIR, "10-K_Filings")

DB_PATH = os.path.join(DATA_DIR, "chats.db")
EXP_DB_PATH = os.path.join(DATA_DIR, "experiments.db")
CHROMA_PATH = os.path.abspath(os.path.expanduser(os.environ.get(
    "RAGSCOPE_VECTOR_STORE", os.path.join(DATA_DIR, "vector_store", "chroma")
)))
SENSITIVITY_DB_PATH = os.path.abspath(os.path.expanduser(os.environ.get(
    "RAGSCOPE_SENSITIVITY_DB", os.path.join(DATA_DIR, "sensitivity_ablation", "sensitivity_runs.db")
)))


def get_sensitivity_db_path() -> Optional[str]:
    """Return the configured sensitivity results database when it exists."""
    return SENSITIVITY_DB_PATH if os.path.isfile(SENSITIVITY_DB_PATH) else None


CHROMA_CANDIDATE_PATHS = [CHROMA_PATH]


def get_chroma_path() -> str:
    """Return the configured, project-relative ChromaDB path."""
    return CHROMA_PATH

DEFAULT_MODELS: List[str] = [
    "llama3.1:8b",
    "llama3.2:3b",
    "qwen3.5:9b",
    "deepseek-r1:8b",
    "gemma4:latest",
]

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "llama3.2:3b": {
        "id": "llama3.2:3b",
        "display_name": "Llama 3.2 3B",
        "params": "3.2B",
        "params_num": 3.2,
        "quant": "Q4_K_M",
        "arch": "llama",
        "role": "Edge / Efficiency",
        "has_thinking": False,
        "default_context": 8192,
        "num_predict": 1024,
        "expected_tps": "30–35 t/s",
        "color": "#10B981",
    },
    "llama3.1:8b": {
        "id": "llama3.1:8b",
        "display_name": "Llama 3.1 8B",
        "params": "8.0B",
        "params_num": 8.0,
        "quant": "Q4_K_M",
        "arch": "llama",
        "role": "General Baseline",
        "has_thinking": False,
        "default_context": 8192,
        "num_predict": 1024,
        "expected_tps": "12–14 t/s",
        "color": "#3B82F6",
    },
    "deepseek-r1:8b": {
        "id": "deepseek-r1:8b",
        "display_name": "DeepSeek R1 8B",
        "params": "8.2B",
        "params_num": 8.2,
        "quant": "Q4_K_M",
        "arch": "qwen3",
        "role": "Reasoning / Thinking",
        "has_thinking": True,
        "default_context": 4096,
        "num_predict": 1024,
        "expected_tps": "10–14 t/s",
        "color": "#8B5CF6",
    },
    "qwen3.5:9b": {
        "id": "qwen3.5:9b",
        "display_name": "Qwen 3.5 9B",
        "params": "9.7B",
        "params_num": 9.7,
        "quant": "Q4_K_M",
        "arch": "qwen35",
        "role": "Modern Structured / Financial",
        "has_thinking": False,
        "default_context": 2048,
        "num_predict": 1024,
        "expected_tps": "9–13 t/s",
        "color": "#F97316",
    },
    "gemma4:latest": {
        "id": "gemma4:latest",
        "display_name": "Gemma 4 8B",
        "params": "8.0B",
        "params_num": 8.0,
        "quant": "Q4_K_M",
        "arch": "gemma4",
        "role": "Modern Google Architecture",
        "has_thinking": False,
        "default_context": 8192,
        "num_predict": 1024,
        "expected_tps": "9–13 t/s",
        "color": "#EAB308",
    },
}

MODEL_BENCHMARK_PRESETS: Dict[str, Dict[str, Any]] = {
    "llama3.1:8b": {
        "similarity_top_k": 5,
        "bm25_top_k": 5,
        "top_k": 3,
        "hyde_hypo_max_tokens": 128,
        "context_window": MODEL_REGISTRY["llama3.1:8b"]["default_context"],
        "request_timeout": 0,
        "enable_timeout": False,
        "chunk_size": 512,
        "advanced_rag_enabled": False,
        "use_bm25": False,
        "use_reranker": False,
        "use_hyde": False,
        "temperature": 0.0,
        "max_thinking": 1024,
    },
    "llama3.2:3b": {
        "similarity_top_k": 5,
        "bm25_top_k": 5,
        "top_k": 3,
        "hyde_hypo_max_tokens": 128,
        "context_window": MODEL_REGISTRY["llama3.2:3b"]["default_context"],
        "request_timeout": 0,
        "enable_timeout": False,
        "chunk_size": 512,
        "advanced_rag_enabled": False,
        "use_bm25": False,
        "use_reranker": False,
        "use_hyde": False,
        "temperature": 0.0,
        "max_thinking": 1024,
    },
    "deepseek-r1:8b": {
        "similarity_top_k": 5,
        "bm25_top_k": 5,
        "top_k": 3,
        "hyde_hypo_max_tokens": 128,
        "context_window": MODEL_REGISTRY["deepseek-r1:8b"]["default_context"],
        "request_timeout": 0,
        "enable_timeout": False,
        "chunk_size": 512,
        "advanced_rag_enabled": True,
        "use_bm25": True,
        "use_reranker": True,
        "use_hyde": False,
        "temperature": 0.0,
        "max_thinking": 768,
    },
    "qwen3.5:9b": {
        "similarity_top_k": 5,
        "bm25_top_k": 5,
        "top_k": 3,
        "hyde_hypo_max_tokens": 128,
        "context_window": MODEL_REGISTRY["qwen3.5:9b"]["default_context"],
        "request_timeout": 0,
        "enable_timeout": False,
        "chunk_size": 512,
        "advanced_rag_enabled": False,
        "use_bm25": False,
        "use_reranker": False,
        "use_hyde": False,
        "temperature": 0.0,
        "max_thinking": 1024,
    },
    "gemma4:latest": {
        "similarity_top_k": 5,
        "bm25_top_k": 5,
        "top_k": 3,
        "hyde_hypo_max_tokens": 128,
        "context_window": MODEL_REGISTRY["gemma4:latest"]["default_context"],
        "request_timeout": 0,
        "enable_timeout": False,
        "chunk_size": 512,
        "advanced_rag_enabled": False,
        "use_bm25": False,
        "use_reranker": False,
        "use_hyde": False,
        "temperature": 0.0,
        "max_thinking": 1024,
    },
}


def get_available_ollama_models() -> List[str]:
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", headers={"User-Agent": "RAGscope"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name") for m in data.get("models", []) if m.get("name")]
                return models
    except Exception:
        pass
    return []


JUDGE_DISPLAY = {
    "deepseek_v4": {"display_name": "DeepSeek V4 Flash", "short_name": "DeepSeek V4"},
    "azure_gpt5_mini": {"display_name": "GPT-5-mini (Azure)", "short_name": "GPT-5-mini"},
}


def get_judge_config_status(provider_id: str):
    """Check judge environment settings without importing the research provider stack."""
    if provider_id == "deepseek_v4":
        ready = len(os.getenv("DEEPSEEK_API_KEY", "").strip()) >= 10
        message = "Ready (DeepSeek API Key configured)" if ready else "Missing DEEPSEEK_API_KEY in environment (.env)"
        meta = {"provider": "DeepSeek V4 Flash", "model": "deepseek-chat", "api_key_configured": ready,
                "base_url": "https://api.deepseek.com"}
        return ready, message, meta
    if provider_id == "azure_gpt5_mini":
        key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini").strip()
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview").strip()
        missing = []
        if len(key) < 10:
            missing.append("AZURE_OPENAI_API_KEY")
        if not endpoint.startswith("http"):
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not deployment:
            missing.append("AZURE_OPENAI_DEPLOYMENT")
        ready = not missing
        message = (f"Ready (Azure deployment '{deployment}' configured)" if ready
                   else f"Missing Azure configuration: {', '.join(missing)}")
        meta = {"provider": "GPT-5-mini (Azure)", "endpoint": endpoint or "Not Set",
                "deployment": deployment, "api_version": api_version,
                "api_key_configured": len(key) >= 10, "reasoning_effort": "low",
                "max_completion_tokens": 2048}
        return ready, message, meta
    raise ValueError(f"Unknown judge provider: {provider_id}")


__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "UI_DIR",
    "DB_PATH",
    "EXP_DB_PATH",
    "CHROMA_CANDIDATE_PATHS",
    "get_chroma_path",
    "CHROMA_PATH",
    "DEFAULT_MODELS",
    "MODEL_REGISTRY",
    "MODEL_BENCHMARK_PRESETS",
    "get_available_ollama_models",
    "SENSITIVITY_DB_PATH",
    "get_sensitivity_db_path",
]
