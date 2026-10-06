

import os
import sys
import types
import json
import time
import re
import sqlite3
import subprocess
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable
import pandas as pd
import numpy as np
from dotenv import load_dotenv


load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
if os.path.exists(os.path.expanduser("~/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5")):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"



from config import MODEL_REGISTRY


CONFIG_PRESETS: Dict[str, Dict[str, Any]] = {
    "CFG-1": {
        "id": "CFG-1",
        "name": "CFG-1 - Naive / Dense RAG",
        "short_name": "CFG-1 (Naive)",
        "description": "Dense vector search only, Top-K=3, No BM25, No Reranker, No HyDE",
        "top_k": 3,
        "initial_retrieval_k": 3,
        "use_bm25": False,
        "use_reranker": False,
        "use_hyde": False,
        "context_window": 8192,
        "temperature": 0.0,
        "category": "Baseline",
    },
    "CFG-2": {
        "id": "CFG-2",
        "name": "CFG-2 - Hybrid RAG (Ablation)",
        "short_name": "CFG-2 (Hybrid)",
        "description": "Dense + BM25 keyword search, Reciprocal Rank Fusion, Top-K=5, No Reranker",
        "top_k": 5,
        "initial_retrieval_k": 5,
        "use_bm25": True,
        "use_reranker": False,
        "use_hyde": False,
        "context_window": 8192,
        "temperature": 0.0,
        "category": "Ablation",
    },
    "CFG-3": {
        "id": "CFG-3",
        "name": "CFG-3 - Advanced RAG / Reranking (Core)",
        "short_name": "CFG-3 (Advanced)",
        "description": "Dense + BM25 -> Top 15 Candidates -> BGE Cross-Encoder -> Final Top 3",
        "top_k": 3,
        "initial_retrieval_k": 15,
        "use_bm25": True,
        "use_reranker": True,
        "use_hyde": False,
        "context_window": 8192,
        "temperature": 0.0,
        "category": "Core Comparison",
    },
    "CFG-4": {
        "id": "CFG-4",
        "name": "CFG-4 - Advanced RAG + HyDE (Ablation)",
        "short_name": "CFG-4 (HyDE)",
        "description": "HyDE Query Transform -> Dense + BM25 -> Top 15 -> BGE Cross-Encoder -> Top 3",
        "top_k": 3,
        "initial_retrieval_k": 15,
        "use_bm25": True,
        "use_reranker": True,
        "use_hyde": True,
        "context_window": 8192,
        "temperature": 0.0,
        "category": "Ablation",
    },
}

CORE_EXPERIMENT_MATRIX = [
    {"run_num": 1, "model": "llama3.2:3b", "config": "CFG-1", "name": "Run_01_Llama3.2_3B_CFG1"},
    {"run_num": 2, "model": "llama3.2:3b", "config": "CFG-3", "name": "Run_02_Llama3.2_3B_CFG3"},
    {"run_num": 3, "model": "llama3.1:8b", "config": "CFG-1", "name": "Run_03_Llama3.1_8B_CFG1"},
    {"run_num": 4, "model": "llama3.1:8b", "config": "CFG-3", "name": "Run_04_Llama3.1_8B_CFG3"},
    {"run_num": 5, "model": "deepseek-r1:8b", "config": "CFG-1", "name": "Run_05_DeepSeek_R1_8B_CFG1"},
    {"run_num": 6, "model": "deepseek-r1:8b", "config": "CFG-3", "name": "Run_06_DeepSeek_R1_8B_CFG3"},
    {"run_num": 7, "model": "qwen3.5:9b", "config": "CFG-1", "name": "Run_07_Qwen3.5_9B_CFG1"},
    {"run_num": 8, "model": "qwen3.5:9b", "config": "CFG-3", "name": "Run_08_Qwen3.5_9B_CFG3"},
    {"run_num": 9, "model": "gemma4:latest", "config": "CFG-1", "name": "Run_09_Gemma4_8B_CFG1"},
    {"run_num": 10, "model": "gemma4:latest", "config": "CFG-3", "name": "Run_10_Gemma4_8B_CFG3"},
]

ABLATION_EXPERIMENT_MATRIX = [
    {"run_num": 11, "model": "llama3.1:8b", "config": "CFG-2", "name": "Run_11_Llama3.1_8B_CFG2_Hybrid"},
    {"run_num": 12, "model": "llama3.1:8b", "config": "CFG-4", "name": "Run_12_Llama3.1_8B_CFG4_HyDE"},
]

from config import CHROMA_CANDIDATE_PATHS, CHROMA_PATH, DATA_DIR, EXP_DB_PATH

DATASET_CANDIDATE_PATHS = [
    os.path.join(DATA_DIR, "thesis_benchmark_103_complete.csv"),
    os.path.join(DATA_DIR, "thesis_benchmark.csv"),
]

DEFAULT_EXP_DB = EXP_DB_PATH
CHECKPOINTS_DIR = os.path.join(DATA_DIR, "checkpoints")

COMPLETED_DIR = os.path.join(CHECKPOINTS_DIR, "completed")

def init_experiments_db(db_path: str = DEFAULT_EXP_DB):
    
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_name TEXT UNIQUE,
                model_name TEXT,
                model_params TEXT,
                model_quant TEXT,
                config_id TEXT,
                config_name TEXT,
                top_k INTEGER,
                initial_retrieval_k INTEGER,
                use_bm25 BOOLEAN,
                use_reranker BOOLEAN,
                use_hyde BOOLEAN,
                context_window INTEGER,
                temperature REAL,
                sample_size INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                avg_faithfulness REAL,
                avg_answer_relevancy REAL,
                avg_context_precision REAL,
                avg_context_recall REAL,
                avg_overall_score REAL,
                avg_ttft_sec REAL,
                avg_retrieval_duration_sec REAL,
                avg_rerank_duration_sec REAL,
                avg_generation_duration_sec REAL,
                avg_total_duration_sec REAL,
                avg_speed_tps REAL,
                avg_visible_tokens REAL,
                avg_thinking_tokens REAL,
                avg_total_tokens REAL,
                gpu_processor_status TEXT,
                status TEXT DEFAULT 'completed',
                notes TEXT
            )
        ''')
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS detailed_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                question_idx INTEGER,
                question TEXT,
                ground_truth TEXT,
                generated_answer TEXT,
                thinking_text TEXT,
                has_thinking BOOLEAN,
                contexts TEXT,
                retrieved_files TEXT,
                category TEXT,
                faithfulness REAL,
                answer_relevancy REAL,
                context_precision REAL,
                context_recall REAL,
                ttft_sec REAL,
                retrieval_duration_sec REAL,
                rerank_duration_sec REAL,
                generation_duration_sec REAL,
                total_duration_sec REAL,
                speed_tps REAL,
                visible_tokens INTEGER,
                thinking_tokens INTEGER,
                total_tokens INTEGER,
                FOREIGN KEY(experiment_id) REFERENCES experiments(id)
            )
        ''')
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS checkpoints (
                experiment_name TEXT PRIMARY KEY,
                model_name TEXT,
                config_json TEXT,
                completed_count INTEGER,
                total_count INTEGER,
                partial_data_json TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        try:
            c.execute("PRAGMA table_info(experiments)")
            existing_cols = [col[1] for col in c.fetchall()]
            new_columns = [
                ("model_params", "TEXT"),
                ("model_quant", "TEXT"),
                ("config_id", "TEXT"),
                ("config_name", "TEXT"),
                ("initial_retrieval_k", "INTEGER"),
                ("use_bm25", "BOOLEAN"),
                ("use_reranker", "BOOLEAN"),
                ("use_hyde", "BOOLEAN"),
                ("context_window", "INTEGER"),
                ("temperature", "REAL"),
                ("sample_size", "INTEGER"),
                ("avg_overall_score", "REAL"),
                ("avg_ttft_sec", "REAL"),
                ("avg_retrieval_duration_sec", "REAL"),
                ("avg_rerank_duration_sec", "REAL"),
                ("avg_generation_duration_sec", "REAL"),
                ("avg_total_duration_sec", "REAL"),
                ("avg_speed_tps", "REAL"),
                ("avg_visible_tokens", "REAL"),
                ("avg_thinking_tokens", "REAL"),
                ("avg_total_tokens", "REAL"),
                ("gpu_processor_status", "TEXT"),
                ("status", "TEXT DEFAULT 'completed'"),
                ("notes", "TEXT"),
            ]
            for col_name, col_type in new_columns:
                if col_name not in existing_cols:
                    c.execute(f"ALTER TABLE experiments ADD COLUMN {col_name} {col_type}")
        except Exception:
            pass

        try:
            c.execute("PRAGMA table_info(detailed_results)")
            existing_cols = [col[1] for col in c.fetchall()]
            new_det_columns = [
                ("question_idx", "INTEGER"),
                ("thinking_text", "TEXT"),
                ("has_thinking", "BOOLEAN"),
                ("retrieved_files", "TEXT"),
                ("category", "TEXT"),
                ("ttft_sec", "REAL"),
                ("retrieval_duration_sec", "REAL"),
                ("rerank_duration_sec", "REAL"),
                ("generation_duration_sec", "REAL"),
                ("total_duration_sec", "REAL"),
                ("speed_tps", "REAL"),
                ("visible_tokens", "INTEGER"),
                ("thinking_tokens", "INTEGER"),
                ("total_tokens", "INTEGER"),
            ]
            for col_name, col_type in new_det_columns:
                if col_name not in existing_cols:
                    c.execute(f"ALTER TABLE detailed_results ADD COLUMN {col_name} {col_type}")
        except Exception:
            pass

        c.execute('''
            CREATE TABLE IF NOT EXISTS judge_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                question_idx INTEGER NOT NULL,
                judge_provider TEXT NOT NULL,
                judge_model TEXT NOT NULL,
                faithfulness REAL,
                answer_relevancy REAL,
                context_precision REAL,
                context_recall REAL,
                overall_score REAL,
                scored_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'completed',
                error_message TEXT,
                evaluation_time_sec REAL,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                FOREIGN KEY(experiment_id) REFERENCES experiments(id),
                UNIQUE(experiment_id, question_idx, judge_provider)
            )
        ''')

        try:
            c.execute("PRAGMA table_info(judge_scores)")
            js_cols = {col[1] for col in c.fetchall()}
            new_js_cols = [
                ("reasoning_tokens", "INTEGER DEFAULT 0"),
                ("judge_version", "TEXT DEFAULT 'v2'"),
                ("error_type", "TEXT"),
                ("retry_count", "INTEGER DEFAULT 0")
            ]
            for col_name, col_type in new_js_cols:
                if col_name not in js_cols:
                    c.execute(f"ALTER TABLE judge_scores ADD COLUMN {col_name} {col_type}")
        except Exception:
            pass

        c.execute('''
            CREATE TABLE IF NOT EXISTS judge_experiment_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                judge_provider TEXT NOT NULL,
                judge_model TEXT NOT NULL,
                avg_faithfulness REAL,
                avg_answer_relevancy REAL,
                avg_context_precision REAL,
                avg_context_recall REAL,
                avg_overall_score REAL,
                scored_questions INTEGER NOT NULL DEFAULT 0,
                total_questions INTEGER NOT NULL DEFAULT 0,
                total_input_tokens INTEGER DEFAULT 0,
                total_output_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                status TEXT DEFAULT 'completed',
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(experiment_id) REFERENCES experiments(id),
                UNIQUE(experiment_id, judge_provider)
            )
        ''')

        conn.commit()

    sync_existing_deepseek_scores(db_path)


def sync_existing_deepseek_scores(db_path: str):
    
    if not os.path.exists(db_path):
        return
    try:
        with sqlite3.connect(db_path) as conn:
            c = conn.cursor()
            c.execute('''
                INSERT OR IGNORE INTO judge_scores 
                (experiment_id, question_idx, judge_provider, judge_model, faithfulness, answer_relevancy, context_precision, context_recall, overall_score, scored_at, status)
                SELECT 
                    experiment_id, 
                    question_idx, 
                    'deepseek_v4', 
                    'deepseek-chat', 
                    faithfulness, 
                    answer_relevancy, 
                    context_precision, 
                    context_recall, 
                    (COALESCE(faithfulness, 0) + COALESCE(answer_relevancy, 0) + COALESCE(context_precision, 0) + COALESCE(context_recall, 0)) / 4.0,
                    CURRENT_TIMESTAMP,
                    'completed'
                FROM detailed_results
                WHERE faithfulness IS NOT NULL
            ''')
            
            c.execute('''
                INSERT OR REPLACE INTO judge_experiment_summaries
                (experiment_id, judge_provider, judge_model, avg_faithfulness, avg_answer_relevancy, avg_context_precision, avg_context_recall, avg_overall_score, scored_questions, total_questions, status, updated_at)
                SELECT 
                    e.id,
                    'deepseek_v4',
                    'deepseek-chat',
                    e.avg_faithfulness,
                    e.avg_answer_relevancy,
                    e.avg_context_precision,
                    e.avg_context_recall,
                    e.avg_overall_score,
                    (SELECT COUNT(*) FROM detailed_results d WHERE d.experiment_id = e.id AND d.faithfulness IS NOT NULL),
                    COALESCE(e.sample_size, 103),
                    CASE WHEN e.avg_faithfulness IS NOT NULL THEN 'completed' ELSE 'pending' END,
                    CURRENT_TIMESTAMP
                FROM experiments e
                WHERE e.avg_faithfulness IS NOT NULL
            ''')
            conn.commit()
    except Exception as e:
        pass


def load_evaluation_experiments(db_path: str = DEFAULT_EXP_DB, judge_provider: str = "deepseek_v4") -> pd.DataFrame:
    'Scores are loaded only from the requested judge provider.'
    if not os.path.exists(db_path):
        return pd.DataFrame()

    with sqlite3.connect(db_path) as conn:
        exp_df = pd.read_sql("SELECT * FROM experiments ORDER BY timestamp DESC", conn)
        if exp_df.empty:
            return exp_df

        exp_df["avg_faithfulness"] = np.nan
        exp_df["avg_answer_relevancy"] = np.nan
        exp_df["avg_context_precision"] = np.nan
        exp_df["avg_context_recall"] = np.nan
        exp_df["avg_overall_score"] = np.nan
        exp_df["scored_questions_count"] = 0
        exp_df["judge_status"] = "unscored"

        try:
            sum_df = pd.read_sql(
                "SELECT * FROM judge_experiment_summaries WHERE judge_provider = ?",
                conn,
                params=(judge_provider,)
            )
            sum_map = sum_df.set_index("experiment_id").to_dict(orient="index") if not sum_df.empty else {}
        except Exception:
            sum_map = {}

        try:
            scores_agg = pd.read_sql("""
                SELECT experiment_id, 
                       AVG(faithfulness) as calc_f,
                       AVG(answer_relevancy) as calc_r,
                       AVG(context_precision) as calc_cp,
                       AVG(context_recall) as calc_cr,
                       AVG(overall_score) as calc_ov,
                       COUNT(CASE WHEN status = 'completed' THEN 1 END) as calc_scored
                FROM judge_scores
                WHERE judge_provider = ?
                GROUP BY experiment_id
            """, conn, params=(judge_provider,))
            scores_map = scores_agg.set_index("experiment_id").to_dict(orient="index") if not scores_agg.empty else {}
        except Exception:
            scores_map = {}

        for idx, row in exp_df.iterrows():
            eid = row["id"]
            total_q = row.get("sample_size", 103) or 103

            if eid in sum_map and sum_map[eid].get("avg_faithfulness") is not None and sum_map[eid].get("scored_questions", 0) > 0:
                s_row = sum_map[eid]
                exp_df.at[idx, "avg_faithfulness"] = s_row["avg_faithfulness"]
                exp_df.at[idx, "avg_answer_relevancy"] = s_row["avg_answer_relevancy"]
                exp_df.at[idx, "avg_context_precision"] = s_row["avg_context_precision"]
                exp_df.at[idx, "avg_context_recall"] = s_row["avg_context_recall"]
                exp_df.at[idx, "avg_overall_score"] = s_row["avg_overall_score"]
                exp_df.at[idx, "scored_questions_count"] = s_row["scored_questions"]
                exp_df.at[idx, "judge_status"] = s_row["status"]
                exp_df.at[idx, "status"] = s_row["status"]
            elif eid in scores_map and scores_map[eid].get("calc_scored", 0) > 0:
                sc_row = scores_map[eid]
                scored_cnt = sc_row["calc_scored"]
                exp_df.at[idx, "avg_faithfulness"] = sc_row["calc_f"]
                exp_df.at[idx, "avg_answer_relevancy"] = sc_row["calc_r"]
                exp_df.at[idx, "avg_context_precision"] = sc_row["calc_cp"]
                exp_df.at[idx, "avg_context_recall"] = sc_row["calc_cr"]
                exp_df.at[idx, "avg_overall_score"] = sc_row["calc_ov"]
                exp_df.at[idx, "scored_questions_count"] = scored_cnt
                st = "completed" if scored_cnt >= total_q else "partial"
                exp_df.at[idx, "judge_status"] = st
                exp_df.at[idx, "status"] = st
            else:
                exp_df.at[idx, "status"] = "unscored"
                exp_df.at[idx, "judge_status"] = "unscored"

        return exp_df



def check_system_health() -> Dict[str, Any]:
    
    health: Dict[str, Any] = {
        "ollama_running": False,
        "installed_models": [],
        "active_models_ps": [],
        "deepseek_api_ready": False,
        "chromadb_ready": False,
        "chromadb_path": None,
        "chromadb_chunks_count": 0,
        "benchmark_csv_ready": False,
        "benchmark_csv_path": None,
        "benchmark_questions_count": 0,
        "gpu_offload_status": "Unknown",
        "warnings": [],
    }
    
    try:
        res = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            health["ollama_running"] = True
            lines = res.stdout.strip().split("\n")[1:]
            for line in lines:
                parts = line.split()
                if parts:
                    health["installed_models"].append(parts[0])
    except Exception as e:
        health["warnings"].append(f"Ollama CLI error: {e}")

    try:
        res_ps = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=5)
        if res_ps.returncode == 0:
            ps_lines = res_ps.stdout.strip().split("\n")
            if len(ps_lines) > 1:
                for line in ps_lines[1:]:
                    parts = line.split()
                    if len(parts) >= 4:
                        m_name = parts[0]
                        m_size = parts[2]
                        m_proc = parts[3]  
                        health["active_models_ps"].append({"name": m_name, "size": m_size, "processor": m_proc})
                        if "100% GPU" in line:
                            health["gpu_offload_status"] = "100% Metal GPU (Optimal)"
                        elif "CPU" in line:
                            health["gpu_offload_status"] = "CPU / Partial Offload (Sub-optimal)"
                            health["warnings"].append(f"Model {m_name} is using CPU offload: {m_proc}")
    except Exception:
        pass

    from config import get_judge_config_status
    ds_ready, ds_msg, ds_meta = get_judge_config_status("deepseek_v4")
    health["deepseek_api_ready"] = ds_ready
    health["deepseek_meta"] = ds_meta
    if not ds_ready:
        health["warnings"].append("DEEPSEEK_API_KEY is not set or invalid in .env.")

    az_ready, az_msg, az_meta = get_judge_config_status("azure_gpt5_mini")
    health["azure_gpt5_mini_ready"] = az_ready
    health["azure_gpt5_mini_meta"] = az_meta
    health["azure_gpt5_mini_status"] = az_msg

    for path in DATASET_CANDIDATE_PATHS:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                health["benchmark_csv_ready"] = True
                health["benchmark_csv_path"] = path
                health["benchmark_questions_count"] = len(df)
                break
            except Exception:
                pass

    for db_p in CHROMA_CANDIDATE_PATHS:
        if os.path.exists(db_p):
            try:
                import chromadb
                client = chromadb.PersistentClient(path=db_p)
                coll = client.get_collection("apple_financials")
                cnt = coll.count()
                if cnt > 0:
                    health["chromadb_ready"] = True
                    health["chromadb_path"] = db_p
                    health["chromadb_chunks_count"] = cnt
                    break
            except Exception:
                pass

    return health



def load_benchmark_dataset(custom_path: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    
    target_path = custom_path
    if not target_path or not os.path.exists(target_path):
        for path in DATASET_CANDIDATE_PATHS:
            if os.path.exists(path):
                target_path = path
                break
                
    if not target_path or not os.path.exists(target_path):
        return None, None
        
    df = pd.read_csv(target_path)
    if "user_input" in df.columns and "question" not in df.columns:
        df["question"] = df["user_input"]
    elif "question" in df.columns and "user_input" not in df.columns:
        df["user_input"] = df["question"]
        
    if "reference" in df.columns and "ground_truth" not in df.columns:
        df["ground_truth"] = df["reference"]
    elif "ground_truth" in df.columns and "reference" not in df.columns:
        df["reference"] = df["ground_truth"]
        
    return df, target_path


def extract_reasoning_and_answer(raw_text: str) -> Tuple[str, str, bool]:
    
    if not raw_text:
        return "", "", False
        
    think_match = re.search(r'<think>(.*?)</think>', raw_text, flags=re.DOTALL)
    if think_match:
        thinking_text = think_match.group(1).strip()
        clean_answer = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
        return clean_answer, thinking_text, True
        
    unclosed_match = re.search(r'<think>(.*)', raw_text, flags=re.DOTALL)
    if unclosed_match:
        thinking_text = unclosed_match.group(1).strip()
        clean_answer = re.sub(r'<think>.*', '', raw_text, flags=re.DOTALL).strip()
        return clean_answer, thinking_text, True
        
    return raw_text.strip(), "", False


def build_prompt_with_context(system_prompt: str, context_chunks: List[str], query: str) -> str:
    
    formatted_context = ""
    for i, chunk in enumerate(context_chunks, 1):
        formatted_context += f"\n--- [DOCUMENT CHUNK {i}] ---\n{chunk.strip()}\n"
    
    prompt = f"""{system_prompt}

### PROVIDED CONTEXT:
{formatted_context}

### AUDIT QUESTION:
{query}

### AUDITOR RESPONSE:
Based on the financial reports,"""
    return prompt


try:
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import QueryBundle, NodeWithScore
except ImportError:
    class BaseRetriever:
        pass
    QueryBundle = Any  # type: ignore
    NodeWithScore = Any  # type: ignore


class FastHybridRetriever(BaseRetriever):
    
    def __init__(self, vec_retriever: Any, bm25_retriever: Any, top_k_val: int = 15):
        super().__init__()
        self.vec_retriever = vec_retriever
        self.bm25_retriever = bm25_retriever
        self.top_k_val = top_k_val

    def _retrieve(self, query_bundle: Any) -> List[Any]:
        v_nodes = self.vec_retriever.retrieve(query_bundle) if hasattr(self.vec_retriever, "retrieve") else []
        b_nodes = self.bm25_retriever.retrieve(query_bundle) if hasattr(self.bm25_retriever, "retrieve") else []
        seen = set()
        merged = []
        for n in v_nodes + b_nodes:
            nid = n.node.node_id if hasattr(n, "node") else getattr(n, "node_id", str(n))
            if nid not in seen:
                seen.add(nid)
                merged.append(n)
        return merged[:self.top_k_val]


def build_evaluation_retriever_and_engine(
    model_name: str,
    config: Dict[str, Any],
    chroma_path: Optional[str] = None,
    collection_name: str = "apple_financials"
):
    
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.ollama import Ollama
    from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
    from llama_index.vector_stores.chroma import ChromaVectorStore
    import chromadb

    if not chroma_path or not os.path.exists(chroma_path):
        for p in CHROMA_CANDIDATE_PATHS:
            if os.path.exists(p):
                chroma_path = p
                break

    if not chroma_path or not os.path.exists(chroma_path):
        raise FileNotFoundError("ChromaDB vector store path not found on disk.")

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model
    
    model_opt_ctx = MODEL_REGISTRY.get(model_name, {}).get("default_context")
    if model_opt_ctx is not None:
        ctx_win = int(model_opt_ctx)
    else:
        ctx_win = int(config.get("context_window", 8192))
    
    num_predict = int(MODEL_REGISTRY.get(model_name, {}).get("num_predict", 1024))
    temp = float(config.get("temperature", 0.0))
    
    llm = Ollama(
        model=model_name,
        request_timeout=600.0,
        temperature=temp,
        context_window=ctx_win,
        additional_kwargs={"num_predict": num_predict}
    )
    Settings.llm = llm

    db = chromadb.PersistentClient(path=chroma_path)
    coll = db.get_collection(collection_name)
    vs = ChromaVectorStore(chroma_collection=coll)
    idx = VectorStoreIndex.from_vector_store(vs, embed_model=embed_model)

    top_k = int(config.get("top_k", 3))
    initial_k = int(config.get("initial_retrieval_k", 15 if config.get("use_reranker") else top_k))
    use_bm25 = bool(config.get("use_bm25", False))
    use_reranker = bool(config.get("use_reranker", False))
    use_hyde = bool(config.get("use_hyde", False))

    base_retriever = None
    if use_bm25:
        try:
            from llama_index.core.schema import TextNode, QueryBundle
            from llama_index.retrievers.bm25 import BM25Retriever
            from llama_index.core.retrievers import BaseRetriever

            class FastHybridRetriever(BaseRetriever):
                def __init__(self, vec_retriever, bm25_retriever, top_k_val=15):
                    super().__init__()
                    self.vec_retriever = vec_retriever
                    self.bm25_retriever = bm25_retriever
                    self.top_k_val = top_k_val

                def _retrieve(self, query_bundle: QueryBundle):
                    v_nodes = self.vec_retriever.retrieve(query_bundle)
                    b_nodes = self.bm25_retriever.retrieve(query_bundle)
                    seen = set()
                    merged = []
                    for n in v_nodes + b_nodes:
                        nid = n.node.node_id if hasattr(n, "node") else getattr(n, "node_id", str(n))
                        if nid not in seen:
                            seen.add(nid)
                            merged.append(n)
                    return merged[:self.top_k_val]

            all_data = coll.get(include=["metadatas", "documents"])
            nodes = []
            docs_list = all_data.get("documents") or [] if all_data else []
            metas_list = all_data.get("metadatas") or [] if all_data else []
            if docs_list and metas_list:
                for doc, meta in zip(docs_list, metas_list):
                    if doc and str(doc).strip():
                        nodes.append(TextNode(text=str(doc), metadata=meta if isinstance(meta, dict) else {}))

            if nodes:
                bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=initial_k)
                vec_retriever = idx.as_retriever(similarity_top_k=initial_k)
                base_retriever = FastHybridRetriever(vec_retriever, bm25_retriever, top_k_val=initial_k)
        except Exception:
            pass

    if base_retriever is None:
        base_retriever = idx.as_retriever(similarity_top_k=initial_k)

    final_retriever = base_retriever
    if use_hyde:
        try:
            from llama_index.core.indices.query.query_transform import HyDEQueryTransform
            from llama_index.core.retrievers import TransformRetriever
            hyde = HyDEQueryTransform(include_original=True)
            final_retriever = TransformRetriever(base_retriever, hyde)
        except Exception:
            pass

    node_postprocessors = []
    if use_reranker:
        from llama_index.core.postprocessor import SentenceTransformerRerank
        reranker = SentenceTransformerRerank(
            model="BAAI/bge-reranker-base",
            top_n=top_k
        )
        node_postprocessors.append(reranker)

    system_prompt = (
        "You are a strict Expert Financial Auditor evaluating Apple Inc. 10-K filings.\n"
        "Formatting & Analysis Guidelines:\n"
        "- Base your answer ONLY on the provided context.\n"
        "- For calculations, show clear mathematical steps.\n"
        "- State exact reported numbers, fiscal years, and units (e.g. millions, percentages).\n"
        "- If the context does not contain sufficient information, state that clearly."
    )

    return {
        "index": idx,
        "retriever": final_retriever,
        "node_postprocessors": node_postprocessors,
        "llm": llm,
        "system_prompt": system_prompt,
        "top_k": top_k,
        "initial_k": initial_k,
    }



def evaluate_single_query(
    pipeline_bundle: Dict[str, Any],
    question: str,
    ground_truth: str,
    category: str = "General",
    q_idx: int = 0
) -> Dict[str, Any]:
    
    from llama_index.core.schema import MetadataMode, QueryBundle
    from llama_index.core.llms import ChatMessage, MessageRole

    retriever = pipeline_bundle["retriever"]
    postprocessors = pipeline_bundle.get("node_postprocessors", [])
    llm = pipeline_bundle["llm"]
    sys_prompt = pipeline_bundle["system_prompt"]

    t_start = time.time()
    
    t_ret_start = time.time()
    raw_nodes = retriever.retrieve(question)
    t_ret_end = time.time()
    retrieval_sec = t_ret_end - t_ret_start

    t_rerank_start = time.time()
    nodes = raw_nodes
    for pp in postprocessors:
        nodes = pp.postprocess_nodes(nodes, query_bundle=QueryBundle(question))
    t_rerank_end = time.time()
    rerank_sec = t_rerank_end - t_rerank_start

    contexts = [n.node.get_content(metadata_mode=MetadataMode.LLM) if hasattr(n, "node") else str(n.text) for n in nodes]
    retrieved_files = list(set([
        (n.node.metadata.get("file_name") or n.node.metadata.get("filename", "Unknown"))
        for n in nodes if hasattr(n, "node") and hasattr(n.node, "metadata")
    ]))

    context_str = "\n\n".join(contexts)

    formatted_system = f"{sys_prompt}\n---------------------\n{context_str}\n---------------------\nAnswer the question accurately based on the context above."
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=formatted_system),
        ChatMessage(role=MessageRole.USER, content=question)
    ]

    t_gen_start = time.time()
    ttft_sec = None
    full_output_chunks = []
    
    try:
        stream_resp = llm.stream_chat(messages)
        for chunk in stream_resp:
            if ttft_sec is None:
                ttft_sec = time.time() - t_gen_start
            delta = chunk.delta or ""
            full_output_chunks.append(delta)
    except Exception:
        resp = llm.chat(messages)
        full_output_chunks = [resp.message.content if hasattr(resp, "message") else str(resp)]
        if ttft_sec is None:
            ttft_sec = time.time() - t_gen_start

    t_gen_end = time.time()
    generation_sec = max(t_gen_end - t_gen_start, 0.001)
    total_duration_sec = time.time() - t_start

    raw_response_text = "".join(full_output_chunks)
    clean_answer, thinking_text, has_thinking = extract_reasoning_and_answer(raw_response_text)

    visible_tokens = max(len(clean_answer) // 4, 1)
    thinking_tokens = len(thinking_text) // 4 if thinking_text else 0
    total_tokens = visible_tokens + thinking_tokens
    
    decode_sec = max(generation_sec - (ttft_sec or 0.0), 0.05)
    speed_tps = round(total_tokens / decode_sec, 2)
    if speed_tps > 100.0:  
        speed_tps = round(total_tokens / generation_sec, 2)

    return {
        "question_idx": q_idx,
        "question": question,
        "ground_truth": ground_truth,
        "generated_answer": clean_answer,
        "raw_response": raw_response_text,
        "thinking_text": thinking_text,
        "has_thinking": has_thinking,
        "contexts": contexts,
        "retrieved_files": retrieved_files,
        "category": category,
        "ttft_sec": round(ttft_sec or 0.0, 3),
        "retrieval_duration_sec": round(retrieval_sec, 3),
        "rerank_duration_sec": round(rerank_sec, 3),
        "generation_duration_sec": round(generation_sec, 3),
        "total_duration_sec": round(total_duration_sec, 3),
        "speed_tps": speed_tps,
        "visible_tokens": visible_tokens,
        "thinking_tokens": thinking_tokens,
        "total_tokens": total_tokens,
    }




def _patch_langchain_vertexai_imports() -> None:
    try:
        import langchain_community.chat_models
        import langchain_community.llms
        if not hasattr(langchain_community.chat_models, "vertexai"):
            vx_chat = types.ModuleType("langchain_community.chat_models.vertexai")
            setattr(vx_chat, "ChatVertexAI", type("ChatVertexAI", (), {}))
            setattr(langchain_community.chat_models, "vertexai", vx_chat)
            sys.modules["langchain_community.chat_models.vertexai"] = vx_chat
        if not hasattr(langchain_community.llms, "vertexai"):
            vx_llm = types.ModuleType("langchain_community.llms.vertexai")
            setattr(vx_llm, "VertexAI", type("VertexAI", (), {}))
            setattr(langchain_community.llms, "vertexai", vx_llm)
            setattr(langchain_community.llms, "VertexAI", getattr(vx_llm, "VertexAI"))
            sys.modules["langchain_community.llms.vertexai"] = vx_llm
    except ImportError:
        pass


def run_ragas_scoring(
    eval_items: List[Dict[str, Any]],
    deepseek_api_key: str
) -> List[Dict[str, Any]]:
    
    if not eval_items:
        return []

    _patch_langchain_vertexai_imports()
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from ragas import evaluate
    from datasets import Dataset
    from langchain_openai import ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings

    if hasattr(answer_relevancy, 'strict'):
        answer_relevancy.strict = False
    if hasattr(faithfulness, 'strict'):
        faithfulness.strict = False

    from ragas.llms import LangchainLLMWrapper

    judge_llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=deepseek_api_key, # type: ignore
        base_url="https://api.deepseek.com",
        temperature=0.0,
        max_retries=3
    )
    # Explicitly bypass n>1 batching for DeepSeek API compatibility
    ragas_judge_llm = LangchainLLMWrapper(judge_llm, bypass_n=True)

    judge_embed = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"}
    )

    ragas_dataset_dict = {
        "question": [item["question"] for item in eval_items],
        "answer": [item["generated_answer"] for item in eval_items],
        "contexts": [item["contexts"] if item["contexts"] else [""] for item in eval_items],
        "ground_truth": [item["ground_truth"] for item in eval_items]
    }

    ragas_ds = Dataset.from_dict(ragas_dataset_dict)
    
    results = evaluate(
        dataset=ragas_ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_judge_llm,
        embeddings=judge_embed
    )
    
    res_df = results.to_pandas() if hasattr(results, "to_pandas") else pd.DataFrame(results) # type: ignore
    
    scored_items = []
    for i, item in enumerate(eval_items):
        item_copy = dict(item)
        item_copy["faithfulness"] = round(float(res_df.iloc[i].get("faithfulness", 0.0) or 0.0), 4)
        item_copy["answer_relevancy"] = round(float(res_df.iloc[i].get("answer_relevancy", 0.0) or 0.0), 4)
        item_copy["context_precision"] = round(float(res_df.iloc[i].get("context_precision", 0.0) or 0.0), 4)
        item_copy["context_recall"] = round(float(res_df.iloc[i].get("context_recall", 0.0) or 0.0), 4)
        scored_items.append(item_copy)

    return scored_items



class ExperimentRunner:
    def __init__(self, db_path: str = DEFAULT_EXP_DB):
        self.db_path = db_path
        self.checkpoints_dir = CHECKPOINTS_DIR
        self.completed_dir = COMPLETED_DIR
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.completed_dir, exist_ok=True)
        init_experiments_db(self.db_path)

    def _get_checkpoint_filepath(self, exp_name: str) -> str:
        safe_name = exp_name.replace(":", "_").replace("/", "_").replace(" ", "_")
        return os.path.join(self.checkpoints_dir, f"{safe_name}.json")

    def save_checkpoint(
        self,
        exp_name: str,
        model_name: str,
        config: Dict[str, Any],
        completed_items: List[Dict[str, Any]],
        total_count: int,
        error_msg: Optional[str] = None
    ):
        
        valid_items = [
            it for it in completed_items
            if it.get("generated_answer") and not it.get("is_error")
        ]
        
        data_payload = {
            "experiment_name": exp_name,
            "model_name": model_name,
            "config": config,
            "completed_count": len(valid_items),
            "total_count": total_count,
            "last_updated": datetime.now().isoformat(),
            "status": "error" if error_msg else "in_progress",
            "error_message": error_msg,
            "completed_items": valid_items
        }

        try:
            target_path = self._get_checkpoint_filepath(exp_name)
            tmp_path = target_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data_payload, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, target_path)
        except Exception:
            pass

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO checkpoints (experiment_name, model_name, config_json, completed_count, total_count, partial_data_json, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    exp_name, model_name, json.dumps(config), len(valid_items), total_count, json.dumps(valid_items)
                ))
        except Exception:
            pass

    def load_checkpoint(self, exp_name: str) -> Optional[Dict[str, Any]]:
        
        target_path = self._get_checkpoint_filepath(exp_name)
        if os.path.exists(target_path):
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    items = data.get("completed_items", [])
                    valid_items = [it for it in items if it.get("generated_answer") and not it.get("is_error")]
                    data["completed_items"] = valid_items
                    data["completed_count"] = len(valid_items)
                    return data
            except Exception:
                pass

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute("SELECT * FROM checkpoints WHERE experiment_name = ?", (exp_name,)).fetchone()
                if row:
                    items = json.loads(row["partial_data_json"])
                    valid_items = [it for it in items if it.get("generated_answer") and not it.get("is_error")]
                    return {
                        "experiment_name": row["experiment_name"],
                        "model_name": row["model_name"],
                        "config": json.loads(row["config_json"]),
                        "completed_count": len(valid_items),
                        "total_count": row["total_count"],
                        "completed_items": valid_items,
                    }
        except Exception:
            pass
        return None

    def delete_checkpoint(self, exp_name: str):
        
        try:
            target_path = self._get_checkpoint_filepath(exp_name)
            if os.path.exists(target_path):
                safe_name = exp_name.replace(":", "_").replace("/", "_").replace(" ", "_")
                archive_path = os.path.join(self.completed_dir, f"{safe_name}.json")
                os.replace(target_path, archive_path)
        except Exception:
            pass

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM checkpoints WHERE experiment_name = ?", (exp_name,))
        except Exception:
            pass

    def get_matrix_status(self, matrix_type: str = "core", sample_size: int = 103) -> List[Dict[str, Any]]:
        
        matrix_list = CORE_EXPERIMENT_MATRIX if matrix_type == "core" else ABLATION_EXPERIMENT_MATRIX
        status_list = []

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            for item in matrix_list:
                name = item["name"]
                m_name = item["model"]
                cfg_id = item["config"]
                cfg = CONFIG_PRESETS[cfg_id]

                db_row = conn.execute(
                    "SELECT id, avg_faithfulness, avg_speed_tps, sample_size, timestamp FROM experiments WHERE experiment_name = ?",
                    (name,)
                ).fetchone()

                if db_row and db_row["sample_size"] >= sample_size:
                    status_list.append({
                        "run_num": item.get("run_num", 0),
                        "name": name,
                        "model": m_name,
                        "config_id": cfg_id,
                        "config": cfg,
                        "status": "completed",
                        "status_label": "🟢 Completed",
                        "completed_count": sample_size,
                        "total_count": sample_size,
                        "resume_from": None,
                        "db_id": db_row["id"],
                        "faithfulness": db_row["avg_faithfulness"],
                        "speed_tps": db_row["avg_speed_tps"],
                        "last_updated": db_row["timestamp"]
                    })
                    continue

                cp = self.load_checkpoint(name)
                if cp and cp.get("completed_count", 0) > 0:
                    c_count = cp["completed_count"]
                    status_list.append({
                        "run_num": item.get("run_num", 0),
                        "name": name,
                        "model": m_name,
                        "config_id": cfg_id,
                        "config": cfg,
                        "status": "in_progress",
                        "status_label": f"🟡 Paused ({c_count}/{sample_size} Qs)",
                        "completed_count": c_count,
                        "total_count": sample_size,
                        "resume_from": c_count + 1,
                        "db_id": None,
                        "faithfulness": None,
                        "speed_tps": None,
                        "last_updated": cp.get("last_updated")
                    })
                else:
                    status_list.append({
                        "run_num": item.get("run_num", 0),
                        "name": name,
                        "model": m_name,
                        "config_id": cfg_id,
                        "config": cfg,
                        "status": "pending",
                        "status_label": f"⚪ Pending (0/{sample_size} Qs)",
                        "completed_count": 0,
                        "total_count": sample_size,
                        "resume_from": 1,
                        "db_id": None,
                        "faithfulness": None,
                        "speed_tps": None,
                        "last_updated": None
                    })

        return status_list

    def save_final_experiment(
        self,
        exp_name: str,
        model_name: str,
        config: Dict[str, Any],
        items: List[Dict[str, Any]],
        status: str = "completed",
        notes: str = ""
    ) -> int:
        
        df_items = pd.DataFrame(items)
        
        has_f = "faithfulness" in df_items.columns and bool(df_items["faithfulness"].notna().any())
        has_r = "answer_relevancy" in df_items.columns and bool(df_items["answer_relevancy"].notna().any())
        has_cp = "context_precision" in df_items.columns and bool(df_items["context_precision"].notna().any())
        has_cr = "context_recall" in df_items.columns and bool(df_items["context_recall"].notna().any())

        avg_f = float(df_items["faithfulness"].mean()) if has_f else None
        avg_r = float(df_items["answer_relevancy"].mean()) if has_r else None
        avg_cp = float(df_items["context_precision"].mean()) if has_cp else None
        avg_cr = float(df_items["context_recall"].mean()) if has_cr else None

        valid_scores = [s for s in [avg_f, avg_r, avg_cp, avg_cr] if s is not None]
        avg_overall = float(np.mean(valid_scores)) if valid_scores else None

        avg_ttft = float(df_items["ttft_sec"].mean() if "ttft_sec" in df_items else 0.0)
        avg_ret = float(df_items["retrieval_duration_sec"].mean() if "retrieval_duration_sec" in df_items else 0.0)
        avg_rer = float(df_items["rerank_duration_sec"].mean() if "rerank_duration_sec" in df_items else 0.0)
        avg_gen = float(df_items["generation_duration_sec"].mean() if "generation_duration_sec" in df_items else 0.0)
        avg_tot = float(df_items["total_duration_sec"].mean() if "total_duration_sec" in df_items else 0.0)
        avg_speed = float(df_items["speed_tps"].mean() if "speed_tps" in df_items else 0.0)
        avg_vis_tok = float(df_items["visible_tokens"].mean() if "visible_tokens" in df_items else 0.0)
        avg_thk_tok = float(df_items["thinking_tokens"].mean() if "thinking_tokens" in df_items else 0.0)
        avg_all_tok = float(df_items["total_tokens"].mean() if "total_tokens" in df_items else 0.0)

        m_meta = MODEL_REGISTRY.get(model_name, {})
        params = m_meta.get("params", "Unknown")
        quant = m_meta.get("quant", "Q4_K_M")

        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('''
                INSERT OR REPLACE INTO experiments (
                    experiment_name, model_name, model_params, model_quant, config_id, config_name,
                    top_k, initial_retrieval_k, use_bm25, use_reranker, use_hyde,
                    context_window, temperature, sample_size,
                    avg_faithfulness, avg_answer_relevancy, avg_context_precision, avg_context_recall, avg_overall_score,
                    avg_ttft_sec, avg_retrieval_duration_sec, avg_rerank_duration_sec, avg_generation_duration_sec,
                    avg_total_duration_sec, avg_speed_tps, avg_visible_tokens, avg_thinking_tokens, avg_total_tokens,
                    status, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                exp_name, model_name, params, quant, config.get("id", "Custom"), config.get("name", "Custom"),
                config.get("top_k", 3), config.get("initial_retrieval_k", 3),
                config.get("use_bm25", False), config.get("use_reranker", False), config.get("use_hyde", False),
                config.get("context_window", 8192), config.get("temperature", 0.0), len(items),
                avg_f, avg_r, avg_cp, avg_cr, avg_overall,
                avg_ttft, avg_ret, avg_rer, avg_gen,
                avg_tot, avg_speed, avg_vis_tok, avg_thk_tok, avg_all_tok,
                status, notes
            ))
            exp_id = int(c.lastrowid or 0)

            c.execute("DELETE FROM detailed_results WHERE experiment_id = ?", (exp_id,))

            for item in items:
                c.execute('''
                    INSERT INTO detailed_results (
                        experiment_id, question_idx, question, ground_truth, generated_answer, thinking_text, has_thinking,
                        contexts, retrieved_files, category, faithfulness, answer_relevancy, context_precision, context_recall,
                        ttft_sec, retrieval_duration_sec, rerank_duration_sec, generation_duration_sec, total_duration_sec,
                        speed_tps, visible_tokens, thinking_tokens, total_tokens
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    exp_id, item.get("question_idx", 0), item.get("question", ""), item.get("ground_truth", ""),
                    item.get("generated_answer", ""), item.get("thinking_text", ""), item.get("has_thinking", False),
                    json.dumps(item.get("contexts", [])), json.dumps(item.get("retrieved_files", [])),
                    item.get("category", "General"),
                    item.get("faithfulness"), item.get("answer_relevancy"),
                    item.get("context_precision"), item.get("context_recall"),
                    item.get("ttft_sec", 0.0), item.get("retrieval_duration_sec", 0.0), item.get("rerank_duration_sec", 0.0),
                    item.get("generation_duration_sec", 0.0), item.get("total_duration_sec", 0.0),
                    item.get("speed_tps", 0.0), item.get("visible_tokens", 0), item.get("thinking_tokens", 0),
                    item.get("total_tokens", 0)
                ))
            conn.commit()

        self.delete_checkpoint(exp_name)
        return exp_id

    def score_experiment(
        self,
        exp_id: int,
        judge_provider_id: str = "deepseek_v4",
        deepseek_api_key: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        azure_api_version: Optional[str] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        single_question_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        
        import judge_providers
        if judge_provider_id == "azure_gpt5_mini":
            provider = judge_providers.AzureGPT5MiniJudge(
                api_key=azure_api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                endpoint=azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
                deployment=azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                api_version=azure_api_version or os.getenv("AZURE_OPENAI_API_VERSION")
            )
        elif judge_provider_id == "deepseek_v4":
            provider = judge_providers.DeepSeekJudge(
                api_key=deepseek_api_key or os.getenv("DEEPSEEK_API_KEY")
            )
        else:
            provider = judge_providers.get_judge_provider(judge_provider_id)

        is_ready, err_msg, meta = provider.is_configured()
        if not is_ready:
            raise ValueError(f"Judge Provider '{provider.display_name}' is not properly configured: {err_msg}")

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            exp_row = conn.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,)).fetchone()
            if not exp_row:
                raise ValueError(f"Experiment #{exp_id} not found in database.")

            det_rows = conn.execute(
                "SELECT * FROM detailed_results WHERE experiment_id = ? ORDER BY question_idx ASC",
                (exp_id,)
            ).fetchall()

            all_items = []
            for r in det_rows:
                all_items.append({
                    "id": r["id"],
                    "question_idx": r["question_idx"],
                    "question": r["question"],
                    "ground_truth": r["ground_truth"],
                    "generated_answer": r["generated_answer"],
                    "contexts": json.loads(r["contexts"]) if r["contexts"] else [],
                    "category": r["category"],
                    "ttft_sec": r["ttft_sec"],
                    "retrieval_duration_sec": r["retrieval_duration_sec"],
                    "rerank_duration_sec": r["rerank_duration_sec"],
                    "generation_duration_sec": r["generation_duration_sec"],
                    "total_duration_sec": r["total_duration_sec"],
                    "speed_tps": r["speed_tps"],
                    "visible_tokens": r["visible_tokens"],
                    "thinking_tokens": r["thinking_tokens"],
                    "total_tokens": r["total_tokens"]
                })

            scored_rows = conn.execute(
                "SELECT question_idx FROM judge_scores WHERE experiment_id = ? AND judge_provider = ? AND status = 'completed'",
                (exp_id, provider.provider_id)
            ).fetchall()
            already_scored_indices = set(r["question_idx"] for r in scored_rows)

        if single_question_idx is not None:
            target_items = [it for it in all_items if it["question_idx"] == single_question_idx]
        else:
            target_items = [it for it in all_items if it["question_idx"] not in already_scored_indices]

        total_questions = len(all_items)
        target_count = len(target_items)

        if progress_callback:
            progress_callback({
                "stage": "starting",
                "exp_id": exp_id,
                "exp_name": exp_row["experiment_name"],
                "judge_provider": provider.provider_id,
                "judge_name": provider.display_name,
                "total": total_questions,
                "pending": target_count,
                "already_scored": len(already_scored_indices),
                "status_text": f"Scoring {target_count} pending questions with {provider.display_name}..."
            })

        for i, item in enumerate(target_items):
            q_idx = item["question_idx"]
            
            if progress_callback:
                progress_callback({
                    "stage": "judging",
                    "exp_id": exp_id,
                    "exp_name": exp_row["experiment_name"],
                    "current": len(already_scored_indices) + i + 1,
                    "total": total_questions,
                    "question_idx": q_idx,
                    "status_text": f"Evaluating Q#{q_idx+1} ({item.get('category', 'General')}) with {provider.display_name}..."
                })

            score_res = provider.score_single_item(item)

            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO judge_scores 
                    (experiment_id, question_idx, judge_provider, judge_model, faithfulness, answer_relevancy, context_precision, context_recall, overall_score, scored_at, status, error_message, evaluation_time_sec)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?)
                ''', (
                    exp_id,
                    q_idx,
                    provider.provider_id,
                    provider.model_name,
                    score_res.get("faithfulness", 0.0),
                    score_res.get("answer_relevancy", 0.0),
                    score_res.get("context_precision", 0.0),
                    score_res.get("context_recall", 0.0),
                    score_res.get("overall_score", 0.0),
                    score_res.get("status", "completed"),
                    score_res.get("error_message"),
                    score_res.get("evaluation_time_sec", 0.0)
                ))

                if provider.provider_id == "deepseek_v4" and score_res.get("status") == "completed":
                    conn.execute('''
                        UPDATE detailed_results
                        SET faithfulness = ?, answer_relevancy = ?, context_precision = ?, context_recall = ?
                        WHERE experiment_id = ? AND question_idx = ?
                    ''', (
                        score_res.get("faithfulness", 0.0),
                        score_res.get("answer_relevancy", 0.0),
                        score_res.get("context_precision", 0.0),
                        score_res.get("context_recall", 0.0),
                        exp_id,
                        q_idx
                    ))
                conn.commit()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            summary_row = conn.execute('''
                SELECT 
                    AVG(faithfulness) as avg_f,
                    AVG(answer_relevancy) as avg_r,
                    AVG(context_precision) as avg_cp,
                    AVG(context_recall) as avg_cr,
                    AVG(overall_score) as avg_overall,
                    COUNT(*) as completed_count
                FROM judge_scores
                WHERE experiment_id = ? AND judge_provider = ? AND status = 'completed'
            ''', (exp_id, provider.provider_id)).fetchone()

            avg_f = float(summary_row["avg_f"]) if summary_row and summary_row["avg_f"] is not None else 0.0
            avg_r = float(summary_row["avg_r"]) if summary_row and summary_row["avg_r"] is not None else 0.0
            avg_cp = float(summary_row["avg_cp"]) if summary_row and summary_row["avg_cp"] is not None else 0.0
            avg_cr = float(summary_row["avg_cr"]) if summary_row and summary_row["avg_cr"] is not None else 0.0
            avg_overall = float(summary_row["avg_overall"]) if summary_row and summary_row["avg_overall"] is not None else 0.0
            completed_count = int(summary_row["completed_count"]) if summary_row else 0

            status_str = "completed" if completed_count >= total_questions else ("partial" if completed_count > 0 else "pending")

            conn.execute('''
                INSERT OR REPLACE INTO judge_experiment_summaries
                (experiment_id, judge_provider, judge_model, avg_faithfulness, avg_answer_relevancy, avg_context_precision, avg_context_recall, avg_overall_score, scored_questions, total_questions, status, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                exp_id,
                provider.provider_id,
                provider.model_name,
                round(avg_f, 4),
                round(avg_r, 4),
                round(avg_cp, 4),
                round(avg_cr, 4),
                round(avg_overall, 4),
                completed_count,
                total_questions,
                status_str
            ))

            if provider.provider_id == "deepseek_v4":
                conn.execute('''
                    UPDATE experiments
                    SET avg_faithfulness = ?, avg_answer_relevancy = ?, avg_context_precision = ?, avg_context_recall = ?,
                        avg_overall_score = ?, status = 'completed'
                    WHERE id = ?
                ''', (round(avg_f, 4), round(avg_r, 4), round(avg_cp, 4), round(avg_cr, 4), round(avg_overall, 4), exp_id))
            conn.commit()

        if progress_callback:
            progress_callback({
                "stage": "completed",
                "exp_id": exp_id,
                "exp_name": exp_row["experiment_name"],
                "completed": completed_count,
                "total": total_questions,
                "status_text": f"Finished scoring {completed_count}/{total_questions} with {provider.display_name}."
            })

        return {
            "exp_id": exp_id,
            "judge_provider": provider.provider_id,
            "judge_name": provider.display_name,
            "avg_faithfulness": round(avg_f, 4),
            "avg_answer_relevancy": round(avg_r, 4),
            "avg_context_precision": round(avg_cp, 4),
            "avg_context_recall": round(avg_cr, 4),
            "avg_overall_score": round(avg_overall, 4),
            "scored_questions": completed_count,
            "total_questions": total_questions,
            "status": status_str
        }

    def score_all_unscored_experiments(
        self,
        deepseek_api_key: Optional[str] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> List[Dict[str, Any]]:
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT id, experiment_name FROM experiments WHERE avg_faithfulness IS NULL OR status = 'generated_unscored' ORDER BY id ASC").fetchall()

        results = []
        for r in rows:
            res = self.score_experiment(r["id"], deepseek_api_key, progress_callback)
            results.append(res)
        return results

    def execute_evaluation(
        self,
        exp_name: str,
        model_name: str,
        config: Dict[str, Any],
        dataset_df: pd.DataFrame,
        sample_size: int,
        deepseek_api_key: Optional[str] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
        resume: bool = True,
        run_judge_after_generation: bool = False
    ) -> Dict[str, Any]:
        
        sample_subset = dataset_df.head(int(sample_size)).copy()
        total_questions = len(sample_subset)

        completed_items: List[Dict[str, Any]] = []
        start_idx = 0

        if resume:
            with sqlite3.connect(self.db_path) as conn:
                existing = conn.execute(
                    "SELECT id, status FROM experiments WHERE experiment_name = ? AND sample_size >= ?",
                    (exp_name, total_questions)
                ).fetchone()
                if existing:
                    if progress_callback:
                        progress_callback({
                            "stage": "completed",
                            "current": total_questions,
                            "total": total_questions,
                            "percent": 1.0,
                            "exp_name": exp_name,
                            "model_name": model_name,
                            "status_text": f"Run '{exp_name}' is already recorded in database (Exp ID #{existing[0]}, Status: {existing[1]}). Skipping..."
                        })
                    return {"experiment_id": existing[0], "status": existing[1]}

        if resume:
            cp = self.load_checkpoint(exp_name)
            if cp and len(cp["completed_items"]) > 0:
                completed_items = cp["completed_items"]
                start_idx = len(completed_items)

        pipeline_bundle = build_evaluation_retriever_and_engine(
            model_name=model_name,
            config=config
        )

        q_col = "user_input" if "user_input" in sample_subset.columns else "question"
        gt_col = "reference" if "reference" in sample_subset.columns else "ground_truth"
        cat_col = "synthesizer_name" if "synthesizer_name" in sample_subset.columns else "category"

        for i in range(start_idx, total_questions):
            if is_cancelled and is_cancelled():
                self.save_checkpoint(exp_name, model_name, config, completed_items, total_questions)
                return {
                    "status": "cancelled",
                    "completed_count": len(completed_items),
                    "total_count": total_questions,
                    "items": completed_items
                }

            row = sample_subset.iloc[i]
            q_text = str(row[q_col])
            gt_text = str(row[gt_col])
            cat_text = str(row[cat_col]) if cat_col in row and pd.notna(row[cat_col]) else "General"

            try:
                item_result = evaluate_single_query(
                    pipeline_bundle=pipeline_bundle,
                    question=q_text,
                    ground_truth=gt_text,
                    category=cat_text,
                    q_idx=i + 1
                )
                completed_items.append(item_result)
                self.save_checkpoint(exp_name, model_name, config, completed_items, total_questions)
            except Exception as e:
                self.save_checkpoint(exp_name, model_name, config, completed_items, total_questions, error_msg=str(e))
                raise e

            if progress_callback:
                progress_callback({
                    "stage": "generating",
                    "current": i + 1,
                    "total": total_questions,
                    "percent": (i + 1) / total_questions,
                    "latest_item": item_result,
                    "exp_name": exp_name,
                    "model_name": model_name
                })

        if run_judge_after_generation and deepseek_api_key:
            if progress_callback:
                progress_callback({
                    "stage": "judging",
                    "current": total_questions,
                    "total": total_questions,
                    "percent": 0.9,
                    "exp_name": exp_name,
                    "model_name": model_name,
                    "status_text": "Scoring Ragas metrics with DeepSeek Flash V4 Judge..."
                })

            try:
                scored_items = run_ragas_scoring(completed_items, deepseek_api_key)
                exp_status = "completed"
            except Exception as e:
                scored_items = completed_items
                exp_status = "generated_unscored"
        else:
            scored_items = completed_items
            exp_status = "generated_unscored"

        exp_id = self.save_final_experiment(
            exp_name=exp_name,
            model_name=model_name,
            config=config,
            items=scored_items,
            status=exp_status
        )

        if progress_callback:
            progress_callback({
                "stage": "completed",
                "current": total_questions,
                "total": total_questions,
                "percent": 1.0,
                "exp_id": exp_id,
                "exp_name": exp_name,
                "model_name": model_name,
                "status": exp_status,
                "status_text": f"Run '{exp_name}' generation finished & saved (Exp ID #{exp_id}, Status: {exp_status})."
            })

        return {
            "experiment_id": exp_id,
            "status": exp_status,
            "completed_count": len(scored_items),
            "items": scored_items
        }
