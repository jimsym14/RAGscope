import json
import os
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import streamlit as st
from config import DATA_DIR, DB_PATH, EXP_DB_PATH


def init_experiments_db() -> None:
    import evaluation_engine as ee
    ee.init_experiments_db(EXP_DB_PATH)


def init_db() -> None:
    
    os.makedirs(DATA_DIR, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                title TEXT,
                ts TEXT,
                messages TEXT
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT,
                timestamp TEXT,
                model_name TEXT,
                query_text TEXT,
                response_text TEXT,
                has_thinking BOOLEAN,
                thinking_text TEXT,
                ttft_sec REAL,
                total_duration_sec REAL,
                speed_tps REAL,
                total_tokens INTEGER,
                advanced_rag_enabled BOOLEAN,
                use_reranker BOOLEAN,
                use_bm25 BOOLEAN,
                use_hyde BOOLEAN,
                chunk_size INTEGER,
                top_k INTEGER,
                context_window INTEGER,
                retrieved_files TEXT,
                retrieved_chunks TEXT,
                ragas_faithfulness REAL,
                ragas_answer_relevancy REAL,
                ragas_context_precision REAL,
                ragas_context_recall REAL,
                ragas_overall REAL
            )
        ''')


def log_analytics_to_db(data: Dict[str, Any]) -> None:
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            columns = ', '.join(data.keys())
            placeholders = ', '.join(['?' for _ in data])
            query = f"INSERT INTO analytics ({columns}) VALUES ({placeholders})"
            conn.execute(query, tuple(data.values()))
    except Exception as e:
        print(f"Error logging analytics: {e}")


def save_chat_to_db(chat: Dict[str, Any]) -> None:
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO chats (id, title, ts, messages) VALUES (?, ?, ?, ?)",
            (chat["id"], chat["title"], str(chat["ts"]), json.dumps(chat["messages"]))
        )


def delete_chat_from_db(chat_id: str) -> None:
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))


def load_chats_from_db() -> List[Dict[str, Any]]:
    
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT id, title, ts, messages FROM chats ORDER BY ts ASC")
        chats = []
        for row in c.fetchall():
            try:
                ts = datetime.fromisoformat(row[2])
            except Exception:
                try:
                    ts = datetime.strptime(row[2], "%Y-%m-%d %H:%M:%S.%f")
                except Exception:
                    ts = datetime.now()
            chats.append({
                "id": row[0],
                "title": row[1],
                "ts": ts,
                "messages": json.loads(row[3])
            })
        return chats


def get_active_chat() -> Optional[Dict[str, Any]]:
    
    chats = getattr(st.session_state, "chats", [])
    active_id = getattr(st.session_state, "active_id", None)
    for c in chats:
        if c["id"] == active_id:
            return c
    return None


def create_chat(first_msg: str) -> Dict[str, Any]:
    
    cid = str(uuid.uuid4())[:8]
    title = first_msg
    chat = {"id": cid, "title": title, "messages": [], "ts": datetime.now()}
    if "chats" not in st.session_state:
        st.session_state.chats = []
    st.session_state.chats.append(chat)
    st.session_state.active_id = cid
    save_chat_to_db(chat)
    return chat



def delete_experiment_from_db(experiment_id: int, db_path: str = EXP_DB_PATH) -> bool:
    
    if not os.path.exists(db_path):
        return False
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("DELETE FROM detailed_results WHERE experiment_id = ?", (experiment_id,))
            conn.execute("DELETE FROM judge_scores WHERE experiment_id = ?", (experiment_id,))
            conn.execute("DELETE FROM judge_experiment_summaries WHERE experiment_id = ?", (experiment_id,))
            conn.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
            conn.commit()
            return True
    except Exception as e:
        print(f"Error deleting experiment #{experiment_id}: {e}")
        return False


def get_reasoning_eval_data(judge_provider: str = "deepseek_v4", db_path: str = EXP_DB_PATH) -> pd.DataFrame:
    
    if not os.path.exists(db_path):
        return pd.DataFrame()
    query = """
    SELECT d.id, d.experiment_id, d.question_idx, d.question, d.ground_truth, d.generated_answer,
           d.thinking_text, d.has_thinking, d.contexts, d.retrieved_files, d.category,
           j.faithfulness, j.answer_relevancy, j.context_precision, j.context_recall,
           d.ttft_sec, d.retrieval_duration_sec, d.rerank_duration_sec, d.generation_duration_sec,
           d.total_duration_sec, d.speed_tps, d.visible_tokens, d.thinking_tokens, d.total_tokens,
           e.model_name, e.experiment_name, e.config_id, e.config_name
    FROM detailed_results d
    JOIN experiments e ON d.experiment_id = e.id
    LEFT JOIN judge_scores j 
      ON d.experiment_id = j.experiment_id 
     AND d.question_idx = j.question_idx 
     AND j.judge_provider = ?
    WHERE e.model_name LIKE '%deepseek%' OR e.model_name LIKE '%r1%' OR d.has_thinking = 1 OR d.thinking_tokens > 0
    """
    try:
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql_query(query, conn, params=(judge_provider,))
    except Exception as e:
        print(f"Error loading reasoning data: {e}")
        return pd.DataFrame()


def get_dual_judge_comparison_data(db_path: str = EXP_DB_PATH) -> pd.DataFrame:
    
    if not os.path.exists(db_path):
        return pd.DataFrame()
    query = """
    SELECT 
        e.id as experiment_id,
        e.experiment_name,
        e.model_name,
        e.config_id,
        j1.question_idx,
        d.question,
        d.category,
        j1.faithfulness as ds_faithfulness,
        j1.answer_relevancy as ds_relevancy,
        j1.context_precision as ds_precision,
        j1.context_recall as ds_recall,
        j1.overall_score as ds_overall,
        j2.faithfulness as gpt_faithfulness,
        j2.answer_relevancy as gpt_relevancy,
        j2.context_precision as gpt_precision,
        j2.context_recall as gpt_recall,
        j2.overall_score as gpt_overall
    FROM judge_scores j1
    JOIN judge_scores j2 
      ON j1.experiment_id = j2.experiment_id 
     AND j1.question_idx = j2.question_idx
     AND j1.judge_provider = 'deepseek_v4'
     AND j2.judge_provider = 'azure_gpt5_mini'
    JOIN experiments e ON j1.experiment_id = e.id
    LEFT JOIN detailed_results d ON j1.experiment_id = d.experiment_id AND j1.question_idx = d.question_idx
    WHERE j1.status = 'completed' AND j2.status = 'completed'
      AND j1.faithfulness IS NOT NULL AND j2.faithfulness IS NOT NULL
      AND j1.overall_score IS NOT NULL AND j2.overall_score IS NOT NULL
      AND j1.overall_score >= 0.0 AND j2.overall_score >= 0.0
    ORDER BY e.id, j1.question_idx
    """
    try:
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error loading dual judge comparison data: {e}")
        return pd.DataFrame()


def get_dual_judge_pareto_data(db_path: str = EXP_DB_PATH) -> pd.DataFrame:
    
    if not os.path.exists(db_path):
        return pd.DataFrame()
    query = """
    SELECT 
        e.id as experiment_id,
        e.experiment_name,
        e.model_name,
        e.config_id,
        ROUND(AVG(d.speed_tps), 1) as speed_tps,
        ROUND(AVG(d.total_duration_sec), 1) as latency_s,
        ROUND(AVG(d.ttft_sec), 2) as ttft_s,
        ROUND(j_ds.avg_faithfulness, 3) as ds_faithfulness,
        ROUND(j_ds.avg_answer_relevancy, 3) as ds_relevancy,
        ROUND(j_ds.avg_context_precision, 3) as ds_precision,
        ROUND(j_ds.avg_context_recall, 3) as ds_recall,
        ROUND(j_ds.avg_overall_score, 3) as ds_overall,
        ROUND(j_gpt.avg_faithfulness, 3) as gpt_faithfulness,
        ROUND(j_gpt.avg_answer_relevancy, 3) as gpt_relevancy,
        ROUND(j_gpt.avg_context_precision, 3) as gpt_precision,
        ROUND(j_gpt.avg_context_recall, 3) as gpt_recall,
        ROUND(j_gpt.avg_overall_score, 3) as gpt_overall
    FROM experiments e
    LEFT JOIN detailed_results d ON e.id = d.experiment_id
    JOIN judge_experiment_summaries j_ds 
      ON e.id = j_ds.experiment_id 
     AND j_ds.judge_provider = 'deepseek_v4' 
     AND j_ds.avg_faithfulness IS NOT NULL 
     AND j_ds.avg_faithfulness > 0.05
    JOIN judge_experiment_summaries j_gpt 
      ON e.id = j_gpt.experiment_id 
     AND j_gpt.judge_provider = 'azure_gpt5_mini' 
     AND j_gpt.avg_faithfulness IS NOT NULL 
     AND j_gpt.avg_faithfulness > 0.05
    GROUP BY e.id
    ORDER BY e.id ASC
    """
    try:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql(query, conn)
            if not df.empty:
                df = df.sort_values(by=['model_name', 'config_id', 'experiment_id']).drop_duplicates(subset=['model_name', 'config_id'], keep='last')
            return df
    except Exception as e:
        print(f"Error loading dual judge pareto data: {e}")
        return pd.DataFrame()


def get_case_study_questions(db_path: str = EXP_DB_PATH) -> pd.DataFrame:
    
    if not os.path.exists(db_path):
        return pd.DataFrame()
    try:
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql("SELECT DISTINCT question_idx, question, category FROM detailed_results ORDER BY question_idx", conn)
    except Exception as e:
        print(f"Error loading case study questions: {e}")
        return pd.DataFrame()


def get_case_study_answers(question_idx: int, judge_provider: str = "deepseek_v4", db_path: str = EXP_DB_PATH) -> pd.DataFrame:
    
    if not os.path.exists(db_path):
        return pd.DataFrame()
    query = """
    SELECT d.id, d.experiment_id, e.model_name, e.config_id, d.generated_answer,
           d.thinking_text, d.visible_tokens, d.thinking_tokens, d.total_tokens,
           d.speed_tps, d.ttft_sec, d.total_duration_sec, d.contexts, d.ground_truth,
           j.faithfulness, j.answer_relevancy, j.context_precision, j.context_recall, j.overall_score
    FROM detailed_results d
    JOIN experiments e ON d.experiment_id = e.id
    LEFT JOIN judge_scores j ON d.experiment_id = j.experiment_id AND d.question_idx = j.question_idx AND j.judge_provider = ?
    WHERE d.question_idx = ?
    ORDER BY e.model_name, e.config_id
    """
    try:
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql(query, conn, params=(judge_provider, question_idx))
    except Exception as e:
        print(f"Error loading case study answers: {e}")
        return pd.DataFrame()


def get_experiments_list(db_path: str = EXP_DB_PATH) -> pd.DataFrame:
    
    if not os.path.exists(db_path):
        return pd.DataFrame()
    try:
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql("SELECT id, experiment_name, model_name, config_name FROM experiments ORDER BY timestamp DESC", conn)
    except Exception as e:
        print(f"Error loading experiments list: {e}")
        return pd.DataFrame()


def get_experiment_detailed_rows(experiment_id: int, judge_provider: str = "deepseek_v4", db_path: str = EXP_DB_PATH) -> pd.DataFrame:
    
    if not os.path.exists(db_path):
        return pd.DataFrame()
    query = """
    SELECT d.id, d.experiment_id, d.question_idx, d.question, d.ground_truth, d.generated_answer,
           d.thinking_text, d.has_thinking, d.contexts, d.retrieved_files, d.category,
           j.faithfulness, j.answer_relevancy, j.context_precision, j.context_recall,
           d.ttft_sec, d.retrieval_duration_sec, d.rerank_duration_sec, d.generation_duration_sec,
           d.total_duration_sec, d.speed_tps, d.visible_tokens, d.thinking_tokens, d.total_tokens
    FROM detailed_results d
    LEFT JOIN judge_scores j 
      ON d.experiment_id = j.experiment_id 
     AND d.question_idx = j.question_idx 
     AND j.judge_provider = ?
    WHERE d.experiment_id = ?
    ORDER BY d.question_idx ASC
    """
    try:
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql(query, conn, params=(judge_provider, experiment_id))
    except Exception as e:
        print(f"Error loading experiment detailed rows: {e}")
        return pd.DataFrame()


def get_adversarial_category_breakdown(db_path: str = EXP_DB_PATH) -> pd.DataFrame:
    
    if not os.path.exists(db_path):
        return pd.DataFrame()
    query = """
    SELECT e.experiment_name, a.category,
           100.0 * SUM(CASE WHEN a.is_refusal = 1 THEN 1 ELSE 0 END) / COUNT(*) as refusal_rate
    FROM adversarial_results a
    JOIN experiments e ON a.exp_id = e.id
    GROUP BY e.experiment_name, a.category
    """
    try:
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql(query, conn)
    except Exception as e:
        print(f"Error loading adversarial category breakdown: {e}")
        return pd.DataFrame()


def get_adversarial_results_for_experiment(exp_id: int, db_path: str = EXP_DB_PATH) -> pd.DataFrame:
    
    if not os.path.exists(db_path):
        return pd.DataFrame()
    try:
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql(
                "SELECT * FROM adversarial_results WHERE exp_id = ? ORDER BY question_idx ASC",
                conn,
                params=(exp_id,)
            )
    except Exception as e:
        print(f"Error loading adversarial results: {e}")
        return pd.DataFrame()


def get_operations_summary(db_path: str = EXP_DB_PATH) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    
    if not os.path.exists(db_path):
        return pd.DataFrame(), {}, {}
    try:
        with sqlite3.connect(db_path) as conn:
            exp_df = pd.read_sql_query("SELECT * FROM experiments ORDER BY timestamp DESC", conn)
            try:
                ds_summaries = pd.read_sql_query(
                    "SELECT * FROM judge_experiment_summaries WHERE judge_provider = 'deepseek_v4'", conn
                ).set_index("experiment_id").to_dict("index")
            except Exception:
                ds_summaries = {}
            try:
                gpt_summaries = pd.read_sql_query(
                    "SELECT * FROM judge_experiment_summaries WHERE judge_provider = 'azure_gpt5_mini'", conn
                ).set_index("experiment_id").to_dict("index")
            except Exception:
                gpt_summaries = {}
            return exp_df, ds_summaries, gpt_summaries
    except Exception as e:
        print(f"Error loading operations summary: {e}")
        return pd.DataFrame(), {}, {}


__all__ = [
    "init_experiments_db",
    "init_db",
    "log_analytics_to_db",
    "save_chat_to_db",
    "delete_chat_from_db",
    "load_chats_from_db",
    "get_active_chat",
    "create_chat",
    "delete_experiment_from_db",
    "get_reasoning_eval_data",
    "get_dual_judge_comparison_data",
    "get_operations_summary",
    "DB_PATH",
    "EXP_DB_PATH",
]
