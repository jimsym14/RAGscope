import os
import sqlite3
import subprocess
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from config import (
    CHROMA_CANDIDATE_PATHS, DATA_DIR, EXP_DB_PATH, MODEL_REGISTRY,
    get_judge_config_status,
)

DATASET_CANDIDATE_PATHS = [
    os.path.join(DATA_DIR, "thesis_benchmark_103_complete.csv"),
    os.path.join(DATA_DIR, "thesis_benchmark.csv"),
]


def validate_experiments_db(db_path: str = EXP_DB_PATH) -> Optional[str]:
    if not os.path.isfile(db_path):
        return f"Evaluation results database was not found: {db_path}"
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            result = conn.execute("PRAGMA quick_check").fetchone()
            if not result or result[0] != "ok":
                return f"SQLite integrity check failed for {db_path}: {result[0] if result else 'no result'}"
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
            required = {"experiments", "detailed_results", "judge_scores"}
            missing = sorted(required - tables)
            if missing:
                return f"Evaluation database is missing required tables: {', '.join(missing)}"
    except sqlite3.Error as exc:
        return f"Evaluation results database could not be read: {exc}"
    return None


def load_evaluation_experiments(db_path: str = EXP_DB_PATH, judge_provider: str = "deepseek_v4") -> pd.DataFrame:
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
    except FileNotFoundError:
        pass
    except Exception as e:
        health["warnings"].append(f"Ollama status check failed: {e}")

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

def load_adversarial_summary(db_path: str) -> pd.DataFrame:
    if not os.path.isfile(db_path):
        return pd.DataFrame()
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            return pd.read_sql_query(
                "SELECT id, experiment_name, model_name, config_id, config_name, "
                "total_questions, refused_count, hallucinated_count, "
                "round(refusal_rate * 100, 1) AS refusal_rate_pct, "
                "round(hallucination_rate * 100, 1) AS hallucination_rate_pct, "
                "avg_speed_tps, avg_duration_sec, timestamp "
                "FROM adversarial_experiments "
                "ORDER BY refusal_rate_pct DESC, hallucination_rate_pct ASC",
                conn,
            )
    except sqlite3.Error:
        return pd.DataFrame()
