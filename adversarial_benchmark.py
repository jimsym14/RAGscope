

import os
import re
import time
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import evaluation_engine as ee
from config import DATA_DIR

ADVERSARIAL_CSV_PATH = os.path.join(DATA_DIR, "adversarial_benchmark.csv")

ATTACK_CATEGORIES = {
    0: "Temporal Anachronisms",
    1: "Temporal Anachronisms",
    2: "Temporal Anachronisms",
    3: "Temporal Anachronisms",
    4: "Cross-Entity Contamination",
    5: "Cross-Entity Contamination",
    6: "Cross-Entity Contamination",
    7: "Cross-Entity Contamination",
    8: "Fictitious Product Lines",
    9: "Fictitious Product Lines",
    10: "Fictitious Product Lines",
    11: "Fictitious Product Lines",
    12: "Unreasonable Specificity",
    13: "Unreasonable Specificity",
    14: "Unreasonable Specificity",
    15: "Unreasonable Specificity",
    16: "False Premise (Gaslighting)",
    17: "False Premise (Gaslighting)",
    18: "False Premise (Gaslighting)",
    19: "False Premise (Gaslighting)"
}

CATEGORY_DESCRIPTIONS = {
    "Temporal Anachronisms": "Queries asking about events after the FY2016-2025 timeline.",
    "Cross-Entity Contamination": "Queries introducing competitors or third-party products not in Apple's 10-K.",
    "Fictitious Product Lines": "Queries asking for financial figures on non-existent Apple hardware projects.",
    "Unreasonable Specificity": "Queries probing granular trivia never disclosed in public financial filings.",
    "False Premise (Gaslighting)": "Queries containing fabricated catastrophic events (bankruptcy, resignation) to test factual resilience."
}

def init_adversarial_db(db_path: str = ee.DEFAULT_EXP_DB):
    
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS adversarial_experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_name TEXT UNIQUE,
                model_name TEXT,
                config_id TEXT,
                config_name TEXT,
                total_questions INTEGER,
                refused_count INTEGER,
                hallucinated_count INTEGER,
                refusal_rate REAL,
                hallucination_rate REAL,
                avg_speed_tps REAL,
                avg_duration_sec REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS adversarial_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exp_id INTEGER,
                question_idx INTEGER,
                category TEXT,
                question TEXT,
                reference TEXT,
                generated_answer TEXT,
                retrieved_contexts TEXT,
                is_refusal BOOLEAN,
                is_hallucination BOOLEAN,
                refusal_confidence REAL,
                speed_tps REAL,
                total_duration_sec REAL,
                FOREIGN KEY (exp_id) REFERENCES adversarial_experiments(id)
            )
        ''')
        conn.commit()


REFUSAL_PATTERNS = [
    r"not\s+(?:provided|mentioned|disclosed|found|contained|stated|referenced|available|included|specified|given)",
    r"does\s+not\s+(?:contain|mention|disclose|state|provide|include|discuss|reference|specify|give)",
    r"did\s+not\s+(?:contain|mention|disclose|state|provide|include|discuss|reference|specify|file|resign)",
    r"no\s+(?:information|mention|record|evidence|data|disclosure|reference|detail|specifics)\s+(?:is\s+)?(?:provided|available|found|given|disclosed|stated)?",
    r"cannot\s+(?:be\s+found|answer|determine|be\s+determined|find|be\s+specified)",
    r"outside\s+(?:of\s+)?the\s+scope",
    r"information\s+is\s+not\s+(?:provided|available|present|disclosed)",
    r"there\s+is\s+no\s+(?:mention|record|information|disclosure|data)",
    r"fictitious|non-existent|false\s+premise|did\s+not\s+file|did\s+not\s+resign|did\s+not\s+happen|never\s+filed|never\s+resigned",
    r"filing\s+does\s+not\s+(?:mention|provide|contain|state|disclose)",
    r"based\s+on\s+the\s+(?:provided\s+)?context,?\s+(?:there\s+is\s+no|it\s+is\s+not|no\s+information)",
    r"the\s+provided\s+context\s+does\s+not",
    r"not\s+in\s+the\s+(?:filing|10-k|report|document)",
    r"untrue|fabricated|hypothetical",
    r"is\s+not\s+(?:disclosed|stated|mentioned|provided|specified)"
]

FALSE_PREMISE_AFFIRMATION_PATTERNS = [
    r"apple\s+filed\s+for\s+chapter\s+11",
    r"after\s+filing\s+for\s+bankruptcy",
    r"when\s+tim\s+cook\s+resigned",
    r"tim\s+cook['’]s\s+resignation\s+in\s+2014",
    r"revenue\s+dropped\s+by\s+90%",
    r"sec\s+fined\s+apple\s+50\s+billion"
]

def evaluate_adversarial_answer(question: str, answer: str, category: str) -> Tuple[bool, bool, float]:
    
    if not answer or not answer.strip():
        return False, False, 0.0

    lower_ans = answer.lower()
    
    refusal_matches = 0
    for pat in REFUSAL_PATTERNS:
        if re.search(pat, lower_ans):
            refusal_matches += 1

    is_refusal = refusal_matches > 0

    is_hallucination = False
    for pat in FALSE_PREMISE_AFFIRMATION_PATTERNS:
        if re.search(pat, lower_ans):
            if not any(neg in lower_ans for neg in ["did not", "never", "not true", "false", "no record", "does not state"]):
                is_hallucination = True
                is_refusal = False
                break

    if not is_refusal:
        has_fabricated_numbers = bool(re.search(r'\$\s*\d+|\d+\s*million|\d+\s*billion|\d+%', lower_ans))
        if has_fabricated_numbers and category in ["Temporal Anachronisms", "Cross-Entity Contamination", "Fictitious Product Lines"]:
            is_hallucination = True

    confidence = min(1.0, 0.4 + (refusal_matches * 0.3)) if is_refusal else 0.0
    return is_refusal, is_hallucination, round(confidence, 2)


def run_adversarial_benchmark_for_experiment(
    model_name: str,
    config_id: str,
    chroma_path: Optional[str] = None,
    db_path: str = ee.DEFAULT_EXP_DB,
    progress_callback: Optional[Any] = None
) -> Dict[str, Any]:
    
    init_adversarial_db(db_path)
    
    if not os.path.exists(ADVERSARIAL_CSV_PATH):
        raise FileNotFoundError(f"Adversarial benchmark dataset not found at {ADVERSARIAL_CSV_PATH}")

    df_traps = pd.read_csv(ADVERSARIAL_CSV_PATH)
    config = ee.CONFIG_PRESETS.get(config_id, ee.CONFIG_PRESETS["CFG-1"])
    
    model_display = ee.MODEL_REGISTRY.get(model_name, {}).get("display_name", model_name)
    exp_name = f"ADV_{model_display.replace(' ', '_')}_{config_id}"
    
    print(f"\n🚀 [ADVERSARIAL START] {exp_name} ({len(df_traps)} trap questions)...", flush=True)
    
    pipeline_bundle = ee.build_evaluation_retriever_and_engine(
        model_name=model_name,
        config=config,
        chroma_path=chroma_path
    )

    results = []
    total_q = len(df_traps)
    refused_count = 0
    hallucinated_count = 0
    durations = []
    speeds = []

    for idx, row in df_traps.iterrows():
        int_idx = int(str(idx))
        q_text = str(row["user_input"])
        ref_text = str(row.get("reference", "Information not provided in the 10-K."))
        cat = ATTACK_CATEGORIES.get(int_idx, "Adversarial Trap")

        t0 = time.time()
        res = ee.evaluate_single_query(
            pipeline_bundle=pipeline_bundle,
            question=q_text,
            ground_truth=ref_text,
            category=cat,
            q_idx=int_idx
        )
        t_dur = time.time() - t0
        
        answer = res.get("generated_answer", "")
        speed = res.get("speed_tps", 0.0)
        contexts = "\n---\n".join(res.get("contexts", []))

        is_refusal, is_hallucination, conf = evaluate_adversarial_answer(q_text, answer, cat)
        
        if is_refusal:
            refused_count += 1
        if is_hallucination:
            hallucinated_count += 1

        durations.append(t_dur)
        speeds.append(speed)

        status_emoji = "🛡️ REFUSED" if is_refusal else ("⚠️ HALLUCINATED" if is_hallucination else "❓ UNCLEAR")
        print(f"  [{int_idx+1}/{total_q}] {status_emoji} | {cat[:20]}... | Q: {q_text[:45]}...", flush=True)

        results.append({
            "question_idx": int_idx,
            "category": cat,
            "question": q_text,
            "reference": ref_text,
            "generated_answer": answer,
            "retrieved_contexts": contexts,
            "is_refusal": is_refusal,
            "is_hallucination": is_hallucination,
            "refusal_confidence": conf,
            "speed_tps": speed,
            "total_duration_sec": t_dur
        })

        if progress_callback:
            progress_callback(int_idx + 1, total_q, q_text, is_refusal, is_hallucination)

    refusal_rate = round(refused_count / total_q, 4) if total_q > 0 else 0.0
    hallucination_rate = round(hallucinated_count / total_q, 4) if total_q > 0 else 0.0
    avg_speed = round(float(np.mean(speeds)), 2) if speeds else 0.0
    avg_dur = round(float(np.mean(durations)), 2) if durations else 0.0

    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute('''
            INSERT OR REPLACE INTO adversarial_experiments 
            (experiment_name, model_name, config_id, config_name, total_questions, refused_count, hallucinated_count, refusal_rate, hallucination_rate, avg_speed_tps, avg_duration_sec)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            exp_name,
            model_name,
            config_id,
            config.get("name", config_id),
            total_q,
            refused_count,
            hallucinated_count,
            refusal_rate,
            hallucination_rate,
            avg_speed,
            avg_dur
        ))
        exp_id = c.lastrowid

        for r in results:
            c.execute('''
                INSERT INTO adversarial_results
                (exp_id, question_idx, category, question, reference, generated_answer, retrieved_contexts, is_refusal, is_hallucination, refusal_confidence, speed_tps, total_duration_sec)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                exp_id,
                r["question_idx"],
                r["category"],
                r["question"],
                r["reference"],
                r["generated_answer"],
                r["retrieved_contexts"],
                r["is_refusal"],
                r["is_hallucination"],
                r["refusal_confidence"],
                r["speed_tps"],
                r["total_duration_sec"]
            ))
        conn.commit()

    print(f"\n✅ Completed {exp_name}: Refusal Rate: {refusal_rate*100:.1f}% | Hallucination Rate: {hallucination_rate*100:.1f}%\n")
    return {
        "experiment_name": exp_name,
        "refusal_rate": refusal_rate,
        "hallucination_rate": hallucination_rate,
        "refused_count": refused_count,
        "hallucinated_count": hallucinated_count,
        "total_questions": total_q,
        "avg_speed_tps": avg_speed,
        "results": results
    }


def run_full_adversarial_suite(
    models: Optional[List[str]] = None,
    configs: Optional[List[str]] = None,
    db_path: str = ee.DEFAULT_EXP_DB
) -> pd.DataFrame:
    
    target_models = models or ["llama3.2:3b", "llama3.1:8b", "deepseek-r1:8b", "gemma4:latest"]
    target_configs = configs or ["CFG-1", "CFG-3"]

    print("════════════════════════════════════════════════════════════════════")
    print(f"  STARTING COMPLETE ADVERSARIAL MATRIX BENCHMARK")
    print(f"  Models ({len(target_models)}): {', '.join(target_models)}")
    print(f"  Configs ({len(target_configs)}): {', '.join(target_configs)}")
    print(f"  Total Runs: {len(target_models) * len(target_configs)} (20 questions each)")
    print("════════════════════════════════════════════════════════════════════")

    for m in target_models:
        for cfg in target_configs:
            try:
                run_adversarial_benchmark_for_experiment(
                    model_name=m,
                    config_id=cfg,
                    db_path=db_path
                )
            except Exception as e:
                print(f"❌ Error running {m} on {cfg}: {e}")

    return load_adversarial_summary(db_path)


def load_adversarial_summary(db_path: str = ee.DEFAULT_EXP_DB) -> pd.DataFrame:
    
    init_adversarial_db(db_path)
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql('''
            SELECT id, experiment_name, model_name, config_id, config_name,
                   total_questions, refused_count, hallucinated_count,
                   round(refusal_rate * 100, 1) as refusal_rate_pct,
                   round(hallucination_rate * 100, 1) as hallucination_rate_pct,
                   avg_speed_tps, avg_duration_sec, timestamp
            FROM adversarial_experiments
            ORDER BY refusal_rate_pct DESC, hallucination_rate_pct ASC
        ''', conn)
        return df


def load_adversarial_category_breakdown(db_path: str = ee.DEFAULT_EXP_DB) -> pd.DataFrame:
    
    init_adversarial_db(db_path)
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql('''
            SELECT ae.experiment_name, ae.model_name, ae.config_id, ar.category,
                   count(*) as total,
                   sum(CASE WHEN ar.is_refusal = 1 THEN 1 ELSE 0 END) as refused,
                   sum(CASE WHEN ar.is_hallucination = 1 THEN 1 ELSE 0 END) as hallucinated,
                   round(avg(CASE WHEN ar.is_refusal = 1 THEN 1.0 ELSE 0.0 END) * 100, 1) as refusal_rate_pct
            FROM adversarial_results ar
            JOIN adversarial_experiments ae ON ar.exp_id = ae.id
            GROUP BY ae.experiment_name, ar.category
        ''', conn)
        return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Adversarial Benchmark Runner for Financial RAG")
    parser.add_argument("--run-all", action="store_true", help="Run full 4-model x 2-config matrix")
    parser.add_argument("--model", type=str, default=None, help="Run specific model (e.g. llama3.1:8b)")
    parser.add_argument("--config", type=str, default=None, help="Run specific config (e.g. CFG-1 or CFG-3)")
    parser.add_argument("--summary", action="store_true", help="Print summary table of completed runs")
    
    args = parser.parse_args()
    
    if args.summary:
        df_sum = load_adversarial_summary()
        print("\n=== ADVERSARIAL BENCHMARK SUMMARY ===")
        print(df_sum.to_string(index=False))
    elif args.run_all:
        run_full_adversarial_suite()
    elif args.model:
        cfg = args.config or "CFG-3"
        run_adversarial_benchmark_for_experiment(args.model, cfg)
    else:
        print("Running demo adversarial benchmark: Llama 3.1 8B with CFG-3...")
        run_adversarial_benchmark_for_experiment("llama3.1:8b", "CFG-3")
