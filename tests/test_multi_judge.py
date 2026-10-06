

import os
import sys
import tempfile
import unittest
import sqlite3
import pandas as pd
import numpy as np

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import evaluation_engine as ee
import judge_providers as jp


class TestMultiJudgeArchitecture(unittest.TestCase):

    def test_judge_provider_registry(self):
        
        ds = jp.get_judge_provider("deepseek_v4")
        self.assertEqual(ds.provider_id, "deepseek_v4")
        self.assertEqual(ds.display_name, "DeepSeek V4 Flash")
        self.assertEqual(ds.model_name, "deepseek-chat")

        az = jp.get_judge_provider("azure_gpt5_mini")
        self.assertEqual(az.provider_id, "azure_gpt5_mini")
        self.assertEqual(az.display_name, "GPT-5-mini (Azure)")
        self.assertIn(az.model_name, ["gpt-5-mini", os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")])

    def test_deepseek_existing_data_intact(self):
        
        db_path = ee.DEFAULT_EXP_DB
        if not os.path.exists(db_path):
            self.skipTest(f"Experiment database not found at {db_path}")

        with sqlite3.connect(db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM experiments WHERE avg_faithfulness IS NOT NULL")
            completed_exp_count = c.fetchone()[0]
            self.assertGreaterEqual(completed_exp_count, 10, f"Expected at least 10 completed experiments, found {completed_exp_count}")

            c.execute("SELECT COUNT(*) FROM detailed_results WHERE faithfulness IS NOT NULL")
            completed_det_count = c.fetchone()[0]
            self.assertGreaterEqual(completed_det_count, 1000, f"Expected at least 1000 scored detailed results, found {completed_det_count}")

    def test_judge_scores_sync_and_isolation(self):
        
        db_path = ee.DEFAULT_EXP_DB
        if not os.path.exists(db_path):
            self.skipTest(f"Experiment database not found at {db_path}")

        ee.init_experiments_db(db_path)

        df_ds = ee.load_evaluation_experiments(db_path, "deepseek_v4")
        self.assertFalse(df_ds.empty)
        scored_ds = df_ds[df_ds["avg_faithfulness"].notna() & (df_ds["status"] == "completed")]
        self.assertGreaterEqual(len(scored_ds), 10, "DeepSeek should have at least 10 fully scored experiments")

        df_az = ee.load_evaluation_experiments(db_path, "azure_gpt5_mini")
        self.assertFalse(df_az.empty)
        scored_az = df_az[df_az["avg_faithfulness"].notna() & (df_az["status"] == "completed")]
        
        with sqlite3.connect(db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM judge_experiment_summaries WHERE judge_provider = 'azure_gpt5_mini' AND status = 'completed'")
            actual_az_completed = c.fetchone()[0]

        self.assertEqual(len(scored_az), actual_az_completed, "GPT-5-mini must NEVER show DeepSeek scores as its own")

    def test_duplicate_prevention_in_judge_scores(self):
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_db = os.path.join(tmp_dir, "test_experiments.db")
            ee.init_experiments_db(test_db)

            with sqlite3.connect(test_db) as conn:
                c = conn.cursor()
                c.execute('''
                    INSERT INTO experiments (experiment_name, model_name, sample_size, status)
                    VALUES ('Test_Exp', 'llama3.1:8b', 103, 'completed')
                ''')
                exp_id = c.lastrowid

                c.execute('''
                    INSERT INTO judge_scores (experiment_id, question_idx, judge_provider, judge_model, faithfulness, status)
                    VALUES (?, 0, 'deepseek_v4', 'deepseek-chat', 0.85, 'completed')
                ''', (exp_id,))
                conn.commit()

                c.execute('''
                    INSERT INTO judge_scores (experiment_id, question_idx, judge_provider, judge_model, faithfulness, status)
                    VALUES (?, 0, 'azure_gpt5_mini', 'gpt-5-mini', 0.90, 'completed')
                ''', (exp_id,))
                conn.commit()

                c.execute('''
                    INSERT OR REPLACE INTO judge_scores (experiment_id, question_idx, judge_provider, judge_model, faithfulness, status)
                    VALUES (?, 0, 'deepseek_v4', 'deepseek-chat', 0.95, 'completed')
                ''', (exp_id,))
                conn.commit()

                c.execute("SELECT COUNT(*) FROM judge_scores WHERE experiment_id = ? AND question_idx = 0", (exp_id,))
                total_for_q0 = c.fetchone()[0]
                self.assertEqual(total_for_q0, 2, f"Expected exactly 2 records (1 per provider), got {total_for_q0}")

                c.execute("SELECT faithfulness FROM judge_scores WHERE experiment_id = ? AND question_idx = 0 AND judge_provider = 'deepseek_v4'", (exp_id,))
                updated_f = c.fetchone()[0]
                self.assertEqual(updated_f, 0.95, f"Expected updated faithfulness 0.95, got {updated_f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
