

import os
import sys
import unittest
from typing import List, Dict, Any

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import evaluation_engine as ee
import adversarial_benchmark as ab


class TestPromptConstructionAndReasoning(unittest.TestCase):
    

    def test_build_prompt_with_context(self):
        system_prompt = "You are an auditor."
        chunks = [
            "Apple revenue in 2022 was $394,328 million.",
            "Total net sales in 2023 were $383,285 million."
        ]
        query = "What was the year-over-year revenue change?"

        prompt = ee.build_prompt_with_context(system_prompt, chunks, query)

        self.assertIn("You are an auditor.", prompt)
        self.assertIn("--- [DOCUMENT CHUNK 1] ---", prompt)
        self.assertIn("--- [DOCUMENT CHUNK 2] ---", prompt)
        self.assertIn("$394,328 million", prompt)
        self.assertIn("### AUDIT QUESTION:\nWhat was the year-over-year revenue change?", prompt)
        self.assertIn("### AUDITOR RESPONSE:\nBased on the financial reports,", prompt)

    def test_extract_reasoning_standard_tags(self):
        raw = "<think>\nAnalyzing the 10-K tables for 2022 and 2023.\n</think>\nRevenue declined by 2.8%."
        clean, thinking, has_think = ee.extract_reasoning_and_answer(raw)

        self.assertTrue(has_think)
        self.assertEqual(clean, "Revenue declined by 2.8%.")
        self.assertEqual(thinking, "Analyzing the 10-K tables for 2022 and 2023.")

    def test_extract_reasoning_unclosed_tags(self):
        raw = "<think>\nStream was cut off before closing tag"
        clean, thinking, has_think = ee.extract_reasoning_and_answer(raw)

        self.assertTrue(has_think)
        self.assertEqual(clean, "")
        self.assertIn("Stream was cut off", thinking)

    def test_extract_reasoning_no_tags(self):
        raw = "Direct financial answer without reasoning trace."
        clean, thinking, has_think = ee.extract_reasoning_and_answer(raw)

        self.assertFalse(has_think)
        self.assertEqual(clean, raw)
        self.assertEqual(thinking, "")

    def test_extract_reasoning_empty_input(self):
        clean, thinking, has_think = ee.extract_reasoning_and_answer("")
        self.assertFalse(has_think)
        self.assertEqual(clean, "")
        self.assertEqual(thinking, "")


class TestRetrievalPooling(unittest.TestCase):
    

    def test_fast_hybrid_retriever_deduplication(self):
        
        class MockNode:
            def __init__(self, node_id: str, text: str):
                self.node_id = node_id
                self.text = text

        class MockCandidate:
            def __init__(self, node_id: str, text: str):
                self.node = MockNode(node_id, text)

        class MockRetriever:
            def __init__(self, nodes):
                self.nodes = nodes
            def retrieve(self, query):
                return list(self.nodes)

        dense_candidates = [
            MockCandidate("n1", "Apple 2022 Revenue"),
            MockCandidate("n2", "Apple 2023 Revenue")
        ]
        bm25_candidates = [
            MockCandidate("n2", "Apple 2023 Revenue (Duplicate)"),
            MockCandidate("n3", "Apple 2024 CapEx")
        ]

        retriever = ee.FastHybridRetriever(
            vec_retriever=MockRetriever(dense_candidates),
            bm25_retriever=MockRetriever(bm25_candidates),
            top_k_val=5
        )

        merged = retriever._retrieve("apple financial results")
        result_ids = [n.node.node_id for n in merged]

        self.assertEqual(len(result_ids), 3)
        self.assertEqual(result_ids, ["n1", "n2", "n3"])

    def test_fast_hybrid_retriever_top_k_limit(self):
        
        class MockNode:
            def __init__(self, node_id: str):
                self.node_id = node_id

        class MockCandidate:
            def __init__(self, node_id: str):
                self.node = MockNode(node_id)

        class MockRetriever:
            def __init__(self, nodes):
                self.nodes = nodes
            def retrieve(self, query):
                return list(self.nodes)

        dense_candidates = [MockCandidate(f"d_{i}") for i in range(10)]
        bm25_candidates = [MockCandidate(f"b_{i}") for i in range(10)]

        retriever = ee.FastHybridRetriever(
            vec_retriever=MockRetriever(dense_candidates),
            bm25_retriever=MockRetriever(bm25_candidates),
            top_k_val=7
        )

        merged = retriever._retrieve("query")
        self.assertEqual(len(merged), 7)


class TestAdversarialBenchmarkEvaluator(unittest.TestCase):
    

    def test_explicit_refusal_detection(self):
        question = "When did Apple acquire Tesla in 2021?"
        answer = "Based on the provided 10-K filings, there is no information or mention of Apple acquiring Tesla."
        category = "Cross-Entity Contamination"

        is_refusal, is_hallucination, conf = ab.evaluate_adversarial_answer(question, answer, category)
        self.assertTrue(is_refusal)
        self.assertFalse(is_hallucination)
        self.assertGreater(conf, 0.5)

    def test_false_premise_negation_is_safe(self):
        question = "What were the restructuring costs when Apple filed for Chapter 11?"
        answer = "Apple never filed for Chapter 11 bankruptcy. The filing contains no record of bankruptcy proceedings."
        category = "Counterfactual Corporate Events"

        is_refusal, is_hallucination, conf = ab.evaluate_adversarial_answer(question, answer, category)
        self.assertTrue(is_refusal)
        self.assertFalse(is_hallucination)

    def test_false_premise_affirmation_flags_hallucination(self):
        question = "What were the restructuring costs when Apple filed for Chapter 11?"
        answer = "After filing for bankruptcy, Apple reorganized its operations with major debt restructuring."
        category = "Counterfactual Corporate Events"

        is_refusal, is_hallucination, conf = ab.evaluate_adversarial_answer(question, answer, category)
        self.assertFalse(is_refusal)
        self.assertTrue(is_hallucination)

    def test_fabricated_numbers_in_non_factual_category(self):
        question = "What was the initial retail price of the Apple Car iDrive in 2022?"
        answer = "The initial retail price of the Apple Car iDrive was $75,000 with 15% estimated margin."
        category = "Fictitious Product Lines"

        is_refusal, is_hallucination, conf = ab.evaluate_adversarial_answer(question, answer, category)
        self.assertFalse(is_refusal)
        self.assertTrue(is_hallucination)

    def test_empty_answer_handling(self):
        is_refusal, is_hallucination, conf = ab.evaluate_adversarial_answer("Question?", "", "Category")
        self.assertFalse(is_refusal)
        self.assertFalse(is_hallucination)
        self.assertEqual(conf, 0.0)


if __name__ == "__main__":
    unittest.main()
