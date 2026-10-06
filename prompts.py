import json
import os
from typing import Any, Dict, List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_PROMPTS_PATH = os.path.join(BASE_DIR, "data", "test_prompts.json")

FINANCIAL_AUDITOR_TEMPLATE = (
    "You are a strict Expert Financial Auditor.\n"
    "---------------------\n{context_str}\n---------------------\n"
    "Given the context information, answer the query.\n"
    "Query: {query_str}\nAnswer: "
)

STREAMING_SYSTEM_PROMPT_TEMPLATE = (
    "You are a strict Expert Financial Auditor.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information, answer the query accurately.\n"
    "Formatting guidelines:\n"
    "- For mathematical calculations, write standard clean equations (e.g., (136,700 - 155,041) / 155,041 = -11.83%) rather than raw unrendered bracketed LaTeX codes.\n"
    "- State both the exact calculated figure and any reported/rounded figure mentioned in the 10-K filings.\n"
)

HYDE_PROMPT_TEMPLATE = (
    "Please write a financial passage from an SEC Form 10-K filing that accurately answers the following query.\n"
    "Query: {query_str}\n"
    "Passage: "
)

_FALLBACK_TEST_PROMPTS: List[Dict[str, Any]] = [
    {
        "id": "gross_margin",
        "label": "Gross Margin Trajectory (FY21-FY23)",
        "category": "Multi-Hop Financial Analysis",
        "prompt": "What was Apple's gross margin percentage in fiscal years 2021, 2022, and 2023, and what primary factors drove the year-over-year changes according to MD&A?",
    },
    {
        "id": "services_cogs",
        "label": "Services Cost of Sales (FY22 vs FY23)",
        "category": "Comparative Expense Analysis",
        "prompt": "How did Services cost of sales change from FY22 to FY23, and what specific drivers were cited in the 10-K?",
    },
    {
        "id": "rd_expense",
        "label": "R&D Expense Scaling (FY21-FY23)",
        "category": "Ratio & Trend Calculation",
        "prompt": "Compare Research and Development expenses across FY21, FY22, and FY23 as a percentage of total net sales. Why did R&D spending increase in absolute dollar terms?",
    },
    {
        "id": "risk_supply",
        "label": "Supply Chain Single-Source Risks (FY23)",
        "category": "Qualitative Risk Factor",
        "prompt": "What specific risks regarding single-source component suppliers and outsourced manufacturing are outlined in Item 1A of the FY23 10-K?",
    },
    {
        "id": "share_repurchase",
        "label": "Capital Return & Share Repurchases",
        "category": "Capital Structure & Liquidity",
        "prompt": "Detail Apple's share repurchase program activity in fiscal 2023, including total dollars spent, average price per share if reported, and remaining authorized capacity under the program.",
    },
]


def load_test_prompts(filepath: str = TEST_PROMPTS_PATH) -> List[Dict[str, Any]]:
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    return data
        except Exception as e:
            print(f"Warning: Failed to load test prompts from {filepath}: {e}")
    return _FALLBACK_TEST_PROMPTS


__all__ = [
    "FINANCIAL_AUDITOR_TEMPLATE",
    "STREAMING_SYSTEM_PROMPT_TEMPLATE",
    "HYDE_PROMPT_TEMPLATE",
    "load_test_prompts",
    "TEST_PROMPTS_PATH",
]
