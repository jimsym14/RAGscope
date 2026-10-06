

import os
import time
import json
import asyncio
import logging
import threading
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional, Callable
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_openai")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Unexpected type for token usage.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")

from datasets import Dataset
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.outputs import LLMResult, Generation
from ragas.llms.base import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.metrics import (
    Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall,
    faithfulness, answer_relevancy, context_precision, context_recall
)
from ragas import evaluate

logger = logging.getLogger("JudgeProviders")



_SHARED_BGE_EMBEDDINGS: Optional[HuggingFaceEmbeddings] = None
_EMBED_LOCK = threading.Lock()


def get_shared_bge_embeddings() -> HuggingFaceEmbeddings:
    
    global _SHARED_BGE_EMBEDDINGS
    if _SHARED_BGE_EMBEDDINGS is None:
        with _EMBED_LOCK:
            if _SHARED_BGE_EMBEDDINGS is None:
                _SHARED_BGE_EMBEDDINGS = HuggingFaceEmbeddings(
                    model_name="BAAI/bge-small-en-v1.5",
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                    show_progress=False
                )
    return _SHARED_BGE_EMBEDDINGS




class BaseJudgeProvider(ABC):
    

    @property
    @abstractmethod
    def provider_id(self) -> str:
        
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        
        pass

    @property
    @abstractmethod
    def short_name(self) -> str:
        
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        
        pass

    @abstractmethod
    def is_configured(self) -> Tuple[bool, str, Dict[str, Any]]:
        
        pass

    @abstractmethod
    def get_ragas_llm(self) -> LangchainLLMWrapper:
        
        pass

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        
        return get_shared_bge_embeddings()

    def score_single_item(
        self,
        item: Dict[str, Any],
        max_retries: int = 2,
        backoff_sec: float = 2.0,
        requested_metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        
        q = str(item.get("question", ""))
        a = str(item.get("generated_answer", ""))
        gt = str(item.get("ground_truth", ""))
        raw_ctx = item.get("contexts", [])
        
        if isinstance(raw_ctx, str):
            try:
                contexts = json.loads(raw_ctx)
            except Exception:
                contexts = [raw_ctx] if raw_ctx.strip() else [""]
        elif isinstance(raw_ctx, list):
            contexts = [str(c) for c in raw_ctx] if raw_ctx else [""]
        else:
            contexts = [""]

        ragas_data = {
            "question": [q],
            "answer": [a],
            "contexts": [contexts if contexts else [""]],
            "ground_truth": [gt]
        }
        ragas_ds = Dataset.from_dict(ragas_data)

        ragas_llm = self.get_ragas_llm()
        judge_embed = self.get_embeddings()

        m_faith = Faithfulness(llm=ragas_llm)
        m_rel = AnswerRelevancy(llm=ragas_llm, embeddings=judge_embed)
        m_cp = ContextPrecision(llm=ragas_llm)
        m_cr = ContextRecall(llm=ragas_llm)
        m_faith.strict = False
        m_rel.strict = False

        metric_map = {
            "faithfulness": m_faith,
            "answer_relevancy": m_rel,
            "context_precision": m_cp,
            "context_recall": m_cr
        }

        if requested_metrics:
            active_metrics = [metric_map[m] for m in requested_metrics if m in metric_map]
            if not active_metrics:
                active_metrics = [m_faith, m_rel, m_cp, m_cr]
        else:
            active_metrics = [m_faith, m_rel, m_cp, m_cr]

        ragas_run_config = RunConfig(
            max_workers=1,
            timeout=180,
            max_retries=3
        )

        last_error = None
        last_error_type = None
        retries_used = 0

        for attempt in range(1, max_retries + 1):
            try:
                t0 = time.time()
                results = evaluate(
                    dataset=ragas_ds,
                    metrics=active_metrics,
                    llm=ragas_llm,
                    embeddings=judge_embed,
                    run_config=ragas_run_config,
                    show_progress=False
                )
                dur = time.time() - t0

                res_df = results.to_pandas() if hasattr(results, "to_pandas") else pd.DataFrame(results)
                if res_df.empty:
                    raise RuntimeError("Ragas evaluation returned an empty result dataframe.")

                def extract_metric(col_name: str) -> Optional[float]:
                    val = res_df.iloc[0].get(col_name)
                    if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)):
                        return None
                    try:
                        f_val = float(val)
                        return round(min(max(f_val, 0.0), 1.0), 4)
                    except (ValueError, TypeError):
                        return None

                active_names = [m.name for m in active_metrics]
                
                f_val = extract_metric("faithfulness") if "faithfulness" in active_names else item.get("faithfulness")
                r_val = extract_metric("answer_relevancy") if "answer_relevancy" in active_names else item.get("answer_relevancy")
                cp_val = extract_metric("context_precision") if "context_precision" in active_names else item.get("context_precision")
                cr_val = extract_metric("context_recall") if "context_recall" in active_names else item.get("context_recall")

                valid_scores = [v for v in [f_val, r_val, cp_val, cr_val] if v is not None]

                if len(valid_scores) == 0:
                    raise RuntimeError("Ragas evaluator worker jobs failed: all metrics returned NaN.")

                if len(valid_scores) == 4:
                    status = "completed"
                    overall = round(float(np.mean(valid_scores)), 4)
                else:
                    status = "partial"
                    overall = None  # Do NOT average partial results with hidden zeros!

                return {
                    "status": status,
                    "faithfulness": f_val,
                    "answer_relevancy": r_val,
                    "context_precision": cp_val,
                    "context_recall": cr_val,
                    "overall_score": overall,
                    "evaluation_time_sec": round(dur, 3),
                    "error_message": None,
                    "error_type": None,
                    "retry_count": retries_used,
                    "judge_provider": self.provider_id,
                    "judge_model": self.model_name,
                    "judge_version": "v2"
                }

            except Exception as e:
                retries_used += 1
                last_error = str(e)
                last_error_type = type(e).__name__
                logger.warning(f"[{self.display_name}] Attempt {attempt}/{max_retries} failed for Q#{item.get('question_idx')}: {e}")
                if attempt < max_retries:
                    sleep_dur = backoff_sec * (2 ** (attempt - 1))
                    time.sleep(sleep_dur)

        return {
            "status": "error",
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": None,
            "overall_score": None,
            "evaluation_time_sec": 0.0,
            "error_message": last_error or "Unknown evaluation error",
            "error_type": last_error_type or "EvaluationError",
            "retry_count": retries_used,
            "judge_provider": self.provider_id,
            "judge_model": self.model_name,
            "judge_version": "v2"
        }




class DeepSeekJudge(BaseJudgeProvider):
    

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key

    @property
    def provider_id(self) -> str:
        return "deepseek_v4"

    @property
    def display_name(self) -> str:
        return "DeepSeek V4 Flash"

    @property
    def short_name(self) -> str:
        return "DeepSeek V4"

    @property
    def model_name(self) -> str:
        return "deepseek-chat"

    def get_api_key(self) -> Optional[str]:
        return self._api_key or os.getenv("DEEPSEEK_API_KEY")

    def is_configured(self) -> Tuple[bool, str, Dict[str, Any]]:
        key = self.get_api_key()
        is_ready = bool(key and len(key.strip()) >= 10)
        msg = "Ready (DeepSeek API Key configured)" if is_ready else "Missing DEEPSEEK_API_KEY in environment (.env)"
        return is_ready, msg, {
            "provider": self.display_name,
            "model": self.model_name,
            "api_key_configured": is_ready,
            "base_url": "https://api.deepseek.com"
        }

    def get_ragas_llm(self) -> LangchainLLMWrapper:
        key = self.get_api_key()
        if not key:
            raise ValueError("DEEPSEEK_API_KEY is required for DeepSeek Judge.")
        
        judge_llm = ChatOpenAI(
            model=self.model_name,
            api_key=key,  # type: ignore
            base_url="https://api.deepseek.com",
            temperature=0.0,
            max_retries=3
        )
        return LangchainLLMWrapper(judge_llm, bypass_n=True)




class ReasoningLangchainLLMWrapper(LangchainLLMWrapper):
    

    async def agenerate_text(
        self,
        prompt: Any,
        n: int = 1,
        temperature: float = 1e-8,
        callbacks: Any = None,
        **kwargs: Any
    ) -> LLMResult:
        if n > 1:
            tasks = [
                self.langchain_llm.agenerate_prompt([prompt], callbacks=callbacks)
                for _ in range(n)
            ]
            results = await asyncio.gather(*tasks)
            combined_gens: List[Generation] = []
            for res in results:
                if res.generations and res.generations[0]:
                    combined_gens.extend(res.generations[0])
            return LLMResult(generations=[combined_gens])
        else:
            return await self.langchain_llm.agenerate_prompt(
                [prompt],
                callbacks=callbacks
            )

    def generate_text(
        self,
        prompt: Any,
        n: int = 1,
        temperature: float = 1e-8,
        callbacks: Any = None,
        **kwargs: Any
    ) -> LLMResult:
        if n > 1:
            combined_gens: List[Generation] = []
            for _ in range(n):
                res = self.langchain_llm.generate_prompt([prompt], callbacks=callbacks)
                if res.generations and res.generations[0]:
                    combined_gens.extend(res.generations[0])
            return LLMResult(generations=[combined_gens])
        else:
            return self.langchain_llm.generate_prompt(
                [prompt],
                callbacks=callbacks
            )




class AzureGPT5MiniJudge(BaseJudgeProvider):
    

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment: Optional[str] = None,
        api_version: Optional[str] = None
    ):
        self._api_key = api_key
        self._endpoint = endpoint
        self._deployment = deployment
        self._api_version = api_version

    @property
    def provider_id(self) -> str:
        return "azure_gpt5_mini"

    @property
    def display_name(self) -> str:
        return "GPT-5-mini (Azure)"

    @property
    def short_name(self) -> str:
        return "GPT-5-mini"

    @property
    def model_name(self) -> str:
        return self._deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT") or "gpt-5-mini"

    def get_api_key(self) -> Optional[str]:
        return self._api_key or os.getenv("AZURE_OPENAI_API_KEY")

    def get_endpoint(self) -> Optional[str]:
        return self._endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")

    def get_deployment(self) -> str:
        return self._deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT") or "gpt-5-mini"

    def get_api_version(self) -> str:
        return self._api_version or os.getenv("AZURE_OPENAI_API_VERSION") or "2024-08-01-preview"

    def is_configured(self) -> Tuple[bool, str, Dict[str, Any]]:
        key = self.get_api_key()
        endpoint = self.get_endpoint()
        deployment = self.get_deployment()
        api_ver = self.get_api_version()

        missing = []
        if not key or len(key.strip()) < 10:
            missing.append("AZURE_OPENAI_API_KEY")
        if not endpoint or not endpoint.strip().startswith("http"):
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not deployment or not deployment.strip():
            missing.append("AZURE_OPENAI_DEPLOYMENT")

        meta = {
            "provider": self.display_name,
            "endpoint": endpoint if endpoint else "Not Set",
            "deployment": deployment if deployment else "Not Set",
            "api_version": api_ver,
            "api_key_configured": bool(key and len(key.strip()) >= 10),
            "reasoning_effort": "low",
            "max_completion_tokens": 2048
        }

        if missing:
            return False, f"Missing Azure configuration: {', '.join(missing)}", meta
        return True, f"Ready (Azure deployment '{deployment}' configured)", meta

    def get_ragas_llm(self) -> LangchainLLMWrapper:
        is_ready, msg, _ = self.is_configured()
        if not is_ready:
            raise ValueError(f"Azure GPT-5-mini Judge configuration error: {msg}")

        endpoint = str(self.get_endpoint() or "").strip()
        deployment = self.get_deployment()
        key = self.get_api_key()

        if "services.ai.azure.com" in endpoint:
            base_url = endpoint.split("/api/projects/")[0].rstrip("/") + "/models"
            
            judge_llm = ChatOpenAI(
                base_url=base_url,
                api_key=key,  # type: ignore
                model=deployment,
                max_completion_tokens=4096,
                reasoning_effort="low",
                max_retries=5,
                timeout=90
            )
            return LangchainLLMWrapper(judge_llm, bypass_temperature=True, bypass_n=True)
        else:
            judge_llm = AzureChatOpenAI(
                azure_endpoint=endpoint,
                api_key=key,  # type: ignore
                azure_deployment=deployment,
                api_version=self.get_api_version(),
                max_completion_tokens=4096,
                max_retries=5,
                timeout=90
            )
            return LangchainLLMWrapper(judge_llm, bypass_temperature=True, bypass_n=True)




_JUDGE_REGISTRY: Dict[str, BaseJudgeProvider] = {
    "deepseek_v4": DeepSeekJudge(),
    "azure_gpt5_mini": AzureGPT5MiniJudge(),
}


def get_judge_provider(provider_id: str) -> BaseJudgeProvider:
    
    if provider_id not in _JUDGE_REGISTRY:
        raise ValueError(f"Unknown Judge Provider '{provider_id}'. Available: {list(_JUDGE_REGISTRY.keys())}")
    return _JUDGE_REGISTRY[provider_id]


def list_judge_providers() -> List[Dict[str, Any]]:
    
    providers = []
    for pid, provider in _JUDGE_REGISTRY.items():
        is_ready, msg, meta = provider.is_configured()
        providers.append({
            "id": pid,
            "display_name": provider.display_name,
            "short_name": provider.short_name,
            "model_name": provider.model_name,
            "is_configured": is_ready,
            "status_message": msg,
            "metadata": meta
        })
    return providers
