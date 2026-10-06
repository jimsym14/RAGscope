# RAGscope

RAGscope is a research application developed alongside an undergraduate Computer Science thesis. It presents stored evaluations of small language models and retrieval-augmented generation (RAG) configurations for questions grounded in Apple 10-K filings.

## Overview

The repository brings together a results dashboard, an optional local document-chat application, a vector visualizer, and research code. The Evaluation dashboard opens by default and displays the included result databases. Chat and research runs require additional local services, data, or API credentials.

## Main capabilities

- **RAG Chat:** Ask questions about PDF documents indexed locally. Chat uses Ollama for generation and Chroma for retrieval.
- **Evaluation Dashboard:** Explore stored model and RAG experiment results, judge scores, adversarial results, reasoning analysis, case studies, and sensitivity results.
- **Vector Visualizer:** View embeddings from a populated Chroma collection.
- **Research implementation:** Inspect experiment configuration, checkpointing, judge integrations, and the adversarial benchmark runner.

## Architecture

```text
Streamlit app
├── Evaluation → dashboard_data.py → SQLite result databases
├── Chat → document_pipeline.py → Chroma + Ollama
├── Visualizer → Chroma embeddings
└── Research runners → benchmark data + local models / judge APIs
```

The Dashboard reads stored results without loading the experiment runner, judge providers, adversarial runner, Ollama, or Chroma as required startup dependencies.

## Repository structure

```text
app.py, config.py, dashboard_data.py, db.py, prompts.py
views/                         Streamlit pages and Evaluation tabs
ui/                            Interface assets and styles
document_pipeline.py           Local PDF ingestion and RAG chat
evaluation_engine.py            Experiment and evaluation implementation
judge_providers.py              Optional API-based scoring
adversarial_benchmark.py        Adversarial runner and summary
data/                           Benchmark inputs and stored results
docs/                           Data and reproduction notes
tests/                          Unit tests
requirements*.txt               Dashboard, RAG, and research dependencies
```

## Quick start: Evaluation Dashboard

Install Python and create a virtual environment.

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The Evaluation page is the default. Its stored results work without Ollama, model weights, a Chroma index, or API keys. The Windows commands are provided for setup; a clean Windows installation has not yet been tested.

## RAG Chat and Visualizer setup

Install the additional dependencies:

```bash
python -m pip install -r requirements-rag.txt
```

Install and start [Ollama](https://ollama.com/). RAGscope uses any model reported as installed by Ollama; it prefers `llama3.1:8b` when that model is available. To install that model:

```bash
ollama pull llama3.1:8b
```

If Ollama is not already running as a service, start it with `ollama serve`. Then launch RAGscope and select **Chat**. The release does not include Apple 10-K PDFs or a prebuilt Chroma index. Chat supports uploading PDF files from its sidebar and indexing them locally. The default index path is `data/vector_store/chroma`.

Set `RAGSCOPE_VECTOR_STORE` to use an existing Chroma store. The Visualizer reads the `apple_financials` collection; no populated collection is included. Chat-uploaded PDFs use a separate chat collection and do not automatically populate the Visualizer's collection.

## Research and experiment reproduction

Install the research dependencies:

```bash
python -m pip install -r requirements-research.txt
```

The benchmark inputs are `data/thesis_benchmark_103_complete.csv` (103 questions) and `data/adversarial_benchmark.csv`. The stored experiment database contains 12 runs: 11 full-size runs with 103 questions each and one 10-question pilot. The sensitivity database contains 180 generation rows and 180 judge-evaluation rows.

The adversarial runner provides these command-line options:

```bash
python adversarial_benchmark.py --summary
python adversarial_benchmark.py --model llama3.1:8b --config CFG-3
python adversarial_benchmark.py --run-all
```

New runs require the appropriate local model, RAG index, and benchmark input. The repository includes stored results for inspection; it does not provide a single launcher for reproducing the full core/ablation experiment matrix or starting judge scoring. See [Research reproduction](docs/reproduce_experiments.md) for configuration locations and this release's execution limits.

## Evaluation data

- `data/experiments.db` contains experiment records, generated answers, retrieved contexts, reasoning traces, judge scores and summaries, and adversarial results.
- `data/sensitivity_ablation/sensitivity_runs.db` provides the sensitivity-analysis results shown in the Ablation view.
- `data/thesis_benchmark_103_complete.csv` and `data/adversarial_benchmark.csv` contain benchmark inputs.
- `data/test_prompts.json` contains sample chat prompts.

The database includes saved model outputs and retrieved text to support inspection of the evaluations. `data/chats.db`, local uploads, checkpoints, and generated vector stores are runtime data and are not included. See [Data guide](docs/data.md) before replacing or redistributing result data.

## Configuration

`.env.example` lists the optional judge API settings. Copy it to `.env` and add credentials only when using external judge APIs. Supported variables are:

- `DEEPSEEK_API_KEY`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_DEPLOYMENT`
- `AZURE_OPENAI_API_VERSION`

For local paths, the application supports `RAGSCOPE_DATA_DIR`, `RAGSCOPE_VECTOR_STORE`, and `RAGSCOPE_SENSITIVITY_DB`. The Dashboard does not require judge credentials.

## Testing

The repository has `tests/test_rag_pipeline.py` and `tests/test_multi_judge.py`. The RAG pipeline test file was run locally with `python3 tests/test_rag_pipeline.py` and all 12 tests passed. A clean Windows installation and test run has not yet been performed.

## Thesis context

This repository accompanies an undergraduate Computer Science thesis evaluating local language models and RAG on 103 questions based on Apple 10-K filings from fiscal years 2016–2025.

## Limitations and reproducibility

The dashboard displays stored results; installing the software does not regenerate the thesis experiments. Reproduction may require specific model versions, local hardware, an indexed document corpus, and external judge credentials. Results can vary with those conditions.

Apple filings and model files are not bundled. Check the provenance and redistribution terms of source documents and any replacement result data before sharing them.
