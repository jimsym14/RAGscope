# Research reproduction guide

This guide distinguishes reading the saved results from running work that creates new results. The dashboard reads SQLite; it does not launch experiments.

## Install the research environment

Use the full requirements and install Ollama separately:

```bash
python -m pip install -r requirements-research.txt
ollama serve
ollama pull llama3.1:8b
```

Copy `.env.example` to `.env` and add the credentials you will use for external judge scoring. DeepSeek uses `DEEPSEEK_API_KEY`. Azure uses `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, and optionally `AZURE_OPENAI_DEPLOYMENT` and `AZURE_OPENAI_API_VERSION`.

## Inputs and configuration locations

- Main benchmark rows: `data/thesis_benchmark_103_complete.csv`.
- Adversarial prompts: `data/adversarial_benchmark.csv`.
- Model metadata and default chat settings: `config.py`, in `MODEL_REGISTRY` and `MODEL_BENCHMARK_PRESETS`.
- Experiment configurations and core/ablation matrix: `evaluation_engine.py`, in `CONFIG_PRESETS`, `CORE_EXPERIMENT_MATRIX`, and `ABLATION_EXPERIMENT_MATRIX`.
- RAG index location: `RAGSCOPE_VECTOR_STORE`, or the default `data/vector_store/chroma`.
- Results database: `RAGSCOPE_DATA_DIR/experiments.db`, or the default `data/experiments.db`.
- Sensitivity results: `RAGSCOPE_SENSITIVITY_DB`, or the default `data/sensitivity_ablation/sensitivity_runs.db`.

To replace a benchmark, keep a backup of the current CSV, place the replacement at the same path, and preserve the columns consumed by the runner (`question`/`user_input`, `ground_truth`/`reference`, and any category fields used by the selected analysis). A changed dataset is a changed experiment; record its source, version, row count, and checksum with your run notes. Do not overwrite the included reference data if you need to preserve the published results.

To use another model, install it with `ollama pull <model>`, then update the appropriate model ID in `config.py`. For benchmark-matrix runs, also update the model entries in `CORE_EXPERIMENT_MATRIX` or `ABLATION_EXPERIMENT_MATRIX` in `evaluation_engine.py`. Review the model's context size, generation settings, and RAG preset before comparing its results.

To use another corpus, add the source PDFs through the Chat upload controls or create a separate Chroma index, then point `RAGSCOPE_VECTOR_STORE` to that index. Keep corpus and embedding-model versions fixed within a comparison.

## Available command line entry point

The adversarial benchmark is the only experiment runner in this release with a command-line interface:

```bash
python adversarial_benchmark.py --summary
python adversarial_benchmark.py --model llama3.1:8b --config CFG-3
```

Valid configuration IDs are `CFG-1` through `CFG-4`. `--run-all` runs the adversarial model/configuration matrix. These commands write to the experiment database selected by `RAGSCOPE_DATA_DIR` and require the local model, RAG index, and source benchmark to be ready.

## Current limitation: core experiment and judge launch

The code defines `ExperimentRunner`, matrix definitions, checkpoint/resume behavior, and `score_experiment()` in `evaluation_engine.py`. It does not currently expose a supported CLI or UI action that launches the full core/ablation matrix or judge scoring. Do not assume `python evaluation_engine.py` will run them. To reproduce those phases, call the Python APIs from a purpose-built driver or implement a driver that selects a matrix row, loads its dataset/model/retriever, invokes the runner, and records configuration and outputs. Review `ExperimentRunner` and `build_evaluation_retriever_and_engine()` together before writing that driver; the parameters and execution lifecycle must match the thesis experiment protocol.

The dashboard's stored judge scores are read-only. A configured API key only reports readiness; it does not start scoring.

## Verify a run

1. Record dataset path, checksum, row count, model ID/version, RAG configuration, and judge provider.
2. Keep the generated checkpoint and database separate from the included reference files.
3. Compare the number of completed questions with the planned sample size.
4. Open the resulting database in RAGscope and confirm that the intended judge's scores appear under that provider.
5. Keep machine, software, and timing conditions with the run notes; latency values are hardware-dependent.
