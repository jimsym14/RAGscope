# Data guide

## Included result databases

`data/experiments.db` is the dashboard's pre-populated main result database. It contains experiment records, detailed generated answers and contexts, judge scores/summaries, and adversarial results. Do not replace it with an empty database if you want the public dashboard to show the thesis results.

`data/sensitivity_ablation/sensitivity_runs.db` supplies the sensitivity section in the Ablation view. The view expects 180 joined generation/judge rows across nine cells. If the file is absent, only this section reports that its data is unavailable.

The two CSV files in `data/` are benchmark inputs; `test_prompts.json` contains sample chat prompts. Keep benchmark inputs distinct from result databases.

## Runtime data

Chat history is stored in `data/chats.db`. It is created locally and excluded from the repository. Experiment checkpoints and generated Chroma indexes are also local runtime data. `RAGSCOPE_DATA_DIR` moves runtime data to another directory; the included dashboard database files are not copied there automatically, so copy them if you want the populated dashboard while using a custom data directory.

`RAGSCOPE_VECTOR_STORE` selects an existing Chroma index. The repository does not bundle a vector index or source 10-K PDFs. To build an index, install `requirements-rag.txt`, start Chat, and upload PDFs. Do not point the app at a Chroma database while another process is writing to it.

## Replacing or backing up results

Before replacing a database, stop Streamlit and make a copy. Validate a replacement with SQLite's `PRAGMA quick_check`, then open the Evaluation page and inspect each tab. Keep a copy of the reference database so generated results do not overwrite the published snapshot.

Do not publish chat histories, API credentials, raw user uploads, local checkpoints, or model caches. Review all generated answers and contexts for material that should not be redistributed before publishing a different database.
