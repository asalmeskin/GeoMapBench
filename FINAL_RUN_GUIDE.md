# Final experiment protocol

## Before Colab

1. Replace the contents of the GitHub repository with this release, commit a clean
   tree, and create tag `v1.7.0`. In both notebooks set `GIT_REF = "v1.7.0"`.
2. Keep the benchmark and GeoMapRAG corpus in Drive. They are data artifacts and
   are intentionally not duplicated inside the code ZIP.
3. Do not put `OPENROUTER_API_KEY` in GitHub or Drive files. The notebooks request
   it with a hidden prompt for each fresh runtime.

## Duplicate benchmark folders

`geomapbench-eval preflight` reads only the 23 names frozen in `SEEDS`. It prints
any sibling directory it ignored under `extra_directories_ignored`. An official
run proceeds only with 2,300 unique canonical records. This prevents Drive folders
such as `dense_land_cover_labeling (1)` from silently increasing the run to 2,500.

The command does not delete Drive content. If the extra folders still appear in
the report, move them to Trash in Drive when logged in as their owner. Keeping them
is harmless to evaluation correctness in release 1.7.0, but removing them makes the
Drive layout less confusing.

## Recommended order

1. Run `GeoMapBench_Simple_Evaluation.ipynb` with `PER_LEAF_LIMIT = 1`. Expect
   exactly 23 successful records per model. Fix any unavailable model or billing
   issue before the full run.
2. Change `PER_LEAF_LIMIT = None`, select a standard CPU runtime, and run again.
   Expect 2,300 successful records for each of eight models. Rerunning the same
   notebook/output directory skips successful IDs and retries errors.
3. Run `GeoMapBench_Final_RAG.ipynb` on a GPU runtime first with
   `PER_LEAF_LIMIT = 1`, then with `None`. It runs `base`, `base_rag`, and
   `agentic_rag` with exactly the same answer model, prompt, temperature, images,
   and record IDs. Only `agentic_rag` additionally uses the frozen agent model.
4. A report is publication-ready only when `models_reported = 8`, every suite row
   has `n = 2300`, all three RAG comparisons have `paired_record_count = 2300`,
   and every comparison has `base_only_records = rag_only_records = 0`.

If a run stops at its cost cap, raise the cap before rerunning; the cap is cumulative
for that output directory. Never use `--force` for official results.

## Model matrix rationale

The frozen matrix contains eight image-capable models from seven families. Six are
low-cost open-weight or open-family baselines, while Gemini Flash Lite and GPT Mini
act as commercial reference points. Prices in the JSON are a 2026-09-01 snapshot;
the suite checks that model IDs still exist before making paid calls and the actual
OpenRouter-reported cost is used in results.

The free Gemma endpoint may be rate limited. Its failures remain visible in the
report and are retried when the notebook is rerun; they are never counted as zero
scores.

## What the final RAG claim means

Both treatments use BGE dense text retrieval over the frozen 180k-record corpus,
followed by `BAAI/bge-reranker-base` and a strict 6,000-character context budget.
`base_rag` stops there. `agentic_rag` adds cached LLM query planning, evidence
selection, and at most one follow-up retrieval hop. Neither treatment has BM25.
Leaves without direct corpus capability coverage receive no retrieved context,
preventing irrelevant RAG from degrading visual-only tasks.

Report all three paired deltas and 95% bootstrap CIs, plus the separate covered and
uncovered-leaf deltas under `comparisons/*/rag_comparison.json`. The image index
is not used in this final experiment, so describe the method as dense text-evidence
RAG over a multimodal geospatial corpus—not as retrieved-image multimodal RAG.
