# OMP_NUM_THREADS=1 is also set defensively in atkv/__init__.py. Repeated here
# so it is visible to anyone reading the Makefile: faiss and torch each bundle
# an OpenMP runtime, and on macOS loading a torch model after faiss has used
# its runtime segfaults the process with no traceback.
export OMP_NUM_THREADS = 1
UV = uv run

.PHONY: help ingest serve eval eval-retrieval eval-ablation test fmt clean

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | column -t -s "$$(printf '\t')"

ingest:  ## Fetch sources, build the index (cached after the first run)
	$(UV) python -c "from pathlib import Path; from atkv.ingest.build import build_corpus; \
	from atkv.retrieve.pipeline import RetrievalPipeline; \
	c = build_corpus(Path('.')); p = RetrievalPipeline.build(c); \
	p.save(Path('data/index/serving')); print(f'indexed {len(c)} chunks')"

serve:  ## Run the API on :8000 (docs at /docs)
	$(UV) uvicorn atkv.api:app --host 0.0.0.0 --port 8000

test:  ## Ground-truth and unit tests (offline, fast)
	$(UV) pytest evals/ tests/ -q

eval: test eval-retrieval  ## Everything the README reports

eval-retrieval:  ## Dense vs lexical vs hybrid vs reranked
	$(UV) python scripts/eval_retrieval.py

eval-ablation:  ## Stratified pooling and query translation
	$(UV) python scripts/eval_ablation.py

clean:  ## Remove the built index (keeps downloaded sources)
	rm -rf data/index
