# CP5 outputs

Run after CP4 from the repository root:

```powershell
$env:MODEL_CACHE = "D:\path\to\model_cache"
$env:HF_HOME = $env:MODEL_CACHE
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.\.venv\Scripts\python.exe -m starter.src.cp5
```

Versionable artifacts:

- `summary.json`: frozen inputs, retrieval counts, metrics, filter and audit scope.
- `metrics.json`: Hit@1 and Evidence Hit@3 by strategy and dev/test split.
- `retrieval_trace_index.jsonl`: complete query/rank/score/metadata/evidence flags without copyrighted chunk text.
- `metadata_audit.jsonl`: per-result metadata cross-check for all 45 test results.
- `manual_audit.json`: disclosed assisted coherence/grounding review, reasons, distributions, and selection rule.

Local reproducible artifacts ignored by Git:

- `retrieval_traces.local.jsonl`: full 180-result trace including text.
- `coherence_candidates.local.jsonl`: 20 full-text candidates for ten fixed positions across A/B.
- `grounding_candidates.local.jsonl`: full top-3 text for five test questions across A/B/C.

`manual_audit.json` records the coherence and grounding scores with a reason for each label. The frozen configurations were not revised after viewing test results.

Validate all trace hashes, recomputed metrics, filter compliance, metadata checks, and audit arithmetic without reloading the embedding model:

```powershell
.\.venv\Scripts\python.exe -m starter.src.validate_cp5
```
