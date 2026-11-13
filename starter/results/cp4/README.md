# CP4 outputs

CP4 uses the locally cached `google/embeddinggemma-2` snapshot and Qdrant local persistent mode. It does not call an embedding API.

Run from the repository root after CP3:

```powershell
$env:MODEL_CACHE = "D:\path\to\model_cache"
$env:HF_HOME = $env:MODEL_CACHE
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.\.venv\Scripts\python.exe -m starter.src.cp4
```

Outputs:

- `summary.json`: model, vector-store, timing, count, cost, reload, and idempotency evidence.
- `smoke_top3.json`: one fixed smoke query with top-3 results from A/B/C and a trusted venue filter for C.
- `starter/qdrant_storage/`: persistent collections `day6_a`, `day6_b`, and `day6_c`; ignored by Git and reproducible from CP3 chunks.

Strategy C stores the exact same chunks and vectors as B. Its only experimental change is applying an allow-listed metadata filter when the question or context supplies that filter before retrieval.
