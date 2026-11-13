# CP3 outputs

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m starter.src.cp3
```

Versionable evidence:

- `summary.json`: corpus, normalization, configuration, chunk statistics, cost, and validation status.
- `page_audit.jsonl`: one metadata-only audit record per physical PDF page.
- `extraction_issues.json`: empty/error pages and extraction failures; an empty array means none were observed.

Local full-text derivatives:

- `pages.local.jsonl`
- `chunks_A.local.jsonl`
- `chunks_B.local.jsonl`

The full-text derivatives are ignored by Git because the corpus policy in `data/manifest.json` does not assume third-party redistribution rights. They can be reproduced deterministically from the locally downloaded PDFs after verifying each SHA-256 hash.
