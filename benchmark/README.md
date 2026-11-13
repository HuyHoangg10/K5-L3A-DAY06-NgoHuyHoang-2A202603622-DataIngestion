# CP2 — Gold benchmark

This benchmark was frozen before retrieval experiments. It contains 20 answerable questions drawn from the four papers in `data/manifest.json`, balanced at five questions per paper.

## Fixed split

- Development: 15 questions (`q01`–`q04`, `q06`–`q09`, `q11`–`q14`, `q16`–`q18`).
- Test: 5 questions (`q05`, `q10`, `q15`, `q19`, `q20`).
- Test questions must not be used to tune chunking, metadata, embedding, indexing, or retrieval settings.
- All strategies A/B/C must use this exact question file and split.

The set covers direct lookup, paraphrase, comparison, architecture, formula, configuration, condition, efficiency, and updateability questions. Each paper contributes four development questions and one test question, except RAG, which contributes three development and two test questions; the total corpus contribution remains five questions per paper.

## Evidence policy

Every `evidence_quote` is copied from the normalized text of its `gold_source_id` and `gold_page`. Normalization collapses consecutive whitespace but deliberately does not repair PDF extraction artifacts such as split hyphenation, ligatures, or missing spaces. This makes evidence validation deterministic and prevents a manually cleaned quote from silently ceasing to match the source.

`validate_benchmark.py` checks the required schema, unique IDs, fixed split, source and page validity, inclusion of the page in the CP1 corpus, exact normalized evidence containment, and filter validity.

## Filter policy

Four questions have a metadata filter, including one test question. Every filter value is explicitly supplied by the wording of the question before retrieval:

- `q01`: `venue = NAACL-HLT 2019`
- `q06`: `venue = EMNLP-IJCNLP 2019`
- `q11`: `year = 2020`
- `q19`: `venue = NeurIPS 2020`

The filter contains only user-visible scope metadata, never the answer or hidden gold source ID. Questions without `filter` must be run without a filter in strategy C.

## Strategy selection rule fixed before test

Tune only on the development split. Select the final strategy by the following predeclared order:

1. highest development Evidence Hit@3;
2. if tied, highest development Hit@1;
3. if still tied, higher mean manual grounding score;
4. if still tied, higher mean chunk-coherence score;
5. if still tied, lower indexing time, storage, and chunk count.

Run the five-question test split once after selection and report it separately. Do not revise the strategy after seeing test results; if that rule is broken, document the leak and create a new test set.

## Validation

From the repository root:

```powershell
.\.venv\Scripts\python.exe benchmark\validate_benchmark.py
```
