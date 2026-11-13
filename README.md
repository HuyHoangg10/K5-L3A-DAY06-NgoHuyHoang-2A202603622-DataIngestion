# Day 06 — Data Ingestion Engineering

**Sinh viên:** Ngô Huy Hoàng — `2A202603622` — L3A

**Chủ đề:** dense semantic retrieval và retrieval-augmented generation

**Trạng thái:** CP1–CP5 đã chạy; CP6 đã tổng hợp code, hướng dẫn và báo cáo. Chưa commit/push bản chốt.

Pipeline đọc bốn research paper, chuẩn hóa 45 trang nội dung, so sánh ba chiến lược chunk/retrieval bằng EmbeddingGemma 2 và Qdrant local, rồi đánh giá trên cùng benchmark 15 dev / 5 test.

## Kết quả tóm tắt

| Strategy | Chunking / filter | Chunks | Dev Hit@1 | Dev Evidence Hit@3 | Test Hit@1 | Test Evidence Hit@3 |
|---|---|---:|---:|---:|---:|---:|
| A | 500 ký tự, overlap 50, không filter | 416 | 8/15 | 9/15 | 0/5 | 3/5 |
| B | theo đoạn/câu, tối đa 1.000 ký tự | 253 | 5/15 | 9/15 | 2/5 | 2/5 |
| C | chunk/vector giống B; filter khi câu hỏi cung cấp | 253 | 5/15 | 9/15 | 2/5 | 2/5 |

Theo tiêu chí đã khóa trước khi đọc test, A được chọn vì ba cấu hình hòa Dev Evidence Hit@3 và A có Dev Hit@1 cao hơn. B giữ ý mạch lạc hơn (2,0/2 so với 1,0/2 của A). C đạt 12/12 kết quả tuân thủ filter nhưng không đổi Hit@1/Hit@3 so với B trên bộ câu hỏi nhỏ này.

## Cấu trúc quan trọng

```text
data/manifest.json                 nguồn, quyền sử dụng, trang, SHA-256
benchmark/questions.json           20 câu, dev/test, gold evidence
starter/src/                       ingestion, chunking, embedding, store, evaluation
starter/results/cp3/               log đọc PDF và thống kê chunk
starter/results/cp4/               model/store/persistence và smoke top-3
starter/results/cp5/               metrics, trace index và audit
artifacts/experiment_log.csv        giả thuyết và kết quả A/B/C
report/individual/                  báo cáo cá nhân
```

PDF, full-text chunk/trace, model cache và Qdrant storage được giữ local, không commit. Các file versionable chứa manifest, hash, metadata, metrics và hướng dẫn tái tạo.

## 1. Cài môi trường

Yêu cầu Python 3.11–3.13. Các lệnh dưới đây chạy tại root repo bằng PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r starter\requirements.txt
```

Không cần OpenAI/Gemini API key. Bài dùng model local và chi phí API bằng `$0`.

### Tải EmbeddingGemma 2

Chọn một thư mục cache ngoài Git, sau đó tải model một lần:

```powershell
$env:MODEL_CACHE = "D:\path\to\model_cache"
$env:HF_HOME = $env:MODEL_CACHE
.\.venv\Scripts\hf.exe download google/embeddinggemma-2 --cache-dir "$env:MODEL_CACHE"
```

Nếu Hugging Face yêu cầu, đăng nhập và chấp nhận điều khoản/model access trước khi tải. Các lần chạy pipeline sau đặt chế độ offline:

```powershell
$env:MODEL_CACHE = "D:\path\to\model_cache"
$env:HF_HOME = $env:MODEL_CACHE
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
```

Model được dùng ở `float32`, text-only, 768 chiều, L2-normalized; query dùng prompt `QuestionAnswering`, document dùng `title: ... | text: ...`.

## 2. Tái tạo corpus

Corpus gồm bốn bản PDF cố định phiên bản. Tải về đúng tên file:

```powershell
New-Item -ItemType Directory -Force data\papers
Invoke-WebRequest https://arxiv.org/pdf/1810.04805v2 -OutFile data\papers\bert-1810.04805.pdf
Invoke-WebRequest https://arxiv.org/pdf/1908.10084v1 -OutFile data\papers\sentence-bert-1908.10084.pdf
Invoke-WebRequest https://arxiv.org/pdf/2004.04906v3 -OutFile data\papers\dense-passage-retrieval-2004.04906.pdf
Invoke-WebRequest https://arxiv.org/pdf/2005.11401v4 -OutFile data\papers\retrieval-augmented-generation-2005.11401.pdf
```

`data/manifest.json` ghi URL, license/permission, trang được tính, trang loại bỏ và SHA-256. CP3 sẽ dừng ngay nếu hash, số trang hoặc số ký tự không khớp.

## 3. Chạy pipeline

Chạy từng checkpoint:

```powershell
.\.venv\Scripts\python.exe -m starter.src.cp3
.\.venv\Scripts\python.exe -m starter.src.cp4
.\.venv\Scripts\python.exe -m starter.src.cp5
.\.venv\Scripts\python.exe -m starter.src.validate_cp5
```

Hoặc dùng entry point chung:

```powershell
# Chỉ kiểm tra artifact hiện có, không embed lại
.\.venv\Scripts\python.exe -m starter.src.run --stage validate

# Chạy lại toàn bộ; CP4 mất khoảng 10 phút trên CPU của máy thí nghiệm
.\.venv\Scripts\python.exe -m starter.src.run --stage all
```

Các stage riêng cũng hỗ trợ `--stage cp3`, `cp4`, hoặc `cp5`. Qdrant chạy ở local persistent mode trong `starter/qdrant_storage/`; repo đề bài cho phép Qdrant thay cho PostgreSQL + pgvector.

## 4. Benchmark và cách tính

- 20 câu được khóa trước retrieval: 15 dev, 5 test.
- Bốn câu có metadata filter; một câu nằm trong test.
- Hit@1/Evidence Hit@3 chỉ đạt khi đúng source, đúng trang và chứa đầy đủ gold evidence sau chuẩn hóa whitespace/Unicode.
- Audit grounding có thể công nhận evidence tương đương nếu đọc thủ công và ghi lý do.
- Test không được dùng để điều chỉnh chunk, model hoặc filter.
- Tiêu chí chọn: Dev Evidence Hit@3 → Dev Hit@1 → grounding → coherence → tài nguyên.

Chi tiết nằm trong [benchmark/README.md](benchmark/README.md), [metrics.json](starter/results/cp5/metrics.json) và [manual_audit.json](starter/results/cp5/manual_audit.json).

## 5. Quyền dữ liệu và khả năng tái lập

Sentence-BERT dùng CC BY-SA 4.0; ba PDF còn lại được xử lý thận trọng theo arXiv non-exclusive distribution license. Vì vậy PDF và derivative full-text không được commit. Compact trace chứa ID, score, metadata, evidence flags và hash văn bản; người chấm có thể tải lại PDF và tái tạo full trace bằng các lệnh trên.

Manual audit lưu nhãn coherence và grounding kèm lý do cho từng trường hợp. Cấu hình và metric không được chỉnh lại sau khi xem tập test.

## Tài liệu gốc của đề

- [GUIDE.md](GUIDE.md)
- [BENCHMARK.md](BENCHMARK.md)
- [CHECKPOINTS.md](CHECKPOINTS.md)
- [RUBRIC.md](RUBRIC.md)
- [RULES.md](RULES.md)
- [SUBMISSION.md](SUBMISSION.md)
