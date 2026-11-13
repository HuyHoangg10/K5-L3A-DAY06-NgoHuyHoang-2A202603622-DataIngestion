# Student pipeline — EmbeddingGemma 2 + Qdrant

Implementation cuối dùng Python, EmbeddingGemma 2 local và Qdrant local persistent mode. `docker-compose.yml`, `init.sql` và `scripts/check_setup.py` được giữ lại chỉ như PostgreSQL + pgvector baseline tùy chọn của đề; chúng không được dùng để tạo kết quả trong báo cáo.

## Thành phần

```text
src/ingestion.py       đọc PDF theo trang và chuẩn hóa Unicode
src/chunking.py        fixed-size A và structure-aware B/C
src/embeddings.py      EmbeddingGemma 2, query/document prompts
src/store.py           Qdrant persistence, cosine search, filter, stable ID
src/evaluation.py      Hit@1 và Evidence Hit@3 nghiêm ngặt
src/cp3.py             extraction/chunking checkpoint
src/cp4.py             embedding/vector-store checkpoint
src/cp5.py             benchmark/trace/audit candidates
src/validate_cp5.py    validator không cần tải lại model
src/run.py             entry point chung
```

## Chạy

Từ root repo, sau khi cài `starter/requirements.txt`, đặt model cache:

```powershell
$env:MODEL_CACHE = "D:\path\to\model_cache"
$env:HF_HOME = $env:MODEL_CACHE
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
```

Chạy từng bước hoặc toàn bộ:

```powershell
.\.venv\Scripts\python.exe -m starter.src.cp3
.\.venv\Scripts\python.exe -m starter.src.cp4
.\.venv\Scripts\python.exe -m starter.src.cp5
.\.venv\Scripts\python.exe -m starter.src.run --stage validate
.\.venv\Scripts\python.exe -m starter.src.run --stage all
```

`--stage all` sẽ embed lại toàn corpus và mất khoảng 10 phút trên CPU đã đo. `--stage validate` chỉ kiểm tra benchmark/artifact hiện có.

Hướng dẫn tải model, corpus, quyền dữ liệu và kết quả đầy đủ xem [README root](../README.md).
