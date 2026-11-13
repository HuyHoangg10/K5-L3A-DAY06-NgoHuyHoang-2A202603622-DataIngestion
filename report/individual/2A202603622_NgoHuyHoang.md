# Báo cáo cá nhân — Day 06 Data Ingestion Engineering

- **Sinh viên:** Ngô Huy Hoàng
- **MSSV:** 2A202603622
- **Lớp:** L3A
- **Repo:** <https://github.com/HuyHoangg10/K5-L3A-DAY06-NgoHuyHoang-2A202603622-DataIngestion>

## 1. Dữ liệu và pipeline

### 1.1 Corpus

Em chọn bốn paper mô tả quá trình phát triển từ biểu diễn ngôn ngữ đến RAG:

1. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* — arXiv `1810.04805v2`.
2. *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks* — arXiv `1908.10084v1`.
3. *Dense Passage Retrieval for Open-Domain Question Answering* — arXiv `2004.04906v3`.
4. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* — arXiv `2005.11401v4`.

Corpus có 59 trang PDF vật lý. Sau khi loại các trang chỉ chứa references hoặc acknowledgements, còn 45 trang nội dung hợp lệ, vượt yêu cầu tối thiểu 30 trang.

Sentence-BERT dùng giấy phép CC BY-SA 4.0. Ba paper còn lại theo arXiv non-exclusive distribution license nên em chỉ giữ PDF local và không commit lên repo. [Manifest](../../data/manifest.json) lưu URL phiên bản, giấy phép, page range, số ký tự và SHA-256 để có thể tải lại và kiểm tra.

### 1.2 Đọc và chuẩn hóa

Pipeline dùng `pypdf 6.21.4`, đọc theo từng trang và giữ số trang 1-based. Text được chuẩn hóa Unicode NFC, bỏ control character, thu gọn khoảng trắng ngang và giữ ranh giới đoạn. Em không tự động dehyphenate vì không đủ thông tin layout để phân biệt dấu gạch nối thật với dấu gạch nối do xuống dòng.

Kết quả CP3:

- 4/4 file khớp SHA-256 trong manifest.
- Audit đủ 59 trang, trong đó 45 trang được ingest.
- Không có trang rỗng hoặc lỗi extraction.
- Tổng cộng 178.315 ký tự sau chuẩn hóa.

### 1.3 Embedding và vector store

- Model: `google/embeddinggemma-2`.
- Revision: `f9972ac22e696a105c482202646979e33e4b5a1e`.
- Chạy local bằng SentenceTransformers, text-only, `float32`, vector 768 chiều và L2 normalization.
- Query dùng prompt `QuestionAnswering`; document có dạng `title: {title} | text: {chunk text}`.
- Vector store: Qdrant local persistent, `qdrant-client 1.20.0`.
- Metric: cosine, exact local search.
- Chi phí API: `$0`.

Máy không có Docker nên em dùng Qdrant thay cho PostgreSQL + pgvector. Sau khi mở lại store và upsert bằng stable ID, số point vẫn là A=416, B=253 và C=253, không bị nhân đôi.

### 1.4 Metadata

Mỗi chunk giữ `document_id`, `source`, `page`, `title`, `year`, `venue`, `method`, `document_type`, `language`, `corpus_id`, strategy, chunk index và char offsets. `source/page` dùng để truy vết; `year`, `venue`, `method`, `document_type` và `language` dùng cho filter. Các giá trị đều lấy từ manifest.

## 2. Cấu hình và benchmark

### 2.1 Ba chiến lược

- **A:** fixed-size 500 ký tự, overlap 50, không filter và không vượt ranh giới trang.
- **B:** gom theo đoạn/câu, tối đa 1.000 ký tự, không overlap, không filter và không vượt ranh giới trang.
- **C:** dùng chunk và vector giống B, chỉ thêm metadata filter khi câu hỏi đã cung cấp filter trước retrieval.

A–B dùng để so sánh chunking. B–C chỉ khác filter nên có thể tách riêng ảnh hưởng của metadata.

### 2.2 Benchmark

[Bộ benchmark](../../benchmark/questions.json) gồm 20 câu, chia trước thành 15 dev và 5 test, mỗi paper có 5 câu. Có 4 câu sử dụng filter `venue` hoặc `year`, trong đó q19 thuộc tập test. Mỗi câu đều có expected answer, source, page và evidence quote; validator đã kiểm chứng 20/20 evidence.

Cùng một corpus, model và metric được dùng cho A/B/C. Test không được dùng để chỉnh cấu hình. Tiêu chí chọn chiến lược được chốt trước khi chạy test:

1. Dev Evidence Hit@3 cao nhất.
2. Nếu hòa, chọn Dev Hit@1 cao nhất.
3. Nếu vẫn hòa, lần lượt xét grounding, coherence và tài nguyên.

## 3. Kết quả

### 3.1 Retrieval accuracy và tài nguyên

| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Thời gian embedding | Chi phí |
|---|---|---:|---:|---:|---:|---:|
| A | Dev | 8/15 (53,33%) | 9/15 (60,00%) | 416 | 300,695 s | $0 |
| A | Test | 0/5 (0,00%) | 3/5 (60,00%) | 416 | dùng cùng index | $0 |
| B | Dev | 5/15 (33,33%) | 9/15 (60,00%) | 253 | 259,052 s | $0 |
| B | Test | 2/5 (40,00%) | 2/5 (40,00%) | 253 | dùng cùng index | $0 |
| C | Dev | 5/15 (33,33%) | 9/15 (60,00%) | 253 | tái dùng vector B | $0 |
| C | Test | 2/5 (40,00%) | 2/5 (40,00%) | 253 | dùng cùng index | $0 |

CP4 mất 580,857 giây trên CPU, gồm load model, embedding A/B, persistent sync và smoke query. CP5 mất 2,564 giây để embed 20 query và 10,716 giây cho toàn bộ benchmark.

### 3.2 Audit thủ công

| Strategy | Coherence TB /2 (10 vị trí) | Metadata đủ/đúng | Filter tuân thủ | Grounding 0/1/2 và TB /2 |
|---|---:|---:|---:|---:|
| A | 1,0/2 | 15/15 | N/A | 1/1/3; 1,4/2 |
| B | 2,0/2 | 15/15 | N/A | 1/1/3; 1,4/2 |
| C | 2,0/2 vì dùng chunk B | 15/15 | test 3/3; toàn bộ 12/12 | 1/1/3; 1,4/2 |

Metadata audit kiểm tra 45 record và tất cả đều khớp manifest. Filter C tuân thủ 4/4 câu, tương ứng 12/12 kết quả top-3. B và C có accuracy giống nhau vì ở các câu có filter, B vốn đã truy xuất đúng phạm vi.

Coherence được chấm trên cùng 10 vị trí dev đã cố định trước retrieval. A thường cắt mất đầu hoặc cuối câu, còn B giữ ý và section boundary tốt hơn.

Grounding được chấm trên top-3 của toàn bộ 5 câu test. q10 của B/C không trùng gold quote tại trang 4 nhưng rank 1 ở trang 3 có evidence tương đương, nên vẫn được chấm 2/2. q19 không truy xuất được đoạn giải thích thành phần nào được giữ cố định và thành phần nào được fine-tune nên cả ba chiến lược đều nhận 0.

Các artifact chính:

- [Metrics](../../starter/results/cp5/metrics.json)
- [Compact retrieval trace](../../starter/results/cp5/retrieval_trace_index.jsonl)
- [Metadata audit](../../starter/results/cp5/metadata_audit.jsonl)
- [Manual audit](../../starter/results/cp5/manual_audit.json)

Em dùng AI hỗ trợ tổng hợp manual audit và tự kiểm tra lại từng score cùng lý do.

## 4. Phân tích

### 4.1 Retrieval

Dev Evidence Hit@3 của cả ba chiến lược đều là 9/15. A có Dev Hit@1 cao nhất với 8/15, cho thấy chunk ngắn giúp tín hiệu truy vấn tập trung hơn trong corpus này. Tuy nhiên trên 5 câu test, A không có Hit@1 nào nhưng vẫn đạt Hit@3 3/5, tức evidence thường nằm ở rank 2–3. Tập test nhỏ nên chưa thể kết luận A luôn tốt hơn.

### 4.2 Coherence

B đạt 2,0/2, cao hơn A là 1,0/2. Ví dụ ở q03, A cắt mất phần cuối của tỷ lệ 80/10/10, còn B giữ đủ động cơ, ba trường hợp và loss. Đổi lại, chunk B dài hơn nên có thể chứa nhiều ý và làm vector kém tập trung.

### 4.3 Metadata

`document_id/source/page` giúp truy về đúng paper và trang; `venue/year` giúp giới hạn phạm vi mà không dùng gold source. Hạn chế là metadata mới ở cấp document, chưa mô tả section. q19 cũng cho thấy filter đúng không bảo đảm semantic ranking sẽ lấy đúng đoạn cần thiết.

### 4.4 Grounding

Cả ba chiến lược có grounding trung bình 1,4/2 nhưng lỗi khác nhau. A tìm được formal definition của q15 ở rank 3; B/C có evidence đầy đủ hơn cho q10 và q20. q19 là lỗi chung vì top-3 có nội dung liên quan tới RAG nhưng chưa đủ để trả lời câu hỏi cụ thể.

### 4.5 Tác động của chiến lược dữ liệu

- **A–B:** B giảm số chunk từ 416 xuống 253 và tăng coherence, nhưng Dev Hit@1 giảm từ 8/15 xuống 5/15.
- **B–C:** filter hoạt động đúng nhưng không tăng accuracy. Corpus chỉ có bốn paper và câu hỏi đã nêu rõ venue/year nên B thường đã tìm đúng phạm vi.

## 5. Kết luận và giới hạn

Theo rule chốt trước, em chọn **A**: cả ba hòa Dev Evidence Hit@3 9/15, sau đó A thắng ở Dev Hit@1 với 8/15. Nếu ưu tiên chunk dễ đọc trong ứng dụng thực tế, B/C vẫn có thể phù hợp hơn.

Hai lỗi tiêu biểu:

- **q19:** cả A/B/C không lấy được evidence ở RAG trang 4. C lọc đúng paper nhưng vẫn xếp hạng sai đoạn, nên đây là lỗi semantic ranking chứ không phải lỗi metadata.
- **q03:** A cắt evidence qua ranh giới window, còn B lấy được ý đầy đủ ở rank 3. Đây là lỗi chunk boundary.

Giới hạn của thí nghiệm:

- Corpus chỉ có 4 paper tiếng Anh và 5 câu test.
- PDF extraction vẫn còn lỗi line-end hyphen và khoảng trắng.
- Một số câu chỉ có một gold quote dù evidence tương đương có thể nằm ở trang khác.
- Manual audit chỉ do một người kiểm tra nên vẫn có thể mang thiên lệch người chấm.
- Thời gian chỉ được đo một lần trên CPU Windows.
- Chưa đánh giá approximate search hoặc corpus lớn.
- Các câu filter chưa đủ khó để tạo khác biệt rõ giữa B và C.

Nếu làm lại, em sẽ cải thiện parser và dehyphenation, bổ sung section metadata, ghi nhiều gold evidence hợp lệ, tăng test set, nhờ thêm người chấm audit và thử hybrid dense + BM25.

