# CP6 submission checklist

## Hoàn thành bằng code/artifact

- [x] Corpus có 45 trang nội dung hợp lệ, manifest ghi nguồn/quyền/page ranges/hash.
- [x] Benchmark 20 câu, split 15 dev / 5 test, evidence được validator kiểm chứng.
- [x] Embedding model thật `google/embeddinggemma-2`, 768d, local, cùng model cho A/B/C.
- [x] Qdrant persistent cosine search và metadata filter.
- [x] Ba cấu hình A/B/C; B–C chỉ khác filter.
- [x] Full local top-3 trace 180 kết quả và compact versionable trace index.
- [x] Hit@1/Evidence Hit@3 dev/test, metadata audit, filter compliance.
- [x] Coherence và grounding audit có nhãn và lý do cho từng trường hợp.
- [x] README setup/run, experiment log, báo cáo cá nhân và validator.
- [x] PDF, model cache, Qdrant storage và derivative full text được `.gitignore`.
- [x] Không có API key hoặc secret trong source/artifact versionable.

## Còn lại trước khi nộp

- [x] Đã kiểm tra `starter/results/cp5/manual_audit.json`.
- [ ] Kiểm tra `git diff` và file sẽ commit.
- [ ] Commit/push bản cuối.
- [ ] Điền commit SHA thật vào `STUDENT.md`.
- [ ] Mở repo trên GitHub và kiểm tra quyền truy cập.
