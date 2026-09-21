# Receipt Intelligence RAG

Hệ thống đọc hóa đơn (Kassenbon) từ ảnh/PDF, trích xuất dữ liệu có cấu trúc bằng OCR + LLM,
lưu vào database, và cho phép hỏi đáp + thống kê chi tiêu qua chatbot (kết hợp SQL + RAG).

**Team:** Duy · Dương

> File README này là bản NHÁP ban đầu — mục Roadmap ở dưới cần cả team ngồi lại chỉnh
> cho khớp lịch thực tế (số tuần, khối lượng việc, ai làm phần nào) trước khi bắt đầu.

---

## 1. Kiến trúc tổng thể

```
Ảnh/PDF Kassenbon
      │
      ▼
Tiền xử lý ảnh (OpenCV)
      │
      ▼
OCR (Tesseract / PaddleOCR)
      │
      ▼
LLM chuẩn hóa → JSON (validate bằng Pydantic)
      │
      ▼
   Database (SQLite)
      │
   ┌──┴──────────────┐
   ▼                 ▼
Thống kê        Retriever (ChromaDB)
(Python/SQL)          │
                       ▼
                  LLM Chatbot
```

Nguyên tắc quan trọng: **tính toán (tổng/trung bình/so sánh) luôn đi qua SQL/Python,
không giao cho LLM/RAG tính** — RAG chỉ dùng để tìm và diễn giải nội dung hóa đơn bằng
ngôn ngữ tự nhiên.

## 2. Tech stack

| Thành phần        | Công nghệ                    |
|--------------------|-------------------------------|
| Ngôn ngữ           | Python                        |
| Xử lý ảnh          | OpenCV                        |
| OCR                | Tesseract hoặc PaddleOCR      |
| Đọc PDF            | PyMuPDF                       |
| Data validation    | Pydantic                      |
| Data processing    | Pandas                        |
| Database           | SQLite + SQLAlchemy           |
| Vector database    | ChromaDB                      |
| Embedding          | Sentence Transformers         |
| LLM local          | Ollama                        |
| Giao diện          | Streamlit                     |
| Biểu đồ            | Plotly                        |
| Testing            | Pytest                        |
| Quản lý phiên bản  | Git / GitHub                  |

## 3. Cấu trúc thư mục

```
receipt-rag-chatbot/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw_receipts/
│   ├── processed_images/
│   └── database/
│
├── src/
│   ├── image_processing/
│   │   └── preprocess.py
│   ├── ocr/
│   │   └── receipt_ocr.py
│   ├── extraction/
│   │   ├── receipt_extractor.py
│   │   ├── schemas.py
│   │   └── validator.py
│   ├── database/
│   │   ├── models.py
│   │   └── repository.py
│   ├── rag/
│   │   ├── embeddings.py
│   │   ├── retriever.py
│   │   └── generator.py
│   ├── analytics/
│   │   └── statistics.py
│   └── chatbot/
│       └── query_router.py
│
└── tests/
    ├── test_ocr.py
    ├── test_extraction.py
    └── test_statistics.py
```

## 4. Setup môi trường

1. `git clone <url-repo>` rồi `cd receipt-rag-chatbot`
2. Tạo và kích hoạt virtual environment (`python -m venv .venv`, xem hướng dẫn kích hoạt theo OS)
3. `pip install -r requirements.txt`
4. Cài Tesseract OCR (nếu dùng `pytesseract`, đây là chương trình ngoài, không cài qua pip):
   - Windows: tải installer từ UB-Mannheim tesseract build
   - macOS: `brew install tesseract`
   - Linux: `sudo apt install tesseract-ocr`
5. Cài Ollama (https://ollama.com), sau đó pull 1 model nhỏ phù hợp máy, ví dụ:
   `ollama pull llama3.2` (điều chỉnh theo cấu hình máy thật của team)
6. Tạo file `.env` (không commit) cho các biến cấu hình nếu cần (đường dẫn DB, tên model...)

## 5. Phân chia công việc theo tuần (bản nháp)

**Vai trò:**
- **Duy** — Computer Vision: tiền xử lý ảnh, OCR, đánh giá chất lượng nhận dạng
- **Dương** — Repo/Database/IT: cấu trúc project, DB schema, môi trường, kết nối các module
- **AI (LLM extraction + RAG + chatbot)** — chia đều cho cả hai

### Tuần 1 — Setup + nền tảng OCR
- [ ] (Dương) Khởi tạo repo, cấu trúc thư mục, venv, requirements.txt, .gitignore
- [ ] (Dương) Thiết kế sơ bộ DB schema: `receipts`, `receipt_items`, `receipt_documents`
- [ ] (Duy) Thu thập một tập Kassenbon mẫu (10–20 ảnh, nhiều cửa hàng khác nhau: REWE, Lidl, Edeka...)
- [ ] (Duy) Viết pipeline tiền xử lý ảnh: grayscale, giảm nhiễu, thresholding, xoay thẳng
- [ ] (Duy) Tích hợp OCR (chốt Tesseract hay PaddleOCR trong tuần này), chạy thử trên tập mẫu, ghi chú lỗi thường gặp

### Tuần 2 — Hoàn thiện OCR + trích xuất có cấu trúc
- [ ] (Duy) Hoàn thiện OCR, đo accuracy trên tập mẫu, xử lý ảnh nghiêng/mờ/nhăn
- [ ] (Dương + Duy — chia đều) Thiết kế Pydantic schema cho JSON hóa đơn (receipt_id, store_name, items[], total...)
- [ ] (Dương + Duy — chia đều) Viết prompt cho LLM chuẩn hóa raw text → JSON, test bằng Ollama local
- [ ] (Dương) Viết validator: so khớp tổng item với tổng hóa đơn, xử lý trường bị thiếu

### Tuần 3 — Database + bắt đầu RAG
- [ ] (Dương) Implement SQLAlchemy models + repository (lưu receipt/items, chống lưu trùng hóa đơn)
- [ ] (Dương) Nối pipeline OCR → LLM extraction → DB thành 1 luồng chạy được end-to-end cho 1 ảnh
- [ ] (Duy + Dương — chia đều) Bắt đầu RAG: tạo document từ hóa đơn, chunking, embedding, lưu ChromaDB
- [ ] (Duy) Test OCR trên tập mẫu lớn hơn, fix các case còn sai, ghi lại các dạng viết tắt cần xử lý (H-MILCH, MWST, SUMME, PFAND...)

### Tuần 4 — Chatbot + query router + thống kê
- [ ] (Dương + Duy — chia đều) Retriever + Generator cho RAG, trả lời kèm trích dẫn hóa đơn nguồn
- [ ] (Dương + Duy — chia đều) Query router: phân loại câu hỏi cần SQL / cần RAG / cần cả hai
- [ ] (Dương) Viết các hàm thống kê: tổng theo ngày/tuần/tháng, theo cửa hàng, theo category, top sản phẩm
- [ ] (Duy) Xử lý normalize tên sản phẩm (raw_name → normalized_name → category) chính xác hơn

### Tuần 5 — Dashboard + evaluation + hoàn thiện
- [ ] (phân công lại tùy quỹ thời gian) Streamlit UI: upload ảnh, cho sửa dữ liệu nếu OCR sai, danh sách hóa đơn
- [ ] (người còn lại) Biểu đồ Plotly: chi tiêu theo thời gian, theo category, KPI cards
- [ ] (Cả hai) Evaluation: OCR accuracy, độ chính xác trích xuất từng trường, retrieval quality, kiểm tra câu trả lời có căn cứ
- [ ] (Cả hai) Test với hóa đơn từ nhiều cửa hàng khác nhau, README final, chuẩn bị demo

### Backlog (làm sau khi MVP ổn định, không bắt buộc)
- PDF nhiều trang, upload nhiều hóa đơn cùng lúc
- Vision Language Model (thay/song song với OCR truyền thống)
- Query router tự động hoàn toàn (thay vì rule-based)
- Phát hiện giá sản phẩm thay đổi theo thời gian
- Export CSV/Excel

## 6. Quy ước làm việc
- Branch theo dạng `feature/<tên-việc>`, review chéo trước khi merge vào `main`
- Commit message ngắn gọn, mô tả đúng thay đổi
- Không commit: `data/` (ảnh/hóa đơn thật có thể chứa thông tin cá nhân), `.env`, file DB (`*.db`)
- Mọi model/prompt quan trọng nên có 1-2 dòng ghi chú trong code hoặc `docs/` để người kia hiểu tại sao chọn vậy