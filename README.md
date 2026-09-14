# Chatbot RAG cho Tài liệu Nghiên cứu (Research Papers)

Hệ thống Chatbot hỏi đáp dựa trên tài liệu (RAG - Retrieval-Augmented Generation) xây dựng với **LangChain**, mô hình nhúng đa ngôn ngữ **Sentence Transformers**, phân đoạn ngữ nghĩa **Semantic Chunking**, và lưu trữ chỉ mục bằng **FAISS**.

---

## 🛠️ Công nghệ cốt lõi

- **Ngôn ngữ:** Python 3.12+ (hỗ trợ quản lý gói qua `uv`)
- **Framework RAG:** LangChain Core, LangChain Community, LangChain Experimental
- **Document Loader:** `PyPDFDirectoryLoader` (bảo toàn số trang phục vụ trích dẫn)
- **Text Splitter:** `SemanticChunker` (phân đoạn dựa trên độ biến thiên ngữ nghĩa)
- **Embedding Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (hỗ trợ tốt Tiếng Việt & Đa ngôn ngữ)
- **Vector Database:** FAISS (CPU, Cosine Distance)
- **LLM Engine:** DeepSeek (`deepseek-v4-flash-0731` qua Alibaba Cloud MaaS hoặc bất kỳ OpenAI-compatible endpoint nào)

---

## 📁 Cấu trúc dự án

```text
Chatbot-with-rag/
├── papers/                 # Thư mục chứa các tài liệu PDF cần đọc
│   ├── research_paper_1.pdf
│   └── ...
├── adapter.py              # Adapter nhúng cho Sentence Transformers
├── config.py               # Quản lý tập trung toàn bộ cấu hình & biến môi trường
├── load.py                 # Pipeline đọc PDF, semantic chunking và lưu vectorstore
├── chatbot.py              # Chương trình hỏi đáp RAG với trích dẫn chi tiết
├── .env.example            # File mẫu cấu hình biến môi trường
├── pyproject.toml          # Danh sách dependencies (chuẩn uv / pyproject)
└── README.md               # Tài liệu hướng dẫn sử dụng
```

---

## 🚀 Hướng dẫn cài đặt & Chạy

### 1. Cài đặt môi trường với `uv`

Nếu chưa có `uv`, cài đặt qua:
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Cài đặt toàn bộ dependencies vào môi trường ảo:
```bash
uv sync
```

### 2. Cấu hình biến môi trường

Tạo file `.env` từ file mẫu `.env.example`:
```bash
cp .env.example .env
```

Điền API Key của bạn vào `.env`:
```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
LLM_MODEL=deepseek-v4-flash-0731
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

### 3. Nạp và lập chỉ mục tài liệu (Ingestion)

Đặt các file PDF của bạn vào thư mục `papers/`, sau đó chạy:
```bash
uv run python load.py
```
Quá trình này sẽ:
1. Đọc từng trang tài liệu PDF.
2. Phân đoạn ngữ nghĩa dựa trên ngưỡng percentile (mặc định 85%).
3. Tạo vector embeddings và lưu vào thư mục cục bộ `vectorstore_cache/`.

### 4. Khởi động Chatbot

Chạy chatbot tương tác trên dòng lệnh:
```bash
uv run python chatbot.py
```

* Nhập câu hỏi và nhấn `Enter`.
* Để thoát chương trình, gõ `exit`, `quit` hoặc nhấn tổ hợp phím `Ctrl + C`.

---

## 📌 Lưu ý về bảo mật & Git

- Thư mục `vectorstore_cache/` và file cấu hình `.env` đã được cấu hình trong `.gitignore` để không bị push lên kho mã nguồn từ xa.
- Không chia sẻ công khai file `.env` chứa API Key.
