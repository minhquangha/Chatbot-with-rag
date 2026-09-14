import os
from pathlib import Path
from dotenv import load_dotenv

# Tự động nạp biến môi trường từ .env nếu có
load_dotenv()

# Đường dẫn thư mục chính
BASE_DIR = Path(__file__).resolve().parent
PAPERS_DIR = BASE_DIR / "papers"
VECTORSTORE_DIR = BASE_DIR / "vectorstore_cache"

# Cấu hình Mô hình Embedding
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", 
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Cấu hình LLM
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash-0731")
LLM_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.0))

# Tham số RAG & Chunking
RETRIEVER_K = int(os.getenv("RETRIEVER_K", 5))
BREAKPOINT_THRESHOLD_PERCENTILE = int(os.getenv("BREAKPOINT_THRESHOLD_PERCENTILE", 85))
