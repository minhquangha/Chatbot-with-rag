import sys
from pathlib import Path

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from adapter import SentenceTransformerEmbeddings
import config


def get_embeddings() -> SentenceTransformerEmbeddings:
    """Khởi tạo mô hình embedding theo cấu hình."""
    return SentenceTransformerEmbeddings(model_name=config.EMBEDDING_MODEL)


def ingest_documents():
    """Đọc tài liệu PDF từ thư mục papers, cắt đoạn ngữ nghĩa và lưu vào FAISS vector store."""
    papers_dir = config.PAPERS_DIR

    if not papers_dir.exists():
        papers_dir.mkdir(parents=True, exist_ok=True)
        print(f"[!] Đã tạo thư mục {papers_dir}. Vui lòng sao chép các file PDF vào đây.")
        return

    pdf_files = list(papers_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"[!] Không tìm thấy file PDF nào trong thư mục: {papers_dir}")
        return

    print(f"[*] Tìm thấy {len(pdf_files)} file PDF. Đang tiến hành đọc bằng PyPDFDirectoryLoader...")
    loader = PyPDFDirectoryLoader(
        path=str(papers_dir),
        glob="*.pdf"
    )
    docs = loader.load()
    print(f"[+] Đã đọc tổng cộng {len(docs)} trang tài liệu.")

    print(f"[*] Đang tải mô hình embedding: {config.EMBEDDING_MODEL}...")
    embeddings = get_embeddings()

    print(f"[*] Đang phân đoạn ngữ nghĩa (Semantic Chunker, percentile={config.BREAKPOINT_THRESHOLD_PERCENTILE})...")
    text_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_amount=config.BREAKPOINT_THRESHOLD_PERCENTILE
    )
    splits = text_splitter.split_documents(docs)
    print(f"[+] Đã chia thành {len(splits)} chunks ngữ nghĩa.")

    print("[*] Đang lập chỉ mục vectorstore FAISS (Cosine Distance)...")
    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embeddings,
        distance_strategy=DistanceStrategy.COSINE
    )

    config.VECTORSTORE_DIR.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(config.VECTORSTORE_DIR))
    print(f"[✓] Đã lưu thành công FAISS vectorstore tại: {config.VECTORSTORE_DIR}")


if __name__ == "__main__":
    ingest_documents()
