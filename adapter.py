from typing import List, Optional
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings


class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper cho mô hình SentenceTransformer tương thích với chuẩn LangChain Embeddings."""

    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Nhúng danh sách các đoạn văn bản thành vector embeddings."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        """Nhúng một câu truy vấn thành vector embedding."""
        return self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()
