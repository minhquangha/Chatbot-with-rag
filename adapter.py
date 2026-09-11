from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings


class SentenceTransformerEmbeddings(Embeddings):

    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()

    def embed_query(self, text):
        return self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()
