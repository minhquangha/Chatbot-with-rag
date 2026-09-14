# load.py
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from adapter import SentenceTransformerEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embeddings = SentenceTransformerEmbeddings(model_name=MODEL_NAME)

def ingest_documents():
    loader = DirectoryLoader(
        path="./papers",
        glob="**/*.pdf",
        loader_cls=UnstructuredFileLoader,
        show_progress=True,
        use_multithreading=True
    )
    docs = loader.load()

    text_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_amount=85
    )
    splits = text_splitter.split_documents(docs)
    
    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embeddings,
        distance_strategy=DistanceStrategy.COSINE
    )
    vectorstore.save_local("vectorstore_cache")

if __name__ == "__main__":
    ingest_documents()
