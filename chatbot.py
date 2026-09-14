import sys
from pathlib import Path

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from adapter import SentenceTransformerEmbeddings
import config


def format_docs(docs) -> str:
    """Định dạng các đoạn trích cùng nguồn và số trang cụ thể."""
    formatted_chunks = []
    for doc in docs:
        raw_source = doc.metadata.get("source", "Unknown")
        source_name = Path(raw_source).name
        page = doc.metadata.get("page", None)
        page_info = f" - Trang {int(page) + 1}" if page is not None else ""
        formatted_chunks.append(
            f"--- Đoạn trích từ {source_name}{page_info} ---\n{doc.page_content.strip()}"
        )
    return "\n\n".join(formatted_chunks)


def build_rag_chain():
    """Khởi tạo và trả về RAG chain."""
    index_file = config.VECTORSTORE_DIR / "index.faiss"
    if not index_file.exists():
        print(f"\n[!] Chưa tìm thấy chỉ mục vectorstore tại: {config.VECTORSTORE_DIR}")
        print("[!] Vui lòng chạy lệnh: 'uv run python load.py' để nạp dữ liệu trước khi khởi động chatbot.\n")
        sys.exit(1)

    embeddings = SentenceTransformerEmbeddings(model_name=config.EMBEDDING_MODEL)

    vectorstore = FAISS.load_local(
        folder_path=str(config.VECTORSTORE_DIR),
        embeddings=embeddings,
        distance_strategy=DistanceStrategy.COSINE,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.RETRIEVER_K}
    )

    template = (
        "You are a strict, citation-focused assistant for a private knowledge base.\n"
        "RULES:\n"
        "1) Use ONLY the provided context to answer.\n"
        "2) If the answer is not clearly contained in the context, say: "
        "\"Tôi không tìm thấy thông tin này trong tài liệu được cung cấp.\"\n"
        "3) Do NOT use outside knowledge, guessing, or web information.\n"
        "4) Cite sources as (Tên_file: Trang x) based on the section headers in the context.\n"
        "5) Always answer in the same language as the user's question (e.g., respond in Vietnamese if the question is in Vietnamese).\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )

    prompt = ChatPromptTemplate.from_template(template=template)

    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        base_url=config.LLM_BASE_URL
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def main():
    rag_chain = build_rag_chain()

    while True:
        try:
            question = input("Question: ").strip()
            if not question:
                continue
            if question.lower() in ["exit", "quit", "q"]:
                break

            answer = rag_chain.invoke(question)
            print("Answer:", answer)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    main()
