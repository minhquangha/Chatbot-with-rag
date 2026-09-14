from langchain_community.vectorstores import FAISS
from adapter import SentenceTransformerEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from load import embeddings
load_dotenv()

def format_docs(docs):
    return "\n\n".join(
        f"--- Đoạn trích từ {doc.metadata.get('source', 'Unknown')} ---\n{doc.page_content.strip()}"
        for doc in docs
    )


vectorstore = FAISS.load_local(
    folder_path="vectorstore_cache",
    embeddings=embeddings,
    distance_strategy = DistanceStrategy.COSINE,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(
    search_type ="similarity",
    search_kwargs = {"k":5}
)

template = (
    "You are a strict, citation-focused assistant for a private knowledge base.\n"
    "RULES:\n"
    "1) Use ONLY the provided context to answer.\n"
    "2) If the answer is not clearly contained in the context, say: "
    "\"I don't know based on the provided documents.\"\n"
    "3) Do NOT use outside knowledge, guessing, or web information.\n"
    "4) If applicable, cite sources as (source:page) using the metadata.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

prompt = ChatPromptTemplate.from_template(template=template)

#llm
llm = ChatOpenAI(
    model="deepseek-v4-flash-0731",
    temperature=0,
    base_url="https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
)

rag_chain =  (
    {"context":retriever| format_docs,"question":RunnablePassthrough()}
    | prompt
    | llm
    |StrOutputParser()
)

while True:
    question = input("Question: ")
    if question == "exit":
        break
    answer = rag_chain.invoke(question)
    print("Answer:",answer)





