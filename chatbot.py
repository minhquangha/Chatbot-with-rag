from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from adapter import SentenceTransformerEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from google import genai
from langchain_openai import ChatOpenAI
load_dotenv()

loader = DirectoryLoader(
    path="./papers",
    glob="**/*.pdf",
    loader_cls=UnstructuredFileLoader, #dùng để đọc dữ liệu loại file
    show_progress=True,
    use_multithreading= True
)
docs= loader.load()

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

text_splitter =  RecursiveCharacterTextSplitter(
    chunk_size = 1200,    # Số ký tự tối đa cho mỗi chunk
    chunk_overlap=50,     # Số ký tự ghi đè giữa các chunk để giữ ngữ cảnh
    add_start_index = True,
    strip_whitespace = True,
    separators=MARKDOWN_SEPARATORS
)

splits = text_splitter.split_documents(docs)
from pprint import pprint
pprint(splits)


from sentence_transformers import SentenceTransformer
# embbedding chunks -> vector 
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
texts  = [chunk.page_content for chunk in splits]
embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vectorstore = FAISS.from_documents(
    documents=splits,
    embedding=embeddings,
    distance_strategy = DistanceStrategy.COSINE
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
    {"context":retriever,"question":RunnablePassthrough()}
    | prompt
    | llm
    |StrOutputParser()
)

question = input("Question: ")
answer = rag_chain.invoke(question)
print(answer)





