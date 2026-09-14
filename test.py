import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import config

load_dotenv()

llm = ChatOpenAI(
    model=config.LLM_MODEL,
    temperature=config.LLM_TEMPERATURE,
    base_url=config.LLM_BASE_URL
)

print("Before invoke")

response = llm.invoke("Say hello in Vietnamese")

print("After invoke")
print(response.content)