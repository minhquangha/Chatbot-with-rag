from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import os
load_dotenv()

llm = ChatOpenAI(
    model="deepseek-v4-flash-0731",
    temperature=0,
    base_url="https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
)

print("Before invoke")

response = llm.invoke("Say hello in Vietnamese")

print("After invoke")
print(response.content)