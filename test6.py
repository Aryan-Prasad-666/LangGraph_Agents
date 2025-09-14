from typing import Dict, TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere
from langchain_community.tools.google_serper import GoogleSerperResults
from dotenv import load_dotenv
import os
import logging
import json
from datetime import datetime
import re
from functools import lru_cache
from IPython.display import Image, display

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

groq_key = os.getenv('groq_api_key')
gemini_key = os.getenv('gemini_api_key')
cohere_key = os.getenv('cohere_api_key')
serper_key = os.getenv('serper_api_key')
if not all([groq_key, gemini_key, cohere_key, serper_key]):
    raise ValueError("Missing one or more API keys in environment variables")

llm_deepseek = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    api_key=groq_key,
    temperature=0.6
)
llm_gpt = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=groq_key,
    temperature=0.6
)
llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=gemini_key,
    temperature=0.6
)
llm_cohere = ChatCohere(
    api_key=cohere_key,
    temperature=0.6
)


print(llm_cohere.invoke("HEY"))