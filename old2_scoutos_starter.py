# ScoutOS Hybrid (Local Only - Mistral via Ollama) with Memory + Web Search + Configurable Prompt
# ----------------------------------------------------------------------------------
# This version uses a local Mistral model via Ollama only,
# adds session-based memory support, optional web search augmentation,
# and allows configuration of the system prompt via config.yaml.

# 1. Setup:
# pip install streamlit llama-index python-dotenv requests pyyaml

import os
import streamlit as st
import requests
import json
import yaml
from dotenv import load_dotenv
from datetime import datetime

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms.llm import LLM
from llama_index.core.base.llms.types import CompletionResponse, ChatResponse, MessageRole, ChatMessage, LLMMetadata
from typing import List, Optional, Any
from pydantic import BaseModel, Field

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Memory file
MEMORY_FILE = "chat_memory.json"
CONFIG_FILE = "config.yml"

# Load and interpolate system prompt from config
DEFAULT_SYSTEM_PROMPT = """
You are a helpful local AI assistant with access to real-time web search results.
Always use the most recent information provided in the [Web Search Result] section to answer current questions.
If dates or versions are mentioned in that section, treat them as more reliable than your pre-trained data.
"""

current_date = datetime.now().strftime("%Y-%m-%d")
knowledge_cutoff_year = "2023"  # Update as needed

if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as f:
        config_data = yaml.safe_load(f)
        raw_prompt = config_data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
else:
    raw_prompt = DEFAULT_SYSTEM_PROMPT

SYSTEM_PROMPT = raw_prompt.format(current_date=current_date, knowledge_cutoff_year=knowledge_cutoff_year)

logger.info(f"{SYSTEM_PROMPT}")

# Define local Mistral LLM using the LLM base class
class MistralLLM(LLM, BaseModel):
    model: str = Field(default="mistral")
    url: str = Field(default="http://localhost:11434/api/generate")
    system_prompt: str = Field(default=SYSTEM_PROMPT)
    query_wrapper_prompt: Optional[str] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=2048,
            num_output=256,
        )

    def chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        prompt = self.system_prompt + "\n\n" + "\n".join([f"{m.role.value}: {m.content}" for m in messages])
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(self.url, json=payload)
            if response.status_code == 200:
                return ChatResponse(
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=response.json().get("response", "[No response]")
                    )
                )
            else:
                return ChatResponse(
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=f"[Error: {response.status_code}]"
                    )
                )
        except Exception as e:
            return ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=f"[Exception: {e}]"
                )
            )

    async def achat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        return self.chat(messages)

    def stream_chat(self, messages: List[ChatMessage], **kwargs):
        raise NotImplementedError("stream_chat is not implemented.")

    async def astream_chat(self, messages: List[ChatMessage], **kwargs):
        raise NotImplementedError("astream_chat is not implemented.")

    def stream_complete(self, prompt: str, **kwargs):
        raise NotImplementedError("stream_complete is not implemented.")

    async def astream_complete(self, prompt: str, **kwargs):
        raise NotImplementedError("astream_complete is not implemented.")

    async def acomplete(self, prompt: str, **kwargs) -> CompletionResponse:
        return self.complete(prompt, **kwargs)

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(self.url, json=payload)
            if response.status_code == 200:
                return CompletionResponse(text=response.json().get("response", "[No response]"))
            else:
                return CompletionResponse(text=f"[Error from local model: {response.status_code}]")
        except Exception as e:
            return CompletionResponse(text=f"[Failed to reach local model: {e}]")

# Dummy embedding model for local fallback (no OpenAI usage)
class DummyEmbedding(BaseEmbedding):
    def _get_text_embedding_batch(self, texts):
        return [[0.0] * 1536 for _ in texts]

    def _get_query_embedding(self, query):
        return [0.0] * 1536

    async def _aget_query_embedding(self, query):
        return [0.0] * 1536

    def _get_text_embedding(self, text):
        return [0.0] * 1536

# Simple web search using DuckDuckGo Instant Answer API
def web_search(query):
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    api_key = os.getenv("SERPER_API_KEY")
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": query}
    
    try:
        response = requests.post("https://google.serper.dev/search", headers=headers, json=payload)
        data = response.json()
        if "answerBox" in data and "answer" in data["answerBox"]:
            result = data["answerBox"]["answer"]
        elif "organic" in data and len(data["organic"]) > 0:
            result = data["organic"][0]["snippet"]
        else:
            result = "[No relevant result found.]"

        logger.info(f"Web search result: {result}")
        return result
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return f"[Web search error: {e}]"

# Initialize LLM and embedding
llm = MistralLLM()
embed_model = DummyEmbedding()
Settings.llm = llm
Settings.embed_model = embed_model

@st.cache_resource
def load_index():
    target_dirs = ["data"]
    documents = []
    for path in target_dirs:
        if os.path.exists(path):
            docs = SimpleDirectoryReader(path, recursive=True).load_data()
            documents.extend(docs)

    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist("storage")
    return index

index = load_index()
query_engine = index.as_query_engine()

# Memory handling with trimming
MAX_MEMORY_TURNS = 30  # each turn includes both user and assistant

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            memory = json.load(f)
            return memory[-MAX_MEMORY_TURNS:]  # trim older messages
    return []

def save_memory(memory):
    trimmed = memory[-MAX_MEMORY_TURNS:]
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f)

# Streamlit UI
st.title("🧠 ScoutOS")

chat_container = st.container()
input_container = st.container()

memory = load_memory()

with chat_container:
    for turn in memory:
        st.markdown(f"**You:** {turn['user']}")
        st.markdown(f"**ScoutOS:** {turn['assistant']}")

def handle_query(user_query: str, use_web: bool) -> str:
    messages = []
    memory = load_memory()

    for turn in memory:
        messages.append(ChatMessage(role=MessageRole.USER, content=turn["user"]))
        messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=turn["assistant"]))

    search_keywords = ["latest", "news", "version", "release", "update"]
    bypass_search_phrases = [
        "what's the date", "what day is it", "what is the current date",
        "today's date", "what day", "what year is it", "current year"
    ]

    should_bypass = any(phrase in user_query.lower() for phrase in bypass_search_phrases)
    should_search = any(word in user_query.lower() for word in search_keywords) and not should_bypass

    logger.info(f"should_bypass: {should_bypass}")
    logger.info(f"should_search: {should_search}")

    if use_web and should_search:
        web_context = web_search(user_query)
        full_prompt = f"You can use the following real-time web result to help answer the question.\n\n[Web Search Result]\n{web_context}\n\n[Question]\n{user_query}"
    else:
        full_prompt = user_query

    messages.append(ChatMessage(role=MessageRole.USER, content=full_prompt))

    response = llm.chat(messages)
    assistant_reply = response.message.content

    memory.append({"user": user_query, "assistant": assistant_reply})
    save_memory(memory)

    return assistant_reply

with input_container.form("chat_form", clear_on_submit=True):
    user_query = st.text_input("Ask ScoutOS something about your files:", key="chat_input")
    use_web = st.checkbox("Augment with Web Search", value=True)
    submitted = st.form_submit_button("Send")

if submitted and user_query:
    memory = load_memory()
    memory.append({"user": user_query, "assistant": "ScoutOS is thinking..."})
    save_memory(memory)
    st.rerun()

# After rerun, check for thinking placeholder and resolve
if memory and memory[-1]["assistant"] == "thinking...":
    last_query = memory[-1]["user"]
    assistant_reply = handle_query(last_query, use_web)
    memory[-1]["assistant"] = assistant_reply
    save_memory(memory)
    st.rerun()

st.caption("ScoutOS | Local Mistral via Ollama + Persistent Memory + Web Search + Configurable Prompt")