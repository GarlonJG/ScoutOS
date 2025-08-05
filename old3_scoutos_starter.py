# ScoutOS Hybrid (Local Mistral via Ollama) with Sessions + Memory + Web Search + Prompt Config + Persisted Web Context

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
from typing import List, Optional
from pydantic import BaseModel, Field

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------- Setup ----------
load_dotenv()
st.set_page_config(page_title="ScoutOS", layout="wide")

# ---------- Config ----------
MEMORY_FILE = "chat_memory.json"
CONFIG_FILE = "config.yml"
DEFAULT_SYSTEM_PROMPT = """
You are a helpful local AI assistant with access to real-time web search results.
Always use the most recent information provided in the [Web Search Result] section to answer current questions.
If dates or versions are mentioned in that section, treat them as more reliable than your pre-trained data.
"""
current_date = datetime.now().strftime("%Y-%m-%d")
knowledge_cutoff_year = "2023"

config_data = {}
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as f:
        config_data = yaml.safe_load(f)

raw_prompt = config_data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
persist_web_context = config_data.get("persist_web_context", False)

SYSTEM_PROMPT = raw_prompt.format(
    current_date=current_date,
    knowledge_cutoff_year=knowledge_cutoff_year
)

# ---------- Models ----------
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

class DummyEmbedding(BaseEmbedding):
    def _get_text_embedding_batch(self, texts): return [[0.0] * 1536 for _ in texts]
    def _get_query_embedding(self, query): return [0.0] * 1536
    async def _aget_query_embedding(self, query): return [0.0] * 1536
    def _get_text_embedding(self, text): return [0.0] * 1536

# ---------- Search ----------
def web_search(query):
    api_key = os.getenv("SERPER_API_KEY")
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    try:
        resp = requests.post("https://google.serper.dev/search", headers=headers, json={"q": query})
        data = resp.json()
        return data.get("answerBox", {}).get("answer") or (data.get("organic", [{}])[0].get("snippet") or "[No relevant result found.]")
    except Exception as e:
        return f"[Web search error: {e}]"

# ---------- Memory ----------
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

def search_past_conversations(memory, query):
    matches = []
    query_words = query.lower().split()
    for turn in memory:
        if any(word in turn["user"].lower() for word in query_words):
            matches.append(turn)
    return matches

# ---------Removed redundant parts of answers -------- maybe look for key phrases since there are common sentences used that are similar to each other.
def trim_response(response_text: str) -> str:
    common_redundancies = [
        "As a responsible and helpful AI assistant,",
        "To ensure I'm providing you with the most up-to-date and accurate information,",
        "Thank you for using ScoutOS for assistance today!",
        "However, since I don't have real-time access to the internet",
        "However, I should always reference the prompt for the most accurate and up-to-date information regarding the current date when comparing to web search results.",
        "To ensure that I'm providing you with the most up-to-date and accurate information, it would be best to cross-reference this data with reliable sources such as BBC News, CNN, or the White House website."
        "However, please note that it is essential to cross-reference this data with multiple reliable sources such as BBC News, CNN, or the White House website for confirmation and accuracy.",
    ]
    for phrase in common_redundancies:
        response_text = response_text.replace(phrase, "")
    return response_text.strip()

# ---------- Query Logic ----------
def handle_query(user_query: str, use_web: bool, search_memory: bool) -> str:
    memory = load_memory()
    messages = []

    if search_memory:
        past_matches = search_past_conversations(memory, user_query)
        if past_matches:
            context_snippets = "\n".join([f"- {m['user']}: {m['assistant']}" for m in past_matches])
            messages.append(ChatMessage(
                role=MessageRole.USER,
                content=f"These past interactions may be useful:\n{context_snippets}"
            ))

    # Determine if web search is necessary
    search_keywords = ["latest", "news", "version", "release", "update", "current"]
    bypass_search_phrases = [
        "what's the date", "what day is it", "what is the current date",
        "today's date", "what day", "what year is it", "current year"
    ]
    should_bypass = any(phrase in user_query.lower() for phrase in bypass_search_phrases)
    should_search = any(word in user_query.lower() for word in search_keywords) and not should_bypass

    web_context = None
    if use_web and should_search:
        web_context = web_search(user_query)
        full_prompt = f"You can use the following real-time web result to help answer the question.\n\n[Web Search Result]\n{web_context}\n\n[Question]\n{user_query}"
    else:
        full_prompt = user_query

    messages.append(ChatMessage(role=MessageRole.USER, content=full_prompt))
    response = llm.chat(messages)
    assistant_reply = response.message.content

    memory_entry = {
        "timestamp": datetime.now().isoformat(),
        "category": "default",
        "user": user_query,
        "assistant": assistant_reply
    }
    if persist_web_context and web_context:
        memory_entry["web_result"] = web_context

    memory.append(memory_entry)
    save_memory(memory)
    return assistant_reply

# ---------- LLM Setup ----------
llm = MistralLLM()
Settings.llm = llm
Settings.embed_model = DummyEmbedding()

@st.cache_resource
def load_index():
    if os.path.exists("data"):
        documents = SimpleDirectoryReader("data", recursive=True).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist("storage")
        return index
    return VectorStoreIndex.from_documents([])

# ---------- UI ----------
st.title("🧠 ScoutOS")

memory = load_memory()
chat_container = st.container()
input_container = st.container()

with chat_container:
    for turn in memory:
        st.markdown(f"**You:** {turn['user']}")
        st.markdown(f"**ScoutOS:** {turn['assistant']}")

with input_container.form("chat_form", clear_on_submit=True):
    user_query = st.text_input("Ask ScoutOS something:", key="chat_input")
    use_web = st.checkbox("Augment with Web Search", value=True)
    search_memory = st.checkbox("Search Past Conversations", value=True)
    submitted = st.form_submit_button("Send")

if submitted and user_query:
    memory.append({"user": user_query, "assistant": "thinking..."})
    save_memory(memory)
    st.rerun()

if memory and memory[-1]["assistant"] == "thinking...":
    last_query = memory[-1]["user"]
    assistant_reply = handle_query(last_query, use_web, search_memory)
    memory[-1]["assistant"] = assistant_reply
    save_memory(memory)
    st.rerun()

st.caption("ScoutOS | Local Mistral via Ollama + Persistent Memory + Web Search + Configurable Prompt")