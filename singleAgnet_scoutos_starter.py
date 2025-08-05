# ScoutOS Hybrid (Local Mistral via Ollama) with Semantic + Flat Memory + Web Search + Prompt Config
# ---------------------------------------------------------------------------------------------------

import os, json, requests, yaml
import time
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, PrivateAttr
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

import torch
if not hasattr(torch, "get_default_device"):
    def get_default_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.get_default_device = get_default_device

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms.llm import LLM
from llama_index.core.base.llms.types import CompletionResponse, ChatResponse, MessageRole, ChatMessage, LLMMetadata

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------- Setup ----------
load_dotenv()
st.set_page_config(page_title="ScoutOS", layout="wide")

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
SYSTEM_PROMPT = raw_prompt.format(current_date=current_date, knowledge_cutoff_year=knowledge_cutoff_year)

# ---------- Embedding ----------
class HuggingFaceEmbedding(BaseEmbedding):
    _model: SentenceTransformer = PrivateAttr()

    def __init__(self):
        super().__init__()
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
    def _get_query_embedding(self, query): return self.model.encode(query).tolist()
    def _get_text_embedding(self, text): return self.model.encode(text).tolist()
    def _get_text_embedding_batch(self, texts): return self.model.encode(texts).tolist()
    async def _aget_query_embedding(self, query): return self._get_query_embedding(query)

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
            start = time.perf_counter()
            response = requests.post(self.url, json=payload)
            duration = time.perf_counter() - start
            logger.info(f"Mistral chat call took {duration:.3f}s")
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

# ---------- Memory ----------
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE) as f: return json.load(f)
    return []

def save_memory(memory): 
    with open(MEMORY_FILE, "w") as f: json.dump(memory, f, indent=2)

# ---------- Semantic Memory ----------
class SemanticMemory:
    def __init__(self, memory):
        self.memory = memory
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.IndexFlatL2(384)
        self.texts = []
        self.ids = []
        self.build_index()

    def build_index(self):
        start = time.perf_counter()
        embeddings = []
        for idx, item in enumerate(self.memory):
            text = f"{item['user']} -> {item['assistant']}"
            emb = self.model.encode(text)
            embeddings.append(emb)
            self.texts.append(text)
            self.ids.append(idx)
        if embeddings:
            self.index.add(np.array(embeddings).astype("float32"))
            logger.info(f"SemanticMemory: index built in {time.perf_counter() - start:.3f}s")

    def search(self, query: str, top_k=3):
        start = time.perf_counter()
        query_vec = self.model.encode([query]).astype("float32")
        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for i in indices[0]:
            if 0 <= i < len(self.memory):
                results.append(self.memory[self.ids[i]])
        logger.info(f"SemanticMemory: search for '{query}' took {time.perf_counter() - start:.3f}s")
        return results

# ---------- Web Search ----------
def web_search(query):
    start = time.perf_counter()
    api_key = os.getenv("SERPER_API_KEY")
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    try:
        r = requests.post("https://google.serper.dev/search", headers=headers, json={"q": query})
        d = r.json()
        logger.info(f"Web search for '{query}' took {time.perf_counter() - start:.3f}s")
        return d.get("answerBox", {}).get("answer") or d.get("organic", [{}])[0].get("snippet", "[No relevant result found.]")
    except Exception as e:
        return f"[Web search error: {e}]"

# ---------- Trim verbosity ----------
def trim_response(text):
    redundancies = [
        "Thank you for using ScoutOS for assistance today!",
        "To ensure that I'm providing you with the most up-to-date and accurate information,",
        "I will always endeavor to provide you with accurate and up-to-date information.",
    ]
    for phrase in redundancies:
        text = text.replace(phrase, "")
    return text.strip()

# ---------- Query Logic ----------
def handle_query(user_query, use_web, search_memory, semantic_memory_enabled):
    start_total = time.perf_counter()
    memory = load_memory()
    messages = []

    if search_memory and semantic_memory_enabled:
        sem_mem = SemanticMemory(memory)
        logger.info(f"SemMemory: {sem_mem}")
        past_matches = sem_mem.search(user_query)
    elif search_memory:
        past_matches = [m for m in memory if any(w in m["user"].lower() for w in user_query.lower().split())]
    else:
        past_matches = []

    if past_matches:
        ctx = "\n".join([f"- {m['user']}: {m['assistant']}" for m in past_matches])
        messages.append(ChatMessage(role=MessageRole.USER, content=f"Relevant context from past chats:\n{ctx}"))

    search_keywords = ["latest", "news", "version", "release", "update"]
    bypass_phrases = ["what's the date", "what day", "current year"]
    should_search = any(w in user_query.lower() for w in search_keywords) and not any(p in user_query.lower() for p in bypass_phrases)

    web_context = web_search(user_query) if use_web and should_search else None
    prompt = f"[Web Search Result]\n{web_context}\n\n[Question]\n{user_query}" if web_context else user_query
    messages.append(ChatMessage(role=MessageRole.USER, content=prompt))

    response = llm.chat(messages)
    assistant_reply = trim_response(response.message.content)

    memory_entry = {
        "timestamp": datetime.now().isoformat(),
        "user": user_query,
        "assistant": assistant_reply
    }
    if persist_web_context and web_context:
        memory_entry["web_result"] = web_context

    memory.append(memory_entry)
    save_memory(memory)
    logger.info(f"handle_query() for '{user_query}' took {time.perf_counter() - start_total:.3f}s")
    return assistant_reply

# ---------- LLM Setup ----------
llm = MistralLLM()
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding()

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
    semantic_toggle = st.checkbox("Use Semantic Search", value=True)
    submitted = st.form_submit_button("Send")

if submitted and user_query:
    memory.append({"user": user_query, "assistant": "thinking..."})
    save_memory(memory)
    st.rerun()

if memory and memory[-1]["assistant"] == "thinking...":
    last_query = memory[-1]["user"]
    assistant_reply = handle_query(last_query, use_web, search_memory, semantic_toggle)
    memory[-1]["assistant"] = assistant_reply
    save_memory(memory)
    st.rerun()

st.caption("ScoutOS | Local Mistral via Ollama + Flat + Semantic Memory + Web Search")