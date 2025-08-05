import os, json, requests, yaml
import time
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, PrivateAttr
#from langchain.document_loaders import TextLoader, PyPDFLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
# Using HuggingFaceEmbedding instead of SentenceTransformer for consistency
from semantic_document import SemanticDocumentIndex
import numpy as np
import shutil

# Set up logging first
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Now check for FAISS availability
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available, using simple vector search fallback")

from simple_vector_search import create_simple_index

import torch
if not hasattr(torch, "get_default_device"):
    def get_default_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.get_default_device = get_default_device

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document, ServiceContext, Settings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms.llm import LLM
from llama_index.core.base.llms.types import CompletionResponse, ChatResponse, MessageRole, ChatMessage, LLMMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

load_dotenv()
st.set_page_config(page_title="ScoutOS", layout="wide")

MEMORY_FILE = "chat_memory.json"
CONFIG_FILE = "config.yml"
DEFAULT_SYSTEM_PROMPT = """
You are ScoutOS, a helpful local AI assistant. You have access to:
- Your general knowledge and training
- Web search results (when provided)
- Document context (when relevant)
- Previous conversation history

IMPORTANT INSTRUCTIONS:
1. For general questions, rely primarily on your knowledge and training
2. Only use document context when the question is specifically about those documents
3. Use web search results for current events or recent information
4. Be conversational and helpful, matching the user's tone
5. If unsure what context to use, ask for clarification
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
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

INDEX_FILE = "semantic_index.faiss"
ID_MAP_FILE = "semantic_ids.json"

#logger.info(f"OLLAMA_NUM_THREAD={os.getenv('OLLAMA_NUM_THREAD')}")
#logger.info(f"OLLAMA_NUM_CPU={os.getenv('OLLAMA_NUM_CPU')}")

class MistralLLM(LLM, BaseModel):
    model: str = Field(default="mistral-tuned")
    url: str = Field(default="http://localhost:11434/api/chat")
    system_prompt: str = Field(default=SYSTEM_PROMPT)
    query_wrapper_prompt: Optional[str] = Field(default=None)
    temperature: float = 0.7

    class Config:
        arbitrary_types_allowed = True

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=2048,
            num_output=256,
        )

    def chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        payload = {
            "model": self.model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": self.temperature
            }
        }

        try:
            # Add timeout to prevent hanging (60 seconds)
            response = requests.post(self.url, json=payload, timeout=360.0)
            if response.status_code == 200:
                data = response.json()
                content = data.get("message", {}).get("content", "")
                if not content or content.strip() == "":
                    content = "[No response from model]"
                
                return ChatResponse(
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=content
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

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE) as f: return json.load(f)
    return []

def save_memory(memory): 
    with open(MEMORY_FILE, "w") as f: json.dump(memory, f, indent=2)

class SemanticMemory:
    def __init__(self, memory, model):
        self.memory = memory
        self.model = model
        self.texts = [item['user'] for item in memory]
        logger.info(f"Indexing {len(self.texts)} user-only prompts like: {self.texts[:3]}")
        self.ids = list(range(len(memory)))

        if FAISS_AVAILABLE and os.path.exists(INDEX_FILE) and os.path.exists(ID_MAP_FILE):
            logger.info("Loading cached FAISS index and ID map...")
            try:
                self.index = faiss.read_index(INDEX_FILE)
                with open(ID_MAP_FILE, "r") as f:
                    self.ids = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached index: {e}")
                logger.info("Building new index...")
                self.index = self.build_faiss_index(self.texts)
                self.save_index()
        else:
            logger.info("Building new semantic memory index...")
            logger.info(f"Indexing {len(self.texts)} user-only prompts like: {self.texts[:3]}")
            self.index = self.build_faiss_index(self.texts)
            self.save_index()

    def build_faiss_index(self, texts):
        logger.info("🔍 Starting to embed texts for FAISS index...")
        start = time.perf_counter()
        if not texts:
            return None
        
        # Handle different embedding model interfaces
        if hasattr(self.model, 'encode'):
            # Legacy encode() interface
            embeddings = self.model.encode(
                texts,
                batch_size=64,
                normalize_embeddings=True,
                convert_to_tensor=False
            )
        else:
            # HuggingFaceEmbedding interface
            embeddings = [self.model.get_text_embedding(text) for text in texts]
            
        embeddings = np.array(embeddings).astype("float32")
        dim = embeddings.shape[1]
        
        if FAISS_AVAILABLE:
            try:
                index = faiss.IndexFlatIP(dim)
                index.add(embeddings)
                logger.info(f"✅ FAISS index created successfully in {time.perf_counter() - start:.3f}s")
            except Exception as e:
                logger.error(f"FAISS index creation failed: {e}")
                logger.warning("Falling back to simple vector search")
                index = create_simple_index(dim)
                index.add(embeddings)
        else:
            logger.info("Using simple vector search (FAISS not available)")
            index = create_simple_index(dim)
            index.add(embeddings)
            
        logger.info(f"SemanticMemory: built index with {len(texts)} items in {time.perf_counter() - start:.3f}s")
        return index

    def save_index(self):
        if FAISS_AVAILABLE and hasattr(self.index, 'ntotal'):
            # Save FAISS index
            faiss.write_index(self.index, INDEX_FILE)
            with open(ID_MAP_FILE, "w") as f:
                json.dump(self.ids, f)
            logger.info(f"Saved FAISS index with {self.index.ntotal} vectors")
        else:
            # For simple vector search, just save the IDs
            with open(ID_MAP_FILE, "w") as f:
                json.dump(self.ids, f)
            logger.info("Saved semantic memory metadata (no FAISS index)")

    def add_memory(self, new_text):
        embedding = self.model.encode([new_text], normalize_embeddings=True)[0]
        embedding = np.array([embedding]).astype("float32")
        self.index.add(embedding)
        self.ids.append(len(self.ids))
        self.save_index()

    def search(self, query: str, top_k=5, min_score=0.6):
        if not self.index or not self.memory:
            return []
        
        logger.info(f"🔍 Encoding query for: '{query}'")
        start = time.perf_counter()
        
        # Handle different embedding model interfaces
        if hasattr(self.model, 'encode'):
            # Legacy encode() interface
            query_vec = self.model.encode([query], convert_to_numpy=True)
        else:
            # HuggingFaceEmbedding interface (preferred)
            query_vec = np.array([self.model.get_text_embedding(query)])
            
        query_vec = np.array(query_vec).astype("float32")
        
        try:
            scores, indices = self.index.search(query_vec, top_k)
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

        logger.info(f"🔎 Scores: {scores[0]}")
        logger.info(f"🔎 Indices: {indices[0]}")

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.memory) and score >= min_score:
                match = self.memory[self.ids[idx]]
                match['similarity_score'] = float(score)  # Optional: helpful for debugging
                results.append(match)

        logger.info(f"✅ Query encoding + search took {time.perf_counter() - start:.3f}s")
        logger.info(f"SemanticMemory: search for '{query}' took {time.perf_counter() - start:.3f}s with {len(results)} matches over threshold {min_score}")
        return results
    
    def inspect_self(self, file_path="scoutos_starter.py"):
        """
        Return the contents of the main startup script for ScoutOS, if available.
        Useful for self-reflection or debugging.
        """
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return f.read()
        return f"[Error: {file_path} not found]"
    
    def backup_self(self, file_path="scoutos_starter.py"):
        backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            shutil.copyfile(file_path, backup_path)
            return f"[Backup created at {backup_path}]"
        except Exception as e:
            return f"[Backup failed: {e}]"
    
    def write_self(self, new_code: str, file_path="scoutos_starter.py"):
        """
        Overwrite the startup file with new code.
        Creates a backup before overwriting.
        Returns success/error messages.
        """
        backup_msg = self.backup_self(file_path)
        try:
            # Basic validation: check if code contains 'def' or 'class' to reduce nonsense overwrites
            if "def " not in new_code and "class " not in new_code:
                return "[Error] New code looks suspiciously short or invalid."

            with open(file_path, "w") as f:
                f.write(new_code)
            return f"{backup_msg} [Success] Updated {file_path}"
        except Exception as e:
            return f"[Error writing {file_path}: {e}]"

def init_memory_and_index():
    # Only load texts if not already loaded
    if "texts" not in st.session_state:
        st.session_state.texts = load_memory()
        st.session_state.last_memory_size = len(st.session_state.texts)

    # Only create document index if not already created
    if "doc_index" not in st.session_state:
        st.session_state.doc_index = SemanticDocumentIndex(
            folder_path="data",
            embedding_model=Settings.embed_model
        )

    # Only create semantic memory if not already created
    if "semantic_memory" not in st.session_state:
        st.session_state.semantic_memory = SemanticMemory(st.session_state.texts, st.session_state.embedder)
        st.session_state.faiss_index = st.session_state.semantic_memory.index
    
    # Only update semantic memory if texts have actually changed (new conversations)
    elif ("last_memory_size" in st.session_state and 
          st.session_state.last_memory_size != len(st.session_state.texts)):
        # Only rebuild if we have significantly more texts (avoid rebuilding on every minor change)
        if len(st.session_state.texts) > st.session_state.last_memory_size:
            logger.info(f"Updating semantic memory: {st.session_state.last_memory_size} -> {len(st.session_state.texts)} texts")
            st.session_state.semantic_memory = SemanticMemory(st.session_state.texts, st.session_state.embedder)
            st.session_state.faiss_index = st.session_state.semantic_memory.index
        st.session_state.last_memory_size = len(st.session_state.texts)

def web_search(query):
    start = time.perf_counter()
    api_key = os.getenv("SERPER_API_KEY")
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    try:
        r = requests.post("https://google.serper.dev/search", headers=headers, json={"q": query})
        d = r.json()
        logger.info(f"Search result: '{query}' took {time.perf_counter() - start:.3f}s")
        return d.get("answerBox", {}).get("answer") or d.get("organic", [{}])[0].get("snippet", "[No relevant result found.]")
    except Exception as e:
        return f"[Web search error: {e}]"

def trim_response(text):
    redundancies = [
        "Thank you for using ScoutOS for assistance today!",
        "To ensure that I'm providing you with the most up-to-date and accurate information,",
        "I will always endeavor to provide you with accurate and up-to-date information.",
    ]
    for phrase in redundancies:
        text = text.replace(phrase, "")
    return text.strip()

def handle_query(user_query, use_web, search_memory, semantic_memory_enabled):
    start_total = time.perf_counter()
    logger.info(f"🚀 Starting query: '{user_query[:30]}...'")
    
    # Initialize session state if not exists
    if 'response_cache' not in st.session_state:
        st.session_state.response_cache = {}
    
    # Initialize semantic memory if enabled and not already done
    if semantic_memory_enabled and 'semantic_memory' not in st.session_state:
        try:
            from semantic_memory import SemanticMemory  # Import here to avoid circular imports
            st.session_state.semantic_memory = SemanticMemory()
            logger.info("✅ Semantic memory initialized")
        except Exception as e:
            logger.error(f"Failed to initialize semantic memory: {e}")
            semantic_memory_enabled = False
    
    # Check for special commands first (only if semantic_memory is initialized)
    if any(kw in user_query.lower() for kw in ["inspect your source", "show me your code", "startup file"]):
        if hasattr(st.session_state, 'semantic_memory') and st.session_state.semantic_memory is not None:
            code = st.session_state.semantic_memory.inspect_self()
            return f"\n\n```python\n{code}\n```"
        return "Semantic memory not initialized. Please try again in a moment."
    
    # Check cache first (simple hash of the query)
    cache_key = hash(user_query.lower().strip())
    if cache_key in st.session_state.response_cache:
        logger.info("Cache hit for query")
        return st.session_state.response_cache[cache_key]

    # Run searches in parallel
    memory_start = time.perf_counter()
    doc_start = time.perf_counter()
    web_start = time.perf_counter()
    
    # Parallel execution of searches
    import concurrent.futures
    
    def run_memory_search():
        start_time = time.perf_counter()
        try:
            if not search_memory:
                return []
                
            # Simple text search fallback (faster for short queries)
            query_terms = user_query.lower().split()
            if len(query_terms) <= 3:  # Simple search for very short queries
                if hasattr(st.session_state, 'texts'):
                    results = [
                        m for m in st.session_state.texts 
                        if any(term in m["user"].lower() for term in query_terms)
                    ]
                    if results:  # If we found matches in simple search, return them
                        logger.debug(f"Simple text search found {len(results)} matches in {(time.perf_counter() - start_time)*1000:.1f}ms")
                        return results
            
            # Semantic search if enabled and initialized
            if semantic_memory_enabled and hasattr(st.session_state, 'semantic_memory') and st.session_state.semantic_memory:
                try:
                    results = st.session_state.semantic_memory.search(user_query)
                    logger.debug(f"Semantic search completed in {(time.perf_counter() - start_time)*1000:.1f}ms")
                    return results
                except Exception as e:
                    logger.error(f"Semantic search failed: {e}")
            
            # Fallback to simple text search if semantic search not available
            if hasattr(st.session_state, 'texts'):
                results = [m for m in st.session_state.texts if any(term in m["user"].lower() for term in query_terms)]
                logger.debug(f"Fallback text search found {len(results)} matches in {(time.perf_counter() - start_time)*1000:.1f}ms")
                return results
                
            return []
            
        except Exception as e:
            logger.error(f"Error in memory search: {e}")
            return []
    
    def run_doc_search():
        if "doc_index" not in st.session_state or st.session_state.doc_index is None:
            return None
        try:
            return st.session_state.doc_index.search(user_query, top_k=2)  # Reduced from 3 to 2
        except Exception as e:
            logger.error(f"Doc search failed: {e}")
            return None
    
    def run_web_search():
        search_keywords = ["latest", "news", "version", "release", "update"]
        bypass_phrases = ["what's the date", "what day", "current year"]
        should_search = (use_web and 
                        any(w in user_query.lower() for w in search_keywords) and 
                        not any(p in user_query.lower() for p in bypass_phrases))
        return web_search(user_query) if should_search else None
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_memory = executor.submit(run_memory_search)
        future_doc = executor.submit(run_doc_search)
        future_web = executor.submit(run_web_search)
        
        past_matches = future_memory.result()
        doc_context = future_doc.result()
        web_context = future_web.result()
    
    logger.info(f"🔍 All searches completed in {time.perf_counter() - memory_start:.2f}s")
    
    # Build optimized prompt
    build_start = time.perf_counter()
    messages = [ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT)]
    
    # Add most recent conversation context (only last 2 turns)
    if len(st.session_state.texts) >= 1:
        last_turn = st.session_state.texts[-1]
        if last_turn['user'].strip() != user_query.strip():
            messages.append(ChatMessage(
                role=MessageRole.USER,
                content=f"Previous conversation:\nYou: {last_turn['user']}"
            ))
            messages.append(ChatMessage(
                role=MessageRole.ASSISTANT,
                content=last_turn['assistant']
            ))
    
    # Add context sections if they exist
    context_parts = []
    
    # Add web context if available
    if web_context:
        context_parts.append(f"Web Search Results:\n{web_context[:1000]}")
    
    # Add relevant memory matches
    if past_matches:
        memory_context = "\n".join([f"- {m['user']}: {m['assistant']}" for m in past_matches[:3]])
        context_parts.append(f"Relevant Memories:\n{memory_context[:800]}")
    
    # Add document context if relevant
    doc_keywords = ["document", "file", "text", "resume", "application", "job"]
    if doc_context and any(keyword in user_query.lower() for keyword in doc_keywords):
        context_parts.append(f"Document Context:\n{str(doc_context)[:1000]}")
    
    # Combine all context
    if context_parts:
        context = "\n\n".join(context_parts)
        messages.append(ChatMessage(
            role=MessageRole.SYSTEM,
            content=f"Here is some relevant context for your reference:\n\n{context}"
        ))
    
    # Add current query
    messages.append(ChatMessage(role=MessageRole.USER, content=user_query))
    
    # Log prompt assembly time
    logger.info(f"📝 Prompt assembly took {time.perf_counter() - build_start:.2f}s")
    
    # Call LLM with timeout
    llm_start = time.perf_counter()
    try:
        response = Settings.llm.chat(messages)
        assistant_reply = trim_response(response.message.content)
        
        # Cache the response
        st.session_state.response_cache[cache_key] = assistant_reply
        
        # Limit cache size
        if len(st.session_state.response_cache) > 20:  # Keep last 20 responses
            oldest_key = next(iter(st.session_state.response_cache))
            del st.session_state.response_cache[oldest_key]
            
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        assistant_reply = "I'm sorry, I encountered an error processing your request."
    
    llm_time = time.perf_counter() - llm_start
    total_time = time.perf_counter() - start_total
    
    logger.info(f"⏱️  Query processed in {total_time:.1f}s (LLM: {llm_time:.1f}s)")
    
    return assistant_reply

def main():
    # Set up the embedder early and share it (use consistent embedding model)
    if "embedder" not in st.session_state:
        # Use HuggingFaceEmbedding for consistency with LlamaIndex
        embedding_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
        st.session_state.embedder = embedding_model
        Settings.embed_model = embedding_model

    # Only create LLM once and cache it
    if "llm" not in st.session_state:
        with st.spinner("Initializing LLM connection..."):
            st.session_state.llm = MistralLLM()
            # Warm up the model with a simple query to reduce first-query latency
            try:
                warmup_start = time.perf_counter()
                warmup_msg = [ChatMessage(role=MessageRole.USER, content="Hi")]
                st.session_state.llm.chat(warmup_msg)
                warmup_time = time.perf_counter() - warmup_start
                logger.info(f"Model warmup completed in {warmup_time:.1f}s")
            except Exception as e:
                logger.warning(f"Model warmup failed: {e}")
    Settings.llm = st.session_state.llm

    # Only initialize once or when needed
    init_memory_and_index()

    if "pending_code" not in st.session_state:
        st.session_state.pending_code = None

    if "processing_query" not in st.session_state:
        st.session_state.processing_query = False

    st.title("🧠 ScoutOS")
    
    # Show document indexing status in sidebar
    with st.sidebar:
        if "doc_index" in st.session_state and st.session_state.doc_index is not None:
            if hasattr(st.session_state.doc_index, 'get_stats'):
                stats = st.session_state.doc_index.get_stats()
                st.info(f"📚 Documents: {stats['num_documents']} indexed")
                if 'num_vectors' in stats:
                    st.info(f"🔍 Vectors: {stats['num_vectors']} embeddings")
            else:
                st.info("📚 Document index loaded")
        else:
            st.warning("📚 No documents indexed")
            
    chat_container = st.container()
    input_container = st.container()

    with chat_container:
        for turn in st.session_state.texts:
            # Don't show 'thinking...' responses in chat history
            if turn['assistant'] != "thinking...":
                st.markdown(f"**You:** {turn['user']}")
                st.markdown(f"**ScoutOS:** {turn['assistant']}")

    with input_container.form("chat_form", clear_on_submit=True):
        user_query = st.text_input("Ask ScoutOS something:", key="chat_input")
        use_web = st.checkbox("Augment with Web Search", value=True)
        search_memory = st.checkbox("Search Past Conversations", value=True)
        semantic_toggle = st.checkbox("Use Semantic Search", value=True)
        submitted = st.form_submit_button("Send")

    # Phase 1: Capture query
    if submitted and user_query:
        if not st.session_state.processing_query and (not st.session_state.texts or st.session_state.texts[-1]["assistant"] != "thinking..."):
            st.session_state.processing_query = True
            st.session_state.texts.append({"user": user_query, "assistant": "thinking..."})
            logger.info("🔁 Submitted and user+query rerun")
            st.rerun()

    # Phase 2: Handle "thinking..." message
    if st.session_state.processing_query and st.session_state.texts and st.session_state.texts[-1]["assistant"] == "thinking...":
        last_query = st.session_state.texts[-1]["user"]
        
        # Show processing indicator
        with st.spinner(f"Processing: {last_query[:50]}..."):
            assistant_reply = handle_query(last_query, use_web, search_memory, semantic_toggle)

        logger.info("🔁 Prcoessing query")

        # Optional: extract proposed code
        if any(kw in last_query.lower() for kw in ["update your code", "modify your startup file", "change your source"]):
            import re
            match = re.search(r"```(?:python)?\n(.*?)```", assistant_reply, re.DOTALL)
            if match:
                st.session_state.pending_code = match.group(1).strip()
                st.success("ScoutOS has proposed a code change. Please review below.")
            else:
                st.warning("ScoutOS didn't return a valid code block.")

        st.session_state.texts[-1] = {
            "timestamp": datetime.now().isoformat(),
            "user": last_query,
            "assistant": assistant_reply
        }
        st.session_state.processing_query = False
        
        # Save memory without triggering semantic memory rebuild
        save_memory(st.session_state.texts)
        
        # Update last_memory_size to prevent unnecessary rebuilds
        st.session_state.last_memory_size = len(st.session_state.texts)
        
        logger.info("🔁 Processed query true rerun")
        st.rerun()

    if st.session_state.pending_code:
        st.subheader("🛠️ Proposed Code Update from ScoutOS")
        new_code = st.text_area("Review the proposed code:", st.session_state.pending_code, height=400)
        if st.button("Save and Overwrite File"):
            result = st.session_state.semantic_memory.write_self(new_code)
            st.success(result)
            st.session_state.pending_code = None

    st.caption("ScoutOS | Local Mistral via Ollama + Flat + Semantic Memory + Web Search")

if __name__ == "__main__":
    main()