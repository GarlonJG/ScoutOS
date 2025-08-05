import textwrap
from pathlib import Path
try:
    import faiss
    from llama_index.vector_stores.faiss import FaissVectorStore
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    
import numpy as np
from llama_index.core import VectorStoreIndex, Document, SimpleDirectoryReader
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.schema import TextNode

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class SemanticDocumentIndex:
    def __init__(self, folder_path, embedding_model):
        self.folder_path = folder_path
        self.embedding_model = embedding_model
        self.docs = self.load_docs()
        self.index = None
        self.vector_dim = 384  # Fixed dimension for all-MiniLM-L6-v2
        self.chunk_size = 512  # Smaller chunks for better performance
        self.chunk_overlap = 50
        
        if self.docs:
            self.index = self.embed_and_index(self.docs)
        
    def _chunk_document(self, text, metadata):
        """Split document into smaller chunks."""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            doc = Document(
                text=chunk,
                metadata=metadata.copy()
            )
            chunks.append(doc)
        return chunks
            
    def load_docs(self):
        """Load and chunk documents from the specified folder path."""
        try:
            if not Path(self.folder_path).exists():
                logger.warning(f"Document folder '{self.folder_path}' does not exist")
                return []
                
            # Load documents
            documents = SimpleDirectoryReader(self.folder_path).load_data()
            logger.info(f"Loaded {len(documents)} documents from '{self.folder_path}'")
            
            # Chunk documents
            chunked_docs = []
            for doc in documents:
                chunks = self._chunk_document(doc.text, doc.metadata)
                chunked_docs.extend(chunks)
                
            logger.info(f"Split into {len(chunked_docs)} chunks")
            
            # Log first few chunks for debugging
            for i, doc in enumerate(chunked_docs[:3]):
                logger.info(f"Chunk {i+1}: {len(doc.text)} chars")
                
            return chunked_docs
            
        except Exception as e:
            logger.error(f"Failed to load documents from '{self.folder_path}': {e}")
            return []

    def as_query_engine(self, **kwargs):
        """Return a query engine for the indexed documents."""
        if self.index is None:
            logger.warning("No index available - documents may not have loaded properly")
            return None
        return self.index.as_query_engine(**kwargs)

    def embed_and_index(self, documents):
        """Create optimized index from documents with batching and memory management."""
        if not documents:
            logger.warning("No documents to index")
            return None
            
        start_time = time.time()
        logger.info(f"Starting to index {len(documents)} document chunks...")
        
        try:
            # Process in batches to manage memory
            batch_size = 8  # Smaller batch size for better memory management
            nodes = []
            
            # Use a simple text splitter
            text_splitter = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            # Process documents in batches
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                
                # Split documents into nodes
                for doc in batch_docs:
                    nodes.extend(text_splitter.get_nodes_from_documents([doc]))
                
                logger.info(f"Processed {min(i + batch_size, len(documents))}/{len(documents)} documents")
            
            # Create index with optimized settings for CPU
            index = VectorStoreIndex(
                nodes=nodes,
                embed_model=self.embedding_model,
                show_progress=True
            )
            
            logger.info(f"Indexing completed in {time.time() - start_time:.2f}s")
            return index
            
        except Exception as e:
            logger.error(f"Error during indexing: {e}")
            # Fallback to simple in-memory index if available
            try:
                logger.info("Falling back to simple in-memory index")
                return VectorStoreIndex.from_documents(
                    documents,
                    embed_model=self.embedding_model
                )
            except Exception as fallback_error:
                logger.error(f"Fallback index creation failed: {fallback_error}")
                return None
            
            if hasattr(index, '_vector_store') and hasattr(index._vector_store, '_faiss_index'):
                logger.info(f"Successfully created index with {index._vector_store._faiss_index.ntotal} vectors")
            else:
                logger.info("Successfully created index")
            
            logger.info(f"Document indexing completed successfully")
            
            return index
            
        except Exception as e:
            logger.error(f"Failed to create document index: {e}")
            return None

    def search(self, query_text, top_k=3):
        """Search documents using custom retrieval + LLM approach (bypassing LlamaIndex query engine issues)."""
        if self.index is None:
            logger.warning("Cannot search - no index available")
            return None
            
        try:
            # Import Settings to get the LLM
            from llama_index.core import Settings
            from llama_index.core.base.llms.types import ChatMessage, MessageRole
            
            # Step 1: Retrieve relevant documents
            retriever = self.index.as_retriever(similarity_top_k=top_k)
            nodes = retriever.retrieve(query_text)
            
            if not nodes:
                return "No relevant documents found for your query."
            
            # Step 2: Combine document content
            doc_content = "\n\n".join([f"Document {i+1}: {node.text}" for i, node in enumerate(nodes)])
            
            # Step 3: Create a comprehensive prompt
            prompt = f"""Based on the following document(s), please answer the user's question.

Document(s):
{doc_content}

User Question: {query_text}

Please provide a helpful and accurate answer based on the document content above."""
            
            # Step 4: Query the LLM directly
            if hasattr(Settings, 'llm') and Settings.llm is not None:
                messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
                llm_response = Settings.llm.chat(messages)
                response_content = llm_response.message.content
                
                # Create a response object that matches expected format
                class CustomResponse:
                    def __init__(self, content, source_nodes):
                        self.response = content
                        self.source_nodes = source_nodes
                        
                    def __str__(self):
                        return self.response
                
                return CustomResponse(response_content, nodes)
            else:
                return f"Found relevant content: {doc_content[:500]}..."
                
        except Exception as e:
            logger.error(f"Custom document search failed: {e}")
            return f"Search failed: {str(e)}"
            
    def get_stats(self):
        """Get statistics about the indexed documents."""
        stats = {
            "num_documents": len(self.docs) if self.docs else 0,
            "index_available": self.index is not None,
            "folder_path": self.folder_path
        }
        
        if self.index and hasattr(self.index, '_vector_store'):
            try:
                vector_store = self.index._vector_store
                if hasattr(vector_store, '_faiss_index'):
                    stats["num_vectors"] = vector_store._faiss_index.ntotal
            except:
                pass
                
        return stats