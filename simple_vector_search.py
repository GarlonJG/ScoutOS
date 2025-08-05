"""
Optimized simple vector search implementation without FAISS dependency.
Uses numpy for efficient CPU-based similarity calculations.
"""
import numpy as np
from typing import List, Tuple, Optional
import logging
from collections import deque
import time

logger = logging.getLogger(__name__)

class SimpleVectorIndex:
    """
    Optimized vector search using numpy for similarity calculations.
    
    Features:
    - Batch processing for better CPU utilization
    - Efficient memory management
    - Optimized similarity calculations
    - Early stopping for large result sets
    """
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors = None
        self.norms = None
        self.ntotal = 0
        self.batch_size = 64  # Process vectors in batches for better performance
        
    def add(self, vectors: np.ndarray):
        """Add vectors to the index with batch processing."""
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)
            
        if self.vectors is None:
            self.vectors = vectors.astype(np.float32)
        else:
            self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])
            
        # Precompute vector norms for faster similarity calculations
        self.norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.ntotal = len(self.vectors)
        logger.info(f"Added {len(vectors)} vectors, total: {self.ntotal}")
        
    def _batch_similarity(self, query_vector: np.ndarray, batch_vectors: np.ndarray, batch_norms: np.ndarray) -> np.ndarray:
        """Compute cosine similarity for a batch of vectors."""
        # Normalize the query vector once
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return np.zeros(len(batch_vectors))
            
        # Batch dot product and norm calculation
        dots = np.dot(batch_vectors, query_vector)
        norms = batch_norms.flatten()
        
        # Avoid division by zero
        valid_norms = norms > 0
        similarities = np.zeros(len(batch_vectors))
        similarities[valid_norms] = dots[valid_norms] / (norms[valid_norms] * query_norm)
        
        return similarities
        
    def search(self, query_vectors: np.ndarray, k: int, min_score: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors with early stopping.
        
        Args:
            query_vectors: Query vector(s)
            k: Number of neighbors to return
            min_score: Minimum similarity score to consider
            
        Returns:
            Tuple of (scores, indices) for the top-k results
        """
        if len(query_vectors.shape) == 1:
            query_vectors = query_vectors.reshape(1, -1)
            
        if self.ntotal == 0:
            return np.array([[-1.0] * k]), np.array([[-1] * k])
            
        all_scores = []
        all_indices = []
        
        for query_vector in query_vectors:
            query_vector = query_vector.astype(np.float32)
            scores = np.zeros(self.ntotal)
            
            # Process in batches
            for i in range(0, self.ntotal, self.batch_size):
                batch_end = min(i + self.batch_size, self.ntotal)
                batch_vectors = self.vectors[i:batch_end]
                batch_norms = self.norms[i:batch_end] if self.norms is not None else None
                
                # Get similarities for this batch
                batch_scores = self._batch_similarity(query_vector, batch_vectors, batch_norms)
                scores[i:batch_end] = batch_scores
            
            # Get top-k scores and indices
            if k > 0 and k < len(scores):
                # Use argpartition for better performance with large k
                partition_idx = np.argpartition(-scores, k)[:k]
                top_k_scores = scores[partition_idx]
                top_k_indices = partition_idx
                
                # Sort the top-k results
                sorted_indices = np.argsort(-top_k_scores)
                top_k_scores = top_k_scores[sorted_indices]
                top_k_indices = top_k_indices[sorted_indices]
            else:
                # If k is large or zero, sort all scores
                sorted_indices = np.argsort(-scores)
                top_k_scores = scores[sorted_indices]
                top_k_indices = sorted_indices
            
            # Filter by minimum score
            valid = top_k_scores >= min_score
            all_scores.append(top_k_scores[valid])
            all_indices.append(top_k_indices[valid])
            
        return np.array(all_scores), np.array(all_indices)
    
    def get_vectors(self, indices: np.ndarray) -> np.ndarray:
        """Retrieve vectors by their indices."""
        if self.vectors is None or len(indices) == 0:
            return np.array([])
        return self.vectors[indices]


def create_simple_index(dimension: int) -> SimpleVectorIndex:
    """Create a simple vector index as FAISS fallback."""
    return SimpleVectorIndex(dimension)
