"""
Memory Retriever - Finds relevant memories based on current context
"""
import numpy as np
from typing import List, Dict, Any, Optional
import re

from memory_model import Memory
from memory_storage import MemoryStorage


class MemoryRetriever:
    """
    Retrieves relevant memories using a scoring system that considers:
    - Keyword overlap
    - Recency (newer is better)
    - Frequency (often-used is important)
    - Confidence score
    """
    
    def __init__(self, storage: MemoryStorage, embedding_model=None, max_results: int = 5):
        self.storage = storage
        self.embedding_model = embedding_model
        self.max_results = max_results
    
    def retrieve(
        self,
        query: str,
        current_turn: int,
        filter_types: Optional[List[str]] = None,
        min_confidence: float = 0.5
    ) -> List[Memory]:
        """
        Find the most relevant memories for a given query
        """
        # Get all active memories
        all_mems = self.storage.get_all(active_only=True)
        
        # Apply filters
        if filter_types:
            all_mems = [m for m in all_mems if m.type in filter_types]
        
        all_mems = [m for m in all_mems if m.confidence >= min_confidence]
        
        if not all_mems:
            return []
        
        # Score each memory
        scored = []
        for mem in all_mems:
            score = self._score_relevance(mem, query, current_turn)
            scored.append((mem, score))
        
        # Sort by score and take top results
        scored.sort(key=lambda x: x[1], reverse=True)
        top_mems = [m for m, s in scored[:self.max_results] if s > 0]
        
        # Update access tracking
        for mem in top_mems:
            self.storage.mark_accessed(mem.memory_id, current_turn)
        
        return top_mems
    
    def _score_relevance(self, mem: Memory, query: str, turn: int) -> float:
        """
        Calculate how relevant this memory is to the current query
        Combines multiple factors with weights
        """
        scores = []
        weights = []
        
        # Factor 1: Keyword matching (30%)
        kw_score = self._keyword_match(query, mem)
        scores.append(kw_score)
        weights.append(0.3)
        
        # Factor 2: Semantic similarity if available (30%)
        if mem.embedding is not None and self.embedding_model:
            sem_score = self._semantic_match(query, mem)
            scores.append(sem_score)
            weights.append(0.3)
        
        # Factor 3: How recent is this memory? (15%)
        recency = self._recency_score(mem, turn)
        scores.append(recency)
        weights.append(0.15)
        
        # Factor 4: How often has it been used? (10%)
        freq = self._frequency_score(mem)
        scores.append(freq)
        weights.append(0.1)
        
        # Factor 5: Confidence of extraction (15%)
        scores.append(mem.confidence)
        weights.append(0.15)
        
        # Calculate weighted average
        total_weight = sum(weights)
        norm_weights = [w / total_weight for w in weights]
        
        final = sum(s * w for s, w in zip(scores, norm_weights))
        return final
    
    def _keyword_match(self, query: str, mem: Memory) -> float:
        """Calculate overlap between query words and memory content"""
        query_words = set(self._tokenize(query.lower()))
        mem_words = set(self._tokenize(f"{mem.key} {mem.value}".lower()))
        
        if not query_words or not mem_words:
            return 0.0
        
        # Jaccard similarity
        overlap = query_words & mem_words
        union = query_words | mem_words
        
        return len(overlap) / len(union) if union else 0.0
    
    def _semantic_match(self, query: str, mem: Memory) -> float:
        """Calculate semantic similarity using embeddings"""
        if mem.embedding is None or self.embedding_model is None:
            return 0.0
        
        try:
            query_vec = self.embedding_model.encode(query)
            mem_vec = np.array(mem.embedding)
            
            # Cosine similarity
            dot = np.dot(query_vec, mem_vec)
            q_norm = np.linalg.norm(query_vec)
            m_norm = np.linalg.norm(mem_vec)
            
            if q_norm == 0 or m_norm == 0:
                return 0.0
            
            sim = dot / (q_norm * m_norm)
            
            # Normalize from [-1, 1] to [0, 1]
            return (sim + 1) / 2
            
        except Exception as e:
            print(f"Semantic similarity error: {e}")
            return 0.0
    
    def _recency_score(self, mem: Memory, current_turn: int) -> float:
        """Score based on how recent the memory is"""
        turns_ago = current_turn - mem.source_turn
        
        # Exponential decay - half-life of 100 turns
        half_life = 100
        decay = 0.693 / half_life  # ln(2) / half_life
        
        score = np.exp(-decay * turns_ago)
        return max(0.0, min(1.0, score))
    
    def _frequency_score(self, mem: Memory) -> float:
        """Score based on how often this memory has been accessed"""
        if mem.access_count == 0:
            return 0.1  # Small baseline for new memories
        
        # Logarithmic scale - maxes out around 20 accesses
        score = min(1.0, np.log1p(mem.access_count) / np.log1p(20))
        return score
    
    def _tokenize(self, text: str) -> List[str]:
        """Break text into words, removing stopwords"""
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        
        # Common words to ignore
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were'
        }
        
        return [w for w in words if w.lower() not in stopwords]
    
    def get_by_type(self, mem_type: str, current_turn: int, limit: int = 5) -> List[Memory]:
        """Get memories of a specific type, sorted by relevance"""
        mems = self.storage.find_by_type(mem_type)
        
        # Score by recency + confidence
        scored = []
        for m in mems:
            recency = self._recency_score(m, current_turn)
            score = (recency + m.confidence) / 2
            scored.append((m, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored[:limit]]
    
    def get_recent(self, current_turn: int, window: int = 10) -> List[Memory]:
        """Get memories from recent turns"""
        all_mems = self.storage.get_all(active_only=True)
        
        start_turn = max(0, current_turn - window)
        recent = [m for m in all_mems if m.source_turn >= start_turn]
        
        # Sort by turn number (newest first)
        recent.sort(key=lambda m: m.source_turn, reverse=True)
        
        return recent[:self.max_results]