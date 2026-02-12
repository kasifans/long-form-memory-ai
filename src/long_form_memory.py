"""
Long-Form Memory System - Main orchestrator that ties everything together
"""
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from memory_model import Memory, ConversationTurn
from memory_extractor import MemoryExtractor
from memory_storage import MemoryStorage
from memory_retriever import MemoryRetriever


class LongFormMemorySystem:
    """
    Main system that coordinates extraction, storage, and retrieval of memories
    across long conversations (1000+ turns)
    """
    
    def __init__(
        self,
        db_path: str = "memory_store.db",
        llm_client=None,
        embedding_model=None,
        top_k: int = 5,
        auto_extract: bool = True
    ):
        # Initialize components
        self.storage = MemoryStorage(db_path)
        self.extractor = MemoryExtractor(llm_client)
        self.retriever = MemoryRetriever(self.storage, embedding_model, max_results=top_k)
        
        self.auto_extract = auto_extract
        self.turn_count = 0
        self.history = []  # Conversation history
        
        # Track performance metrics
        self.perf_metrics = {
            "extraction_times": [],
            "retrieval_times": [],
            "total_extracted": 0,
            "total_retrieved": 0
        }
    
    def process_turn(
        self,
        user_msg: str,
        assistant_msg: Optional[str] = None,
        should_extract: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single conversation turn
        
        Returns dict with:
        - turn_id
        - extracted_memories
        - retrieved_memories
        - extraction_time (ms)
        - retrieval_time (ms)
        """
        self.turn_count += 1
        
        result = {
            "turn_id": self.turn_count,
            "extracted_memories": [],
            "retrieved_memories": [],
            "extraction_time": 0,
            "retrieval_time": 0
        }
        
        # Step 1: Retrieve relevant memories
        t_start = time.time()
        relevant_mems = self.retrieve_memories(user_msg)
        t_elapsed = (time.time() - t_start) * 1000  # Convert to ms
        
        result["retrieved_memories"] = relevant_mems
        result["retrieval_time"] = round(t_elapsed, 2)
        
        self.perf_metrics["retrieval_times"].append(t_elapsed)
        self.perf_metrics["total_retrieved"] += 1
        
        # Step 2: Extract new memories if requested
        if should_extract and self.auto_extract:
            t_start = time.time()
            
            asst = assistant_msg or ""
            new_mems = self.extractor.extract_memories(
                user_msg,
                asst,
                self.turn_count,
                use_llm=False  # Currently using pattern matching
            )
            
            # Save to storage
            saved_count = self.storage.save_batch(new_mems)
            
            t_elapsed = (time.time() - t_start) * 1000
            
            result["extracted_memories"] = new_mems
            result["extraction_time"] = round(t_elapsed, 2)
            
            self.perf_metrics["extraction_times"].append(t_elapsed)
            self.perf_metrics["total_extracted"] += saved_count
        
        # Record this turn
        turn = ConversationTurn(
            turn_id=self.turn_count,
            user_message=user_msg,
            assistant_message=assistant_msg,
            timestamp=datetime.now().isoformat(),
            extracted_memories=[m.memory_id for m in result["extracted_memories"]],
            retrieved_memories=[m.memory_id for m in result["retrieved_memories"]]
        )
        self.history.append(turn)
        
        return result
    
    def retrieve_memories(
        self,
        query: str,
        types: Optional[List[str]] = None,
        min_conf: float = 0.5
    ) -> List[Memory]:
        """Retrieve relevant memories for a query"""
        return self.retriever.retrieve(
            query,
            self.turn_count,
            filter_types=types,
            min_confidence=min_conf
        )
    
    def format_for_prompt(self, memories: List[Memory], style: str = "natural") -> str:
        """
        Format memories for injection into LLM prompt
        
        Args:
            memories: List of Memory objects
            style: "natural" or "structured"
        """
        if not memories:
            return ""
        
        if style == "natural":
            lines = ["Based on what I know about you:"]
            
            for mem in memories:
                if mem.type == "preference":
                    lines.append(f"- You prefer {mem.value}")
                elif mem.type == "fact":
                    lines.append(f"- {mem.value}")
                elif mem.type == "commitment":
                    lines.append(f"- You have committed to: {mem.value}")
                elif mem.type == "constraint":
                    lines.append(f"- Constraint: {mem.value}")
                elif mem.type == "instruction":
                    lines.append(f"- Standing instruction: {mem.value}")
                else:
                    lines.append(f"- {mem.key}: {mem.value}")
            
            return "\n".join(lines)
        
        else:  # structured
            lines = ["Relevant context from conversation:"]
            for i, mem in enumerate(memories, 1):
                lines.append(
                    f"{i}. [{mem.type}] {mem.key}: {mem.value} "
                    f"(turn {mem.source_turn}, conf: {mem.confidence:.2f})"
                )
            
            return "\n".join(lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics and performance metrics"""
        storage_stats = self.storage.get_stats()
        
        # Calculate averages
        avg_extraction = (
            sum(self.perf_metrics["extraction_times"]) / len(self.perf_metrics["extraction_times"])
            if self.perf_metrics["extraction_times"] else 0
        )
        
        avg_retrieval = (
            sum(self.perf_metrics["retrieval_times"]) / len(self.perf_metrics["retrieval_times"])
            if self.perf_metrics["retrieval_times"] else 0
        )
        
        return {
            "current_turn": self.turn_count,
            "total_memories": storage_stats["total_memories"],
            "memories_by_type": storage_stats["by_type"],
            "average_confidence": storage_stats["average_confidence"],
            "total_extractions": self.perf_metrics["total_extracted"],
            "total_retrievals": self.perf_metrics["total_retrieved"],
            "avg_extraction_time_ms": round(avg_extraction, 2),
            "avg_retrieval_time_ms": round(avg_retrieval, 2),
            "conversation_history_length": len(self.history)
        }
    
    def export_memories(self, filepath: str) -> bool:
        """Export all memories to JSON file"""
        import json
        
        try:
            all_mems = self.storage.get_all()
            data = {
                "export_timestamp": datetime.now().isoformat(),
                "total_turns": self.turn_count,
                "memories": [m.to_dict() for m in all_mems]
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Export failed: {e}")
            return False
    
    def reset(self):
        """Reset the system (clears in-memory state, not database)"""
        self.turn_count = 0
        self.history.clear()
        self.perf_metrics = {
            "extraction_times": [],
            "retrieval_times": [],
            "total_extracted": 0,
            "total_retrieved": 0
        }
    
    def close(self):
        """Clean shutdown"""
        self.storage.close()