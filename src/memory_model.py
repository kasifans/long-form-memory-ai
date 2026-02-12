"""
Memory Model - Core data structures for the memory system
"""
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime
import json


@dataclass
class Memory:
    """
    Represents a single memory extracted from conversation
    """
    memory_id: str
    type: str  # Type of memory (preference, fact, etc.)
    key: str
    value: str
    source_turn: int
    confidence: float
    created_at: str
    last_accessed_turn: Optional[int] = None
    access_count: int = 0
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string (without embedding to save space)"""
        data = self.to_dict()
        if 'embedding' in data:
            data['embedding'] = None  # Too large for JSON
        return json.dumps(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create Memory from dictionary"""
        return cls(**data)
    
    def update_access(self, turn: int):
        """Update when this memory was last accessed"""
        self.last_accessed_turn = turn
        self.access_count += 1


@dataclass
class ConversationTurn:
    """
    Represents a single turn in the conversation
    """
    turn_id: int
    user_message: str
    assistant_message: Optional[str] = None
    timestamp: Optional[str] = None
    extracted_memories: Optional[List[str]] = None
    retrieved_memories: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict"""
        return asdict(self)


class MemoryType:
    """
    Constants for different types of memories we can extract
    """
    PREFERENCE = "preference"
    FACT = "fact"
    ENTITY = "entity"
    CONSTRAINT = "constraint"
    COMMITMENT = "commitment"
    INSTRUCTION = "instruction"
    
    @classmethod
    def all_types(cls) -> List[str]:
        """Get list of all memory types"""
        return [
            cls.PREFERENCE,
            cls.FACT,
            cls.ENTITY,
            cls.CONSTRAINT,
            cls.COMMITMENT,
            cls.INSTRUCTION
        ]