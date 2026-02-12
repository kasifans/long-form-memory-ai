"""
Memory Storage - Handles persistence of memories using SQLite
"""
import json
import sqlite3
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

from memory_model import Memory, MemoryType


class MemoryStorage:
    """
    Stores memories in a hybrid system:
    - SQLite for structured data
    - In-memory dict for embeddings (future use)
    """
    
    def __init__(self, db_path: str = "memory_store.db"):
        self.db_path = db_path
        self.conn = None
        self.embeddings = {}  # memory_id -> vector
        self.embedding_size = 384  # Standard sentence embedding size
        
        self._setup_database()
    
    def _setup_database(self):
        """Initialize the database and create tables"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.conn.cursor()
        
        # Main memories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                source_turn INTEGER NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL,
                last_accessed_turn INTEGER,
                access_count INTEGER DEFAULT 0,
                metadata TEXT,
                active INTEGER DEFAULT 1
            )
        """)
        
        # Create indexes for faster lookups
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_type ON memories(type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_source_turn ON memories(source_turn)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_active ON memories(active)")
        
        self.conn.commit()
    
    def save(self, memory: Memory) -> bool:
        """Save a single memory to the database"""
        try:
            cursor = self.conn.cursor()
            
            # Convert metadata to JSON if present
            meta_json = json.dumps(memory.metadata) if memory.metadata else None
            
            cursor.execute("""
                INSERT OR REPLACE INTO memories 
                (memory_id, type, key, value, source_turn, confidence, 
                 created_at, last_accessed_turn, access_count, metadata, active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory.memory_id,
                memory.type,
                memory.key,
                memory.value,
                memory.source_turn,
                memory.confidence,
                memory.created_at,
                memory.last_accessed_turn,
                memory.access_count,
                meta_json,
                1
            ))
            
            # Store embedding separately if available
            if memory.embedding is not None:
                self.embeddings[memory.memory_id] = np.array(memory.embedding)
            
            self.conn.commit()
            return True
            
        except Exception as e:
            print(f"Error saving memory: {e}")
            return False
    
    def save_batch(self, memories: List[Memory]) -> int:
        """Save multiple memories at once"""
        saved = 0
        for mem in memories:
            if self.save(mem):
                saved += 1
        return saved
    
    def get(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a specific memory by ID"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT memory_id, type, key, value, source_turn, confidence,
                   created_at, last_accessed_turn, access_count, metadata
            FROM memories
            WHERE memory_id = ? AND active = 1
        """, (memory_id,))
        
        row = cursor.fetchone()
        if row is None:
            return None
        
        # Reconstruct the Memory object
        mem = Memory(
            memory_id=row[0],
            type=row[1],
            key=row[2],
            value=row[3],
            source_turn=row[4],
            confidence=row[5],
            created_at=row[6],
            last_accessed_turn=row[7],
            access_count=row[8],
            metadata=json.loads(row[9]) if row[9] else None,
            embedding=self.embeddings.get(row[0])
        )
        
        return mem
    
    def get_all(self, active_only: bool = True) -> List[Memory]:
        """Get all memories from the database"""
        cursor = self.conn.cursor()
        
        query = "SELECT memory_id FROM memories"
        if active_only:
            query += " WHERE active = 1"
        
        cursor.execute(query)
        ids = [row[0] for row in cursor.fetchall()]
        
        # Fetch each memory
        results = []
        for mem_id in ids:
            mem = self.get(mem_id)
            if mem:
                results.append(mem)
        
        return results
    
    def find_by_type(self, mem_type: str) -> List[Memory]:
        """Find all memories of a specific type"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT memory_id FROM memories
            WHERE type = ? AND active = 1
        """, (mem_type,))
        
        ids = [row[0] for row in cursor.fetchall()]
        return [self.get(mid) for mid in ids if self.get(mid)]
    
    def search_by_key(self, search_key: str) -> List[Memory]:
        """Search for memories with matching keys"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT memory_id FROM memories
            WHERE key LIKE ? AND active = 1
        """, (f"%{search_key}%",))
        
        ids = [row[0] for row in cursor.fetchall()]
        return [self.get(mid) for mid in ids if self.get(mid)]
    
    def mark_accessed(self, memory_id: str, turn_num: int) -> bool:
        """Update when this memory was last used"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE memories
                SET last_accessed_turn = ?,
                    access_count = access_count + 1
                WHERE memory_id = ?
            """, (turn_num, memory_id))
            
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error updating access: {e}")
            return False
    
    def deactivate(self, memory_id: str) -> bool:
        """Soft delete - mark memory as inactive"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE memories
                SET active = 0
                WHERE memory_id = ?
            """, (memory_id,))
            
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error deactivating memory: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories"""
        cursor = self.conn.cursor()
        
        # Total count
        cursor.execute("SELECT COUNT(*) FROM memories WHERE active = 1")
        total = cursor.fetchone()[0]
        
        # Count by type
        cursor.execute("""
            SELECT type, COUNT(*) 
            FROM memories 
            WHERE active = 1 
            GROUP BY type
        """)
        by_type = dict(cursor.fetchall())
        
        # Average confidence
        cursor.execute("""
            SELECT AVG(confidence) 
            FROM memories 
            WHERE active = 1
        """)
        avg_conf = cursor.fetchone()[0] or 0
        
        return {
            "total_memories": total,
            "by_type": by_type,
            "average_confidence": round(avg_conf, 3),
            "vector_store_size": len(self.embeddings)
        }
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.close()