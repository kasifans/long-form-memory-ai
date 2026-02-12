"""
Memory Extractor - Identifies and extracts important information from conversations
"""
import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib

from memory_model import Memory, MemoryType


class MemoryExtractor:
    """
    Extracts memorable information from conversation turns
    Currently uses pattern matching, can be upgraded to use LLM
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.extraction_prompt = self._build_extraction_prompt()
    
    def _build_extraction_prompt(self) -> str:
        """Build the prompt for LLM-based extraction"""
        prompt = """You are a memory extraction system. Extract ONLY important information worth remembering.

Memory Types:
- preference: User preferences (e.g., "prefers calls after 11 AM")
- fact: Facts about the user (e.g., "lives in San Francisco")
- entity: Important people/places (e.g., "mother's name is Sarah")
- constraint: Limitations (e.g., "cannot work weekends")
- commitment: Plans/promises (e.g., "meeting Friday at 2 PM")
- instruction: Standing rules (e.g., "always use formal tone")

Conversation:
User: {user_message}
Assistant: {assistant_message}

Return JSON array of memories (or [] if nothing important):
[
  {{
    "type": "preference",
    "key": "language_preference",
    "value": "Kannada",
    "confidence": 0.95,
    "rationale": "User explicitly stated"
  }}
]

Be selective - casual chat doesn't need to be stored."""
        
        return prompt
    
    def extract_memories(
        self, 
        user_msg: str, 
        assistant_msg: str, 
        turn_num: int,
        use_llm: bool = True
    ) -> List[Memory]:
        """
        Extract memories from a conversation turn
        """
        if use_llm and self.llm_client:
            return self._extract_with_llm(user_msg, assistant_msg, turn_num)
        else:
            # Fallback to pattern matching
            return self._extract_with_patterns(user_msg, assistant_msg, turn_num)
    
    def _extract_with_llm(self, user_msg: str, asst_msg: str, turn_num: int) -> List[Memory]:
        """Use LLM to extract memories (if available)"""
        prompt = self.extraction_prompt.format(
            user_message=user_msg,
            assistant_message=asst_msg
        )
        
        try:
            response = self._call_llm(prompt)
            extracted = self._parse_llm_response(response)
            
            memories = []
            for item in extracted:
                mem_id = self._make_memory_id(item['key'], turn_num)
                mem = Memory(
                    memory_id=mem_id,
                    type=item['type'],
                    key=item['key'],
                    value=item['value'],
                    source_turn=turn_num,
                    confidence=item.get('confidence', 0.8),
                    created_at=datetime.now().isoformat(),
                    metadata={'rationale': item.get('rationale', '')}
                )
                memories.append(mem)
            
            return memories
            
        except Exception as err:
            print(f"LLM extraction failed: {err}. Using pattern matching instead.")
            return self._extract_with_patterns(user_msg, asst_msg, turn_num)
    
    def _extract_with_patterns(self, user_msg: str, asst_msg: str, turn_num: int) -> List[Memory]:
        """
        Extract memories using regex patterns
        This is the fallback when LLM isn't available
        """
        memories = []
        msg_lower = user_msg.lower()
        
        # Skip short or casual messages
        boring_phrases = [
            "how are you", "how's the weather", "what's the latest", 
            "tell me a joke", "what day is it", "what can you help",
            "that's interesting", "thanks", "i see", "okay", "sure", 
            "can you explain", "here to help"
        ]
        
        if len(user_msg.split()) < 5:
            return memories
            
        for phrase in boring_phrases:
            if phrase in msg_lower:
                return memories
        
        # Only extract from user messages (not assistant)
        text = msg_lower
        
        # Look for preferences
        pref_patterns = [
            r"(?:my |i )prefer (?:to )?(.+?)(?:\.|$|,)",
            r"(?:always|never) (.+?)(?:\.|$|,)",
            r"(?:language is|speak|communicate in) ([a-z]+)",
        ]
        
        for pattern in pref_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match.strip()) > 3:
                    mem_id = self._make_memory_id(f"pref_{match}", turn_num)
                    key = f"preference_{match[:20].replace(' ', '_')}"
                    memories.append(Memory(
                        memory_id=mem_id,
                        type=MemoryType.PREFERENCE,
                        key=key,
                        value=match.strip(),
                        source_turn=turn_num,
                        confidence=0.85,
                        created_at=datetime.now().isoformat()
                    ))
        
        # Look for facts about the user
        fact_patterns = [
            r"(?:my name is|i am|i'm) ([a-z ]{3,})",
            r"(?:i live in|i'm from|from) ([a-z ]{3,})",
            r"(?:i work at|work for) ([a-z ]{3,})",
            r"allergic to ([a-z]+)",
            r"(?:i'm|i am) (?:a |an )?([a-z]+ (?:engineer|developer|designer|manager))",
        ]
        
        for pattern in fact_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match.strip()) > 2:
                    mem_id = self._make_memory_id(f"fact_{match}", turn_num)
                    key = f"user_{match[:20].replace(' ', '_')}"
                    memories.append(Memory(
                        memory_id=mem_id,
                        type=MemoryType.FACT,
                        key=key,
                        value=match.strip(),
                        source_turn=turn_num,
                        confidence=0.8,
                        created_at=datetime.now().isoformat()
                    ))
        
        # Look for commitments/plans
        commitment_patterns = [
            r"(?:meeting|call|appointment).+?(?:at|@) ([0-9]+\s*(?:am|pm|AM|PM))",
            r"(?:every|each) ([a-z]+day).+?([0-9]+\s*(?:am|pm|AM|PM))",
            r"birthday.+?on ([a-z]+ [0-9]+)",
        ]
        
        for pattern in commitment_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                val = match if isinstance(match, str) else ' '.join(match)
                if len(val.strip()) > 2:
                    mem_id = self._make_memory_id(f"commitment_{val}", turn_num)
                    memories.append(Memory(
                        memory_id=mem_id,
                        type=MemoryType.COMMITMENT,
                        key=f"commitment_{len(memories)}",
                        value=val.strip(),
                        source_turn=turn_num,
                        confidence=0.75,
                        created_at=datetime.now().isoformat()
                    ))
        
        return memories
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM (placeholder for now)"""
        if self.llm_client is None:
            raise ValueError("No LLM client configured")
        
        # TODO: Implement actual LLM call here
        # For OpenAI: self.llm_client.chat.completions.create(...)
        return "[]"
    
    def _parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse JSON from LLM response"""
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            return []
        except json.JSONDecodeError:
            return []
    
    def _make_memory_id(self, key: str, turn: int) -> str:
        """Generate a unique ID for this memory"""
        unique_str = f"{key}_{turn}_{datetime.now().isoformat()}"
        hash_val = hashlib.md5(unique_str.encode()).hexdigest()[:8]
        return f"mem_{hash_val}"