from typing import List, Optional
import json
import os
from pathlib import Path

class APIKeyRotator:
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.keys_file = Path(__file__).parent.parent / 'config' / 'api_keys.json'
        self.keys = self._load_keys()
        self.current_key_index = 0
    
    def _load_keys(self) -> List[str]:
        """Load API keys from secure storage"""
        if not self.keys_file.exists():
            return []
        
        with open(self.keys_file, 'r') as f:
            keys_data = json.load(f)
            return keys_data.get(self.service_name, [])
    
    def get_next_key(self) -> Optional[str]:
        """Get next available API key"""
        if not self.keys:
            return None
        
        self.current_key_index = (self.current_key_index + 1) % len(self.keys)
        return self.keys[self.current_key_index]
    
    def mark_key_invalid(self, key: str):
        """Mark a key as invalid and remove it from rotation"""
        if key in self.keys:
            self.keys.remove(key)
            self._save_keys()
    
    def _save_keys(self):
        """Save updated keys to secure storage"""
        if not self.keys_file.parent.exists():
            self.keys_file.parent.mkdir(parents=True)
        
        current_data = {}
        if self.keys_file.exists():
            with open(self.keys_file, 'r') as f:
                current_data = json.load(f)
        
        current_data[self.service_name] = self.keys
        
        with open(self.keys_file, 'w') as f:
            json.dump(current_data, f) 