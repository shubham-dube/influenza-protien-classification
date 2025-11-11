#!/usr/bin/env python3
"""
cache_manager.py
---------------
Manage processing cache to avoid reprocessing completed maps.
"""

import json
import os
import hashlib
from typing import Dict, Optional
from datetime import datetime


class CacheManager:
    """Manage cache for processed maps."""
    
    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Load cache from disk."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARN] Could not load cache: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to disk."""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def _compute_hash(self, filepath: str) -> str:
        """Compute file hash for change detection."""
        hash_md5 = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""
    
    def is_processed(self, filepath: str) -> bool:
        """Check if file has been processed."""
        file_key = os.path.basename(filepath)
        
        if file_key not in self.cache:
            return False
        
        # Check if file has been modified
        cached_hash = self.cache[file_key].get("hash", "")
        current_hash = self._compute_hash(filepath)
        
        return cached_hash == current_hash
    
    def mark_processed(self, filepath: str, metadata: Optional[Dict] = None):
        """Mark file as processed."""
        file_key = os.path.basename(filepath)
        
        self.cache[file_key] = {
            "hash": self._compute_hash(filepath),
            "processed_at": datetime.now().isoformat(),
            "filepath": filepath,
        }
        
        if metadata:
            self.cache[file_key].update(metadata)
        
        self._save_cache()
    
    def get_metadata(self, filepath: str) -> Optional[Dict]:
        """Get cached metadata for a file."""
        file_key = os.path.basename(filepath)
        return self.cache.get(file_key)
    
    def clear_cache(self):
        """Clear all cache entries."""
        self.cache = {}
        self._save_cache()
    
    def remove_entry(self, filepath: str):
        """Remove specific cache entry."""
        file_key = os.path.basename(filepath)
        if file_key in self.cache:
            del self.cache[file_key]
            self._save_cache()
    
    def get_statistics(self) -> Dict:
        """Get cache statistics."""
        return {
            "total_processed": len(self.cache),
            "cache_file": self.cache_file,
            "cache_size_bytes": os.path.getsize(self.cache_file) if os.path.exists(self.cache_file) else 0
        }