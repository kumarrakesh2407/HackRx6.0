import json
import os
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict, dataclass
from pathlib import Path
from .text_processor import TextChunk, TextProcessor

@dataclass
class SearchResult:
    """Represents a search result with its score and metadata."""
    text: str
    metadata: Dict[str, Any]
    score: float

class DocumentStore:
    """Manages storage and retrieval of document chunks with vector search capabilities."""
    
    def __init__(self, persist_dir: str = "data/document_store"):
        """
        Initialize the document store.
        
        Args:
            persist_dir: Directory to persist the document store
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize FAISS index
        self.dimension = 384  # Dimension of the embeddings (from all-MiniLM-L6-v2)
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Store metadata for each vector
        self.metadata_store = []
        self.text_processor = TextProcessor()
    
    def add_document(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """
        Process and add a document to the store.
        
        Args:
            text: Document text
            metadata: Optional metadata for the document
            
        Returns:
            List of added text chunks
        """
        if metadata is None:
            metadata = {}
            
        # Process the document into chunks with embeddings
        chunks = self.text_processor.process_document(text, metadata)
        
        if not chunks:
            return []
        
        # Add to FAISS index
        embeddings = np.array([chunk.embedding for chunk in chunks], dtype=np.float32)
        self.index.add(embeddings)
        
        # Store metadata
        for chunk in chunks:
            chunk_dict = asdict(chunk)
            chunk_dict['embedding'] = chunk_dict['embedding'].tolist()  # Convert numpy array to list
            self.metadata_store.append(chunk_dict)
        
        return chunks
    
    def search(self, query: str, k: int = 5, threshold: float = 0.5) -> List[SearchResult]:
        """
        Search for relevant document chunks.
        
        Args:
            query: Search query
            k: Number of results to return
            threshold: Minimum similarity threshold (0-1)
            
        Returns:
            List of SearchResult objects
        """
        # Generate query embedding
        query_embedding = self.text_processor.model.encode(
            [query], convert_to_numpy=True
        )[0].astype(np.float32).reshape(1, -1)
        
        # Search the index
        distances, indices = self.index.search(query_embedding, k)
        
        # Convert to search results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.metadata_store):
                continue
                
            # Convert distance to similarity score (smaller distance = higher similarity)
            # Using a simple conversion: similarity = 1 / (1 + distance)
            score = 1.0 / (1.0 + distance)
            
            if score >= threshold:
                chunk_data = self.metadata_store[idx]
                results.append(SearchResult(
                    text=chunk_data['text'],
                    metadata=chunk_data['metadata'],
                    score=score
                ))
        
        # Sort by score in descending order
        results.sort(key=lambda x: x.score, reverse=True)
        return results
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save the document store to disk.
        
        Args:
            path: Directory to save the store. If None, uses the persist_dir.
        """
        if path is None:
            path = self.persist_dir
        else:
            path = Path(path)
            
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save metadata
        with open(path / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(self.metadata_store, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'DocumentStore':
        """
        Load a document store from disk.
        
        Args:
            path: Directory containing the saved store
            
        Returns:
            Loaded DocumentStore instance
        """
        path = Path(path)
        store = cls()
        
        # Load FAISS index
        store.index = faiss.read_index(str(path / "index.faiss"))
        
        # Load metadata
        with open(path / "metadata.json", 'r', encoding='utf-8') as f:
            metadata_list = json.load(f)
            
        # Convert embeddings back to numpy arrays
        store.metadata_store = []
        for item in metadata_list:
            item['embedding'] = np.array(item['embedding'], dtype=np.float32)
            store.metadata_store.append(item)
        
        return store
    
    def clear(self) -> None:
        """Clear all documents from the store."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata_store = []
    
    def __len__(self) -> int:
        """Return the number of chunks in the store."""
        return len(self.metadata_store)
