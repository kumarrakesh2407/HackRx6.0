from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
import re
from dataclasses import dataclass

@dataclass
class TextChunk:
    """Represents a chunk of text with its metadata and embedding."""
    text: str
    metadata: Dict[str, Any]
    embedding: np.ndarray = None

class TextProcessor:
    """Handles text processing, chunking, and embedding generation."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the text processor with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.chunk_size = 1000  # Number of characters per chunk
        self.overlap = 200     # Overlap between chunks
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize the input text.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Normalize unicode characters
        text = text.encode('ascii', 'ignore').decode('ascii')
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s.,;:!?()\[\]\-]', ' ', text)
        return text.strip()
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[TextChunk]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            metadata: Metadata to associate with each chunk
            
        Returns:
            List of TextChunk objects
        """
        if metadata is None:
            metadata = {}
            
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            
            # If we're not at the end, try to end at a sentence boundary
            if end < text_length:
                # Look for sentence-ending punctuation
                sentence_end = max(text.rfind('.', start, end),
                                 text.rfind('!', start, end),
                                 text.rfind('?', start, end))
                if sentence_end > start + (self.chunk_size // 2):
                    end = sentence_end + 1
            
            chunk_text = text[start:end].strip()
            if chunk_text:  # Only add non-empty chunks
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_start': start,
                    'chunk_end': end,
                    'chunk_id': len(chunks)
                })
                chunks.append(TextChunk(text=chunk_text, metadata=chunk_metadata))
            
            # Move the start position, accounting for overlap
            start = end - self.overlap if end > self.overlap else end
            
            # Prevent infinite loop with very small chunks
            if start == end and end < text_length:
                start = end
        
        return chunks
    
    def embed_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """
        Generate embeddings for text chunks.
        
        Args:
            chunks: List of TextChunk objects
            
        Returns:
            List of TextChunk objects with embeddings
        """
        if not chunks:
            return []
            
        texts = [chunk.text for chunk in chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            
        return chunks
    
    def process_document(self, text: str, metadata: Dict[str, Any] = None) -> List[TextChunk]:
        """
        Process a document: clean, chunk, and embed the text.
        
        Args:
            text: Input document text
            metadata: Document metadata
            
        Returns:
            List of processed TextChunk objects with embeddings
        """
        if metadata is None:
            metadata = {}
            
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Chunk the text
        chunks = self.chunk_text(cleaned_text, metadata)
        
        # Generate embeddings for chunks
        chunks = self.embed_chunks(chunks)
        
        return chunks
    
    def calculate_similarity(self, query_embedding: np.ndarray, chunk_embedding: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            query_embedding: Embedding of the query
            chunk_embedding: Embedding of a text chunk
            
        Returns:
            Similarity score between 0 and 1
        """
        return np.dot(query_embedding, chunk_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding) + 1e-10
        )
