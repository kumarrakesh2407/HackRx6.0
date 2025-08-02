from typing import List, Dict, Any, Optional
import json
from pathlib import Path
from dataclasses import asdict
from .document_loader import get_document_loader
from .document_store import DocumentStore
from .query_processor import QueryProcessor, QueryInfo
from .text_processor import TextProcessor

class DocumentProcessor:
    """
    Main class for processing documents and handling natural language queries.
    Handles document ingestion, query processing, and response generation.
    """
    
    def __init__(self, persist_dir: str = "data/document_store"):
        """
        Initialize the document processor.
        
        Args:
            persist_dir: Directory to persist the document store
        """
        self.document_store = DocumentStore(persist_dir)
        self.query_processor = QueryProcessor()
        self.text_processor = TextProcessor()
    
    def add_document(self, file_path: str) -> Dict[str, Any]:
        """
        Add a document to the processor.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with processing results and metadata
        """
        try:
            # Load the document
            loader = get_document_loader(file_path)
            text = loader.load(file_path)
            metadata = loader.get_metadata()
            
            # Add to document store
            chunks = self.document_store.add_document(text, metadata)
            
            # Save the updated document store
            self.document_store.save()
            
            return {
                'status': 'success',
                'document_id': str(hash(file_path)),
                'chunks_processed': len(chunks),
                'metadata': metadata
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def process_query(self, query: str, k: int = 3, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Process a natural language query against the document store.
        
        Args:
            query: The natural language query
            k: Number of relevant chunks to retrieve
            threshold: Minimum similarity threshold (0-1)
            
        Returns:
            Dictionary with query results and structured information
        """
        try:
            # Parse the query
            query_info = self.query_processor.process_query(query)
            
            # Search for relevant chunks
            search_results = self.document_store.search(query, k=k, threshold=threshold)
            
            # Prepare response
            response = {
                'status': 'success',
                'query_info': asdict(query_info),
                'relevant_chunks': [],
                'decision': None,
                'confidence': 0.0,
                'justification': []
            }
            
            # Process search results
            if search_results:
                # Calculate overall confidence as average of top results
                total_score = sum(r.score for r in search_results)
                response['confidence'] = total_score / len(search_results)
                
                # Add relevant chunks to response
                for result in search_results:
                    response['relevant_chunks'].append({
                        'text': result.text,
                        'metadata': result.metadata,
                        'score': float(result.score)  # Convert numpy float to Python float
                    })
                
                # Simple decision logic based on query info and results
                # This can be enhanced based on specific domain requirements
                response['decision'] = self._make_decision(query_info, search_results)
                response['justification'] = self._generate_justification(
                    query_info, 
                    search_results
                )
            
            return response
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _make_decision(self, query_info: QueryInfo, results: List[Any]) -> Dict[str, Any]:
        """
        Make a decision based on the query info and search results.
        
        This is a simple implementation that can be customized for specific domains.
        """
        # Default decision
        decision = {
            'approved': False,
            'reason': 'No matching policy found',
            'details': {}
        }
        
        # If we have search results, consider the query approved
        if results:
            decision['approved'] = True
            decision['reason'] = 'Matching policy found'
            
            # Add details based on query info
            if query_info.procedure:
                decision['details']['procedure'] = {
                    'name': query_info.procedure,
                    'covered': True  # Simplified for this example
                }
            
            if query_info.policy_duration:
                decision['details']['policy_duration'] = {
                    'months': query_info.policy_duration,
                    'meets_requirements': True  # Simplified for this example
                }
        
        return decision
    
    def _generate_justification(self, query_info: QueryInfo, results: List[Any]) -> List[str]:
        """
        Generate a human-readable justification for the decision.
        """
        justifications = []
        
        # Add information about the query
        if query_info.procedure:
            justifications.append(f"Query is about: {query_info.procedure}")
        
        if query_info.age and query_info.gender:
            justifications.append(f"Patient: {query_info.age} year old {query_info.gender}")
        
        if query_info.location:
            justifications.append(f"Location: {query_info.location}")
        
        if query_info.policy_duration:
            justifications.append(f"Policy duration: {query_info.policy_duration} months")
        
        # Add information about the matching documents
        if results:
            source_docs = set()
            for result in results:
                if 'title' in result.metadata and result.metadata['title']:
                    source_docs.add(result.metadata['title'])
            
            if source_docs:
                justifications.append(
                    f"Found relevant information in: {', '.join(source_docs)}"
                )
        
        return justifications
    
    def save(self, path: Optional[str] = None) -> None:
        """Save the document store to disk."""
        self.document_store.save(path)
    
    @classmethod
    def load(cls, path: str) -> 'DocumentProcessor':
        """
        Load a document processor from disk.
        
        Args:
            path: Directory containing the saved processor
            
        Returns:
            Loaded DocumentProcessor instance
        """
        processor = cls()
        processor.document_store = DocumentStore.load(path)
        return processor
    
    def clear(self) -> None:
        """Clear all documents from the processor."""
        self.document_store.clear()
