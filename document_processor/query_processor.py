from typing import Dict, Any, List, Optional
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class QueryInfo:
    """Structured information extracted from a natural language query."""
    age: Optional[int] = None
    gender: Optional[str] = None
    procedure: Optional[str] = None
    location: Optional[str] = None
    policy_duration: Optional[int] = None  # in months
    policy_type: Optional[str] = None
    raw_query: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the query info to a dictionary."""
        return {
            'age': self.age,
            'gender': self.gender,
            'procedure': self.procedure,
            'location': self.location,
            'policy_duration_months': self.policy_duration,
            'policy_type': self.policy_type,
            'raw_query': self.raw_query
        }

class QueryProcessor:
    """Processes natural language queries into structured data."""
    
    def __init__(self):
        # Common patterns for extracting information
        self.age_pattern = r'(\d+)\s*(?:years?\s*old|yo|y\.?o\.?|yrs?|years?)'
        self.gender_pattern = r'\b(male|female|m|f|man|woman|boy|girl|gentleman|lady)\b'
        self.policy_duration_pattern = r'(\d+)[-\s]*(month|year|day|week|mo|yr|wk|d|w|m)s?\s*(?:old)?\s*policy'
        self.location_pattern = r'\b(in|at|near|around|from)\s+([A-Za-z\s]+)(?:\,|$)'
        self.procedure_pattern = r'(surgery|operation|procedure|treatment|therapy|consultation|examination|test|scan|x[-\s]?ray|mri|ct|cat[\s-]?scan|ultrasound)'
    
    def extract_age(self, text: str) -> Optional[int]:
        """Extract age from text."""
        match = re.search(self.age_pattern, text, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                pass
        return None
    
    def extract_gender(self, text: str) -> Optional[str]:
        """Extract gender from text."""
        match = re.search(self.gender_pattern, text, re.IGNORECASE)
        if match:
            gender = match.group(1).lower()
            # Standardize gender representation
            if gender in ['m', 'male', 'man', 'boy', 'gentleman']:
                return 'male'
            elif gender in ['f', 'female', 'woman', 'girl', 'lady']:
                return 'female'
        return None
    
    def extract_policy_duration(self, text: str) -> Optional[int]:
        """Extract policy duration in months from text."""
        match = re.search(self.policy_duration_pattern, text, re.IGNORECASE)
        if match:
            try:
                value = int(match.group(1))
                unit = match.group(2).lower() if match.group(2) else 'm'
                
                # Convert to months
                if unit.startswith('y'):  # years
                    return value * 12
                elif unit.startswith('mo') or unit == 'm':  # months
                    return value
                elif unit.startswith('w'):  # weeks
                    return value // 4  # Approximate
                elif unit == 'd':  # days
                    return value // 30  # Approximate
            except (ValueError, IndexError):
                pass
        return None
    
    def extract_location(self, text: str) -> Optional[str]:
        """Extract location from text."""
        match = re.search(self.location_pattern, text, re.IGNORECASE)
        if match and len(match.groups()) >= 2:
            # Clean up the location string
            location = match.group(2).strip()
            # Remove any trailing punctuation
            location = re.sub(r'[^\w\s]', '', location)
            return location.title() if location else None
        return None
    
    def extract_procedure(self, text: str) -> Optional[str]:
        """Extract medical procedure from text."""
        # First try to find a procedure type
        procedure_match = re.search(self.procedure_pattern, text, re.IGNORECASE)
        if not procedure_match:
            return None
            
        # Get the position of the procedure match
        procedure_pos = procedure_match.start()
        
        # Look for words before the procedure that might describe it
        # (e.g., "knee surgery")
        words = text[:procedure_pos].split()
        if len(words) > 0:
            # Take the last 1-2 words before the procedure
            descriptor = ' '.join(words[-2:]) if len(words) >= 2 else words[-1]
            return f"{descriptor} {procedure_match.group(1)}".lower()
        
        return procedure_match.group(1).lower()
    
    def process_query(self, query: str) -> QueryInfo:
        """
        Process a natural language query and extract structured information.
        
        Args:
            query: The natural language query string
            
        Returns:
            QueryInfo object with extracted information
        """
        query = query.strip()
        if not query:
            return QueryInfo()
        
        # Initialize query info with raw query
        query_info = QueryInfo(raw_query=query)
        
        # Extract information using the helper methods
        query_info.age = self.extract_age(query)
        query_info.gender = self.extract_gender(query)
        query_info.procedure = self.extract_procedure(query)
        query_info.location = self.extract_location(query)
        query_info.policy_duration = self.extract_policy_duration(query)
        
        # Additional processing for policy type if needed
        if 'health' in query.lower() or 'medical' in query.lower():
            query_info.policy_type = 'health'
        elif 'auto' in query.lower() or 'car' in query.lower():
            query_info.policy_type = 'auto'
        elif 'home' in query.lower() or 'house' in query.lower():
            query_info.policy_type = 'home'
        
        return query_info
