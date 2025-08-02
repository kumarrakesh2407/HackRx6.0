from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pathlib import Path
import PyPDF2
from docx import Document
import email
from email import policy
from email.parser import BytesParser
import io

class DocumentLoader(ABC):
    """Abstract base class for document loaders."""
    
    @abstractmethod
    def load(self, file_path: str) -> str:
        """Load and return the text content of a document."""
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return document metadata."""
        pass

class PDFLoader(DocumentLoader):
    """Load text content from PDF files."""
    
    def __init__(self):
        self.metadata = {}
    
    def load(self, file_path: str) -> str:
        """Extract text from a PDF file."""
        text = []
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                self.metadata = {
                    'title': pdf_reader.metadata.get('/Title', ''),
                    'author': pdf_reader.metadata.get('/Author', ''),
                    'pages': len(pdf_reader.pages)
                }
                for page in pdf_reader.pages:
                    text.append(page.extract_text())
            return '\n'.join(text)
        except Exception as e:
            raise Exception(f"Error loading PDF file: {str(e)}")
    
    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata

class WordLoader(DocumentLoader):
    """Load text content from Word documents."""
    
    def __init__(self):
        self.metadata = {}
    
    def load(self, file_path: str) -> str:
        """Extract text from a Word document."""
        try:
            doc = Document(file_path)
            self.metadata = {
                'title': doc.core_properties.title,
                'author': doc.core_properties.author,
                'created': doc.core_properties.created,
                'modified': doc.core_properties.modified
            }
            return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            raise Exception(f"Error loading Word document: {str(e)}")
    
    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata

class EmailLoader(DocumentLoader):
    """Load text content from email files."""
    
    def __init__(self):
        self.metadata = {}
    
    def load(self, file_path: str) -> str:
        """Extract text from an email file."""
        try:
            with open(file_path, 'rb') as file:
                msg = BytesParser(policy=policy.default).parse(file)
                
                self.metadata = {
                    'subject': msg['subject'],
                    'from': msg['from'],
                    'to': msg['to'],
                    'date': msg['date']
                }
                
                # Get email body
                if msg.is_multipart():
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        if content_type == 'text/plain':
                            return part.get_payload(decode=True).decode('utf-8', errors='ignore')
                else:
                    return msg.get_payload(decode=True).decode('utf-8', errors='ignore')
                
                return ""
        except Exception as e:
            raise Exception(f"Error loading email: {str(e)}")
    
    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata

def get_document_loader(file_path: str) -> DocumentLoader:
    """Factory function to get the appropriate document loader based on file extension."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ext = path.suffix.lower()
    if ext == '.pdf':
        return PDFLoader()
    elif ext in ['.docx', '.doc']:
        return WordLoader()
    elif ext in ['.eml', '.msg']:
        return EmailLoader()
    else:
        raise ValueError(f"Unsupported file format: {ext}")
