from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import os
import tempfile
import uvicorn
from typing import List, Dict, Any

from document_processor.processor import DocumentProcessor
from document_processor.query_processor import QueryInfo

app = FastAPI(
    title="LLM Document Processing System",
    description="API for processing and querying documents using LLMs",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the document processor
DOCUMENT_STORE_DIR = os.getenv("DOCUMENT_STORE_DIR", "data/document_store")
document_processor = DocumentProcessor(persist_dir=DOCUMENT_STORE_DIR)

# Create data directory if it doesn't exist
os.makedirs(DOCUMENT_STORE_DIR, exist_ok=True)

# Helper function to get allowed file extensions
def get_allowed_extensions() -> set:
    return {'.pdf', '.docx', '.doc', '.txt', '.eml'}

# Helper function to save uploaded file
def save_upload_file(upload_file: UploadFile) -> str:
    try:
        # Get file extension
        file_extension = Path(upload_file.filename).suffix.lower()
        
        # Create a temporary file with the correct extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            # Save the uploaded file to the temporary file
            content = upload_file.file.read()
            temp_file.write(content)
            return temp_file.name
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint that provides API information."""
    return {
        "name": "LLM Document Processing System",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/documents", "method": "POST", "description": "Upload a document"},
            {"path": "/query", "method": "POST", "description": "Query the document store"},
            {"path": "/health", "method": "GET", "description": "Health check"}
        ]
    }

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document to be processed and added to the document store.
    
    Supported formats: PDF, DOCX, DOC, TXT, EML
    """
    # Check file extension
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in get_allowed_extensions():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not supported. Allowed types: {', '.join(get_allowed_extensions())}"
        )
    
    try:
        # Save the uploaded file to a temporary location
        temp_file_path = save_upload_file(file)
        
        try:
            # Process the document
            result = document_processor.add_document(temp_file_path)
            
            if result['status'] == 'error':
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=result['error']
                )
            
            return {
                "status": "success",
                "document_id": result['document_id'],
                "chunks_processed": result['chunks_processed'],
                "metadata": result['metadata']
            }
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass
                
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )

@app.post("/query")
async def process_query(query: Dict[str, str]):
    """
    Process a natural language query against the document store.
    
    Example request body:
    {
        "query": "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
    }
    """
    if 'query' not in query or not query['query'].strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query parameter is required"
        )
    
    try:
        # Process the query
        result = document_processor.process_query(query['query'])
        
        if result['status'] == 'error':
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result['error']
            )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
