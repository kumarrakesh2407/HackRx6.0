# LLM Document Processing System

A system that uses Large Language Models (LLMs) to process natural language queries and retrieve relevant information from large unstructured documents such as policy documents, contracts, and emails.

## Features

- **Document Ingestion**: Supports multiple document formats including PDF, Word, and email (.eml)
- **Semantic Search**: Uses sentence transformers for semantic understanding of queries
- **Natural Language Processing**: Extracts structured information from natural language queries
- **RESTful API**: Easy integration with other systems via FastAPI
- **Persistence**: Saves document embeddings for fast retrieval

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd HackRx6.0
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Starting the API Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

- `POST /documents/upload`: Upload a document to the system
- `POST /query`: Submit a natural language query
- `GET /health`: Check if the API is running

### Example Usage

1. **Upload a document**:
   ```bash
   curl -X 'POST' \
     'http://localhost:8000/documents/upload' \
     -H 'accept: application/json' \
     -H 'Content-Type: multipart/form-data' \
     -F 'file=@path/to/your/document.pdf;type=application/pdf'
   ```

2. **Query the system**:
   ```bash
   curl -X 'POST' \
     'http://localhost:8000/query' \
     -H 'accept: application/json' \
     -H 'Content-Type: application/json' \
     -d '{"query": "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"}'
   ```

## Project Structure

- `document_processor/`: Core processing modules
  - `document_loader.py`: Handles loading different document formats
  - `document_store.py`: Manages document storage and vector search
  - `query_processor.py`: Processes natural language queries
  - `text_processor.py`: Handles text processing and embeddings
  - `processor.py`: Main processing class that ties everything together
- `main.py`: FastAPI application and endpoints
- `requirements.txt`: Python dependencies

## Configuration

You can configure the system using environment variables:

- `DOCUMENT_STORE_DIR`: Directory to store document embeddings (default: `data/document_store`)
- `MODEL_NAME`: Name of the sentence transformer model to use (default: `all-MiniLM-L6-v2`)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
