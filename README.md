# OmniDoc-AI (ColPali Vision RAG)

## Description
An intelligent, extensible document understanding system designed to transform static documents—PDFs, books, manuals, and more—into dynamic, searchable, and conversational knowledge bases.


OmnoDoc-AI is a vision-driven Retrieval Augmented Generation (RAG) system built using ColPali, Qdrant, and FastAPI. It enables you to index PDF documents by treating each page as an image, perform visual similarity searches based on text queries, and generate answers using a large language model (LLM) like Llama3.2-vision (via Ollama). 

The system includes API endpoints for seamless integration into other applications and features duplicate file detection based on content hashing.

## Features

* Vision-Driven RAG: Leverages the ColPali model to understand visual layouts and content within documents.
* PDF Document Indexing: Converts PDF pages into images and indexes their visual embeddings into Qdrant.
* Duplicate File Detection: Prevents re-indexing the same PDF file by computing and checking its SHA256 hash.
* Efficient Vector Search: Uses Qdrant for fast and scalable similarity search of image embeddings, including multi-vector comparison and binary quantization.
* API Interface: Provides RESTful API endpoints for:
    * Uploading and indexing PDF files.
    * Deleting indexed files by their content hash.
    * Querying the RAG system with text prompts to retrieve relevant visual documents and generate answers.
* LLM Integration: Connects with local LLMs (e.g., Llama3.2-vision via Ollama) to synthesize answers based on retrieved visual context.
* Dockerized Environment: Provides Dockerfile and docker-compose.yml for easy setup and reproducible deployment.

## Project Structure

```
OmniDoc-AI/
├── app/
│   ├── main.py             # Main FastAPI application and RAG logic
│   └── requirements.txt    # Python dependencies
├── docker-compose.yml      # Defines Qdrant and FastAPI services
├── Dockerfile              # Builds the Docker image for the FastAPI app
└── .env                    # Environment variables (e.g., Qdrant host/port)

```


## Prerequisites
Before you begin, ensure you have the following installed:
1. **Docker Desktop**  (for Windows/macOS) or **Docker Engine** (for Linux): Used to containerize and run the application and Qdrant. You can download it from docker.com.
2. **Ollama (Optional, for LLM response generation):** To enable the LLM response generation, you need to have Ollama installed and running on your host machine (or in a separate, accessible Docker container).
    * Install Ollama from ollama.com.
    * Pull the llama3.2-vision model: ollama pull llama3.2-vision
    * Ensure Ollama server is running (usually ollama serve in the background).
    
## Setup and Running the Application
1. **Clone the Project:**
2. **Configure Environment Variables:**
    Open the .env file and ensure the Qdrant configuration matches your setup if you plan to change the default ports.
    ```
    # .env
    QDRANT_HOST=qdrant # For inter-container communication in Docker Compose
    QDRANT_PORT=6333
    COLLECTION_NAME=colpali_demo
    ```

3. **Build and Run with Docker Compose:**
    Open your terminal or command prompt, navigate to the root colpali-rag directory, and execute the following command:
    ```
    docker-compose up --build
    ```
    * --build: This flag tells Docker Compose to build the colpali-rag-app Docker image (and download Qdrant's image if not present) before starting the services.
    * The first run might take a significant amount of time as it downloads Python dependencies, colpali_engine, and the large ColPali model weights (which will be cached in ./hf_cache for future runs).
4. **Access the API:**
    Once both the qdrant_db and colpali_rag_app containers are running and healthy (you'll see logs indicating FastAPI startup), your API will be accessible at:
    
    ```
    http://localhost:8000
    ```

    You can access the interactive API documentation (Swagger UI) at:
    ```
    http://localhost:8000/docs
    ```

## API Endpoints
You can use the /docs page to interact with the API directly or use curl / Postman / Python requests.

1. **Index a PDF File**
Indexes a PDF document's pages as images into the Qdrant vector database. Includes duplicate file detection.
    * Endpoint: POST /index_pdf/
    * Content-Type: multipart/form-data
    * Form Field: file (type: file)
    * Example (using curl):
        ```curl
        curl -X POST "http://localhost:8000/index_pdf/" \
            -H "accept: application/json" \
            -H "Content-Type: multipart/form-data" \
            -F "file=@/path/to/your/document.pdf;type=application/pdf"
        ```
        Replace /path/to/your/document.pdf with the actual path to your PDF file.
    Response Body (Example):
    ```json
    {
      "file_name": "example_document.pdf",
      "file_hash": "a4d3f2c1b0e9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3",
      "indexed_pages": 10,
      "message": "PDF indexed successfully."
    }
    ```

2. **Delete a PDF File by Hash**
   Removes all indexed pages associated with a specific PDF file hash from Qdrant. This is useful for managing document versions.
   * Endpoint: POST /delete_pdf/
   * Content-Type: application/json
   * Request Body:
   ```json
    {
        "file_hash": "the_sha256_hash_of_the_file_to_delete"
    }
    ```

    You would get this file_hash from the response of the /index_pdf/ endpoint or by calculating it yourself.
    
    * Example (using curl):
        ```
        curl -X POST "http://localhost:8000/delete_pdf/" \
            -H "accept: application/json" \
            -H "Content-Type: application/json" \
            -d '{"file_hash": "a4d3f2c1b0e9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3"}'
        ```

    Replace the hash with the actual hash of the file you wish to delete.

3. **Query the RAG System**
    Submits a text query to the RAG system. It retrieves relevant visual documents from Qdrant and generates an answer using Ollama.
    * Endpoint: POST /query/
    * Content-Type: application/json
    * Request Body:
        ```json 
        {
        "query_text": "What are the main features of the ColPali model?"
        }
       ```
       
    * Response Body (Example):
        ```json
        {
            "query": "What are the main features of the ColPali model?",
            "answer": "The ColPali model focuses on vision-driven RAG systems by processing documents as images, breaking them into patches, and generating patch-level embeddings for efficient retrieval and understanding of multimodal content.",
            "retrieved_documents": [
                {
                "id": 123456789,
                "score": 0.85,
                "image_b64": "base64_encoded_image_string...",
                "original_source_type": "PDF_Page",
                "page_number": 2,
                "file_name": "example_document.pdf",
                "file_hash": "a4d3f2c1b0e9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3"
                }
                // ... more retrieved documents
            ]
        }
        ```
    * Example (using curl):
    ```
    curl -X POST "http://localhost:8000/query/" \
        -H "accept: application/json" \
        -H "Content-Type: application/json" \
        -d '{"query_text": "Who developed the ColPali model?"}'
    ```

Important Notes
* PDF Page Rendering: The system now uses PyMuPDF for high-fidelity PDF page rendering, which significantly improves the quality of visual input for ColPali compared to the previous placeholder approach.
* Ollama Integration: The generate_response_with_ollama function relies on an Ollama server running and accessible from the Docker container. Ensure Ollama is set up and the llama3.2-vision model is pulled on your host machine or in a separately configured Ollama container.
* hf_cache Volume: The ./hf_cache volume in docker-compose.yml is crucial. It caches the large ColPali model weights downloaded from Hugging Face, preventing re-downloading them every time the Docker container is rebuilt.
* Qdrant Data Persistence: The ./qdrant_storage volume ensures that your indexed Qdrant data persists even if you stop and restart your Docker containers.