# app/main.py

import torch
import time
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional
import hashlib # For file hashing
from contextlib import asynccontextmanager # For lifespan events

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pydantic import BaseModel
from starlette.responses import JSONResponse

# PDF processing import
# from pdfminer.high_level import extract_pages # No longer needed for image extraction
import fitz # PyMuPDF

# Import necessary Qdrant and ColPali libraries
from qdrant_client import QdrantClient
from qdrant_client.http import models
from colpali_engine.models import ColPali, ColPaliProcessor

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "mahabodha_demo")
COLPALI_MODEL_NAME = "vidore/colpali-v1.2"
BATCH_SIZE = 4 # Batch size for processing images
TOP_K_RETRIEVAL = 4 # Number of top results to retrieve
MODEL_CACHE_DIR=os.getenv("HF_HOME", "./hf_cache"),

# Determine device for ColPali model (CPU or MPS/CUDA if available)
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")

# --- Helper Functions ---

def image_to_base64(image: Image.Image) -> str:
    """
    Converts a PIL Image object to a Base64-encoded string.
    This is used to store the original image in Qdrant's payload for retrieval.
    """
    buffered = BytesIO()
    # Use JPEG for smaller size, adjust format if image transparency is needed (e.g., PNG)
    # Ensure all images are RGB to prevent saving errors with JPEG
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def base64_to_image(base64_string: str) -> Image.Image:
    """
    Converts a Base64-encoded string back to a PIL Image object.
    """
    img_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(img_data))

def calculate_file_hash(file_content: BytesIO, hash_algo='sha256') -> str:
    """
    Calculates the SHA256 hash of a file's content to serve as a unique identifier.
    Resets the buffer's position after hashing.
    """
    hasher = hashlib.sha256()
    # Read the file in chunks to handle large files
    file_content.seek(0) # Ensure we read from the beginning
    for chunk in iter(lambda: file_content.read(4096), b''):
        hasher.update(chunk)
    file_content.seek(0) # Reset pointer for subsequent reads
    return hasher.hexdigest()

def extract_images_from_pdf(pdf_file_path: BytesIO) -> List[Image.Image]:
    """
    Extracts each page of a PDF as a high-fidelity PIL Image using PyMuPDF.
    """
    print("Extracting pages from PDF using PyMuPDF...")
    images = []
    try:
        # Open the PDF document from the BytesIO stream
        # Reset stream position to ensure fitz can read from the beginning
        pdf_file_path.seek(0) 
        doc = fitz.open(stream=pdf_file_path.read(), filetype="pdf")
        
        for i, page in enumerate(doc):
            # Render page to a pixmap (image)
            pix = page.get_pixmap()
            
            # Convert pixmap to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
            
        doc.close() # Close the document
        
        if not images:
            raise ValueError("No pages extracted from PDF or pages were empty.")
            
    except Exception as e:
        print(f"Error extracting PDF pages with PyMuPDF: {e}. Returning a single dummy image.")
        # Fallback to a single dummy image if PDF parsing fails or yields no pages
        dummy_image = Image.new('RGB', (800, 1000), color = 'white')
        from PIL import ImageDraw, ImageFont
        d = ImageDraw.Draw(dummy_image)
        try:
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, 40)
            else:
                font = ImageFont.load_default()
        except IOError:
            font = ImageFont.load_default()
        d.text((50,50), "Error: Could not extract PDF content. Using dummy page.", fill=(0,0,0), font=font)
        images.append(dummy_image)

    return images

def extract_images_from_file(file: UploadFile, file_content: BytesIO) -> List[Image.Image]:
    """
    Extract images from supported file types: PDF, images (JPG, PNG, TIFF), text files (TXT, MD), and placeholder for DOCX/PPTX.
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext == ".pdf":
        return extract_images_from_pdf(file_content)
    elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]:
        try:
            img = Image.open(file_content)
            return [img]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process image: {e}")
    elif ext in [".tiff", ".tif"]:
        try:
            img = Image.open(file_content)
            images = []
            for i in range(getattr(img, "n_frames", 1)):
                img.seek(i)
                images.append(img.copy())
            return images
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process TIFF: {e}")
    elif ext in [".txt", ".md"]:
        try:
            file_content.seek(0)
            text = file_content.read().decode("utf-8")
            # Render text as image
            from PIL import ImageDraw, ImageFont
            width, height = 800, 1000
            img = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(img)
            try:
                font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                font = ImageFont.truetype(font_path, 20) if os.path.exists(font_path) else ImageFont.load_default()
            except Exception:
                font = ImageFont.load_default()
            # Wrap text to fit image width
            import textwrap
            margin, offset = 40, 40
            for line in text.splitlines():
                wrapped = textwrap.wrap(line, width=90)
                for subline in wrapped:
                    draw.text((margin, offset), subline, font=font, fill=(0,0,0))
                    offset += 28
                    if offset > height - 40:
                        break
                if offset > height - 40:
                    break
            return [img]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process text file: {e}")
    elif ext in [".docx", ".pptx"]:
        # Placeholder: implement DOCX/PPTX to image extraction if needed
        raise HTTPException(status_code=415, detail="DOCX/PPTX support not implemented yet.")
    else:
        raise HTTPException(status_code=415, detail="Unsupported file type.")

# --- Main RAG Class ---

class ColPaliRAG:
    _instance = None # Singleton instance

    def __new__(cls):
        """Ensures only one instance of ColPaliRAG is created (Singleton pattern)."""
        if cls._instance is None:
            cls._instance = super(ColPaliRAG, cls).__new__(cls)
            cls._instance._initialized = False # Use an internal flag for initialization
        return cls._instance

    def __init__(self):
        """
        Initializes the ColPali model, processor, and Qdrant client.
        Sets up the Qdrant collection if it doesn't exist.
        """
        if self._initialized:
            return
        
        self.colpali_model = self._load_colpali_model()
        self.colpali_processor = self._load_colpali_processor()
        self.qdrant_client = self._setup_qdrant_client()
        self._create_qdrant_collection()
        self._initialized = True # Mark as initialized

    def _load_colpali_model(self):
        """Loads the ColPali model from Hugging Face Hub."""
        print(f"Loading ColPali model: {COLPALI_MODEL_NAME} on {DEVICE}...")
        model = ColPali.from_pretrained(
            COLPALI_MODEL_NAME,
            cache_dir= MODEL_CACHE_DIR,  # Cache directory for model weights
            torch_dtype=torch.bfloat16,  # Use bfloat16 for performance if supported
            local_files_only=True,
            device_map=DEVICE,
            trust_remote_code=True
        )
        model.eval()  # Set model to evaluation mode
        print("ColPali model loaded.")
        return model

    def _load_colpali_processor(self):
        """Loads the ColPali processor."""
        print(f"Loading ColPali processor: {COLPALI_MODEL_NAME}...")
        processor = ColPaliProcessor.from_pretrained(COLPALI_MODEL_NAME,
                                                     cache_dir= MODEL_CACHE_DIR,
                                                     local_files_only=True)
        print("ColPali processor loaded.")
        return processor

    def _setup_qdrant_client(self):
        """Sets up the connection to the Qdrant vector database."""
        print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        # Verify connection
        try:
            client.get_collections()
            print("Successfully connected to Qdrant.")
        except Exception as e:
            print(f"Failed to connect to Qdrant: {e}")
            raise ConnectionError(f"Could not connect to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        return client

    def _create_qdrant_collection(self):
        """
        Creates a Qdrant collection for storing ColPali embeddings.
        Configured for multi-vector (patch-level) embeddings with MAX_SIM
        and binary quantization for performance.
        """
        print(f"Checking/creating Qdrant collection: {COLLECTION_NAME}...")
        # Check if collection already exists
        collections = self.qdrant_client.get_collections().collections
        if COLLECTION_NAME in [c.name for c in collections]:
            print(f"Collection '{COLLECTION_NAME}' already exists. Skipping creation.")
            return

        self.qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            on_disk_payload=True, # Store metadata on disk for large datasets
            vectors_config=models.VectorParams(
                size=128, # Dimensionality of ColPali patch embeddings
                distance=models.Distance.COSINE, # Cosine similarity for search
                on_disk=True, # Store vector embeddings on disk
                # Multi-vector config for ColPali's late interaction
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM # Use MAX_SIM for ColBERT-like scoring
                ),
            ),
            # Enable binary quantization for faster retrieval as discussed in the PDF
            quantization_config=models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(always_ram=True),
            ),
        )
        print(f"Collection '{COLLECTION_NAME}' created successfully with Binary Quantization.")

    def _upsert_images_to_qdrant(self, images: List[Image.Image], file_hash: str, file_name: str):
        """
        Helper to generate embeddings for a list of images and upsert them to Qdrant.
        Uses file_hash to create stable Point IDs.
        """
        points = []
        for j, image in enumerate(images):
            with torch.no_grad():
                # Process individual image
                batch_images_processed = self.colpali_processor.process_images([image]).to(self.colpali_model.device)
                image_embeddings = self.colpali_model(batch_images_processed)
                
                # Each 'image_embeddings' here is a multi-vector representing patches for one image
                multivector = image_embeddings[0].cpu().float().numpy().tolist()

            # Create a consistent ID for each page based on file hash and page number
            # This allows updating specific pages of a file if re-indexed.
            page_id_str = f"{file_hash}_{j+1}"
            point_id = int(hashlib.sha256(page_id_str.encode()).hexdigest(), 16) % (2**63 - 1) # Generate a long int ID

            points.append(
                models.PointStruct(
                    id=point_id, # Use consistent ID for re-uploading same file/page
                    vector=multivector, # This is a list of vectors (patches)
                    payload={
                        "source_image_b64": image_to_base64(image),
                        "original_source_type": "PDF_Page",
                        "page_number": j + 1,
                        "file_name": file_name,
                        "file_hash": file_hash # Store the file hash
                    },
                )
            )
        try:
            self.qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True,
            )
            print(f"Upserted {len(points)} image embeddings to Qdrant for file hash {file_hash}.")
            return True
        except Exception as e:
            print(f"Error during upsert: {e}")
            return False

    def index_pdf(self, file_content: BytesIO, file_name: str) -> Dict[str, str]:
        """
        Extracts images from a PDF file and indexes them into Qdrant.
        Checks for duplicate files using content hash.
       
        """
        print(f"Indexing PDF: {file_name}...")
        
        file_hash = calculate_file_hash(file_content)
        
        # Check if the file (identified by its hash) is already indexed
        try:
            # Search for any point with this file_hash in the payload
            # We don't need the vector itself, just existence check
            # For this check, we need a dummy query vector. We can get one by processing a simple string.
            dummy_query_vector = self.colpali_model(self.colpali_processor.process_queries(["dummy_query"]).to(self.colpali_model.device))[0].cpu().float().numpy().tolist()

            search_result = self.qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=dummy_query_vector, # Use a dummy query vector for filtering
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_hash",
                            match=models.MatchValue(value=file_hash),
                        )
                    ]
                ),
                limit=1, # We only need to know if one exists
                with_payload=False,
                with_vectors=False
            )
            
            if search_result:
                print(f"File '{file_name}' (hash: {file_hash}) is already indexed.")
                return {
                    "file_name": file_name,
                    "file_hash": file_hash,
                    "indexed_pages": 0,
                    "message": f"File '{file_name}' is already indexed."
                }

        except Exception as e:
            print(f"Error checking for duplicate file hash: {e}")
            # Continue with indexing if check fails, but log the error

        try:
            images = extract_images_from_pdf(file_content)
            if not images:
                raise ValueError("No images could be extracted from the PDF.")
            
            success = self._upsert_images_to_qdrant(images, file_hash, file_name)
            if success:
                return {
                    "file_name": file_name,
                    "file_hash": file_hash,
                    "indexed_pages": len(images),
                    "message": f"PDF indexed successfully."
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to upsert embeddings to Qdrant.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDF indexing failed: {e}")

    def delete_documents_by_hash(self, file_hash: str) -> Dict[str, str]:
        """
        Deletes all documents (pages) associated with a given file hash from Qdrant.
        """
        print(f"Attempting to delete documents with file hash: {file_hash}...")
        try:
            delete_result = self.qdrant_client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.PointSelector(
                    points=None, # Not selecting by specific points, but by filter
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="file_hash",
                                match=models.MatchValue(value=file_hash),
                            )
                        ]
                    )
                ),
                wait=True
            )
            if delete_result.status == models.UpdateStatus.COMPLETED:
                # Qdrant's delete operation doesn't return count directly, so we can't say how many.
                # A successful deletion means the operation completed.
                return {"message": f"Delete operation completed for file hash: {file_hash}. Documents may have been removed."}
            else:
                return {"message": f"Delete operation status: {delete_result.status.value}. Possible issue during deletion for file hash: {file_hash}."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete documents for hash {file_hash}: {e}")

    def query_documents(self, query_text: str) -> List[Dict]:
        """
        Processes a text query, generates its embedding, and searches
        the Qdrant collection for relevant documents (images).
        """
        print(f"Processing query: '{query_text}'...")
        with torch.no_grad():
            query_processed = self.colpali_processor.process_queries([query_text]).to(self.colpali_model.device)
            query_embedding = self.colpali_model(query_processed)
            token_query = query_embedding[0].cpu().float().numpy().tolist()

        start_time = time.time()
        print(f"Searching Qdrant collection: {COLLECTION_NAME}...")
        query_result = self.qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=token_query,
            limit=TOP_K_RETRIEVAL,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=False,
                    rescore=True,
                    oversampling=2.0
                )
            )
        )
        time_taken = time.time() - start_time
        print(f"Query completed in {time_taken:.3f} seconds.")

        retrieved_results = []
        for hit in query_result:
            image_b64 = hit.payload.get("source_image_b64")
            image = base64_to_image(image_b64) if image_b64 else None

            retrieved_results.append({
                "id": hit.id,
                "score": hit.score,
                "image_b64": image_b64, # Return base64 string for API
                "original_source_type": hit.payload.get("original_source_type"),
                "page_number": hit.payload.get("page_number"),
                "file_name": hit.payload.get("file_name"),
                "file_hash": hit.payload.get("file_hash")
            })
        return retrieved_results

    def generate_response_with_ollama(self, query_text: str, retrieved_image_b64s: List[str]):
        """
        Generates a response using a local Ollama Llama3.2-vision model based on the query
        and retrieved images (as base64 strings).
        """
        print("\n--- Generating Response with Ollama ---")
        try:
            import ollama
        except ImportError:
            print("Ollama library not found. Please install it (`pip install ollama`).")
            return "Ollama library not available. Cannot generate response."

        if not retrieved_image_b64s:
            return "No relevant images provided to answer the question."

        # Save images temporarily for Ollama's local file access requirement
        image_paths = []
        for i, b64_str in enumerate(retrieved_image_b64s):
            try:
                img = base64_to_image(b64_str)
                # Use a more robust temp file naming to avoid collisions
                temp_path = f"/tmp/temp_image_{os.getpid()}_{int(time.time())}_{i}.jpg"
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                img.save(temp_path)
                image_paths.append(temp_path)
            except Exception as e:
                print(f"Warning: Could not decode/save image from base64: {e}")
                continue # Skip this image if it's invalid

        if not image_paths:
            return "Could not process any images for Ollama. Cannot generate response."

        qa_prompt_tmpl_str = """The user has asked the following question:
        --------------------
        Query: {query}
        --------------------
        Some images are available to you for this question. You have
        to understand these images thoroughly and extract all relevant information that might
        help you answer the query better.
        Given the context information above I want you
        to think step by step to answer the query in a
        crisp manner, in case you don't know the
        answer say 'I don't know!'
        --------------------
        Answer: """

        prompt = qa_prompt_tmpl_str.format(query=query_text)
        messages = [
            {"role": "user", "content": prompt, "images": image_paths}
        ]

        print("Sending request to Ollama...")
        try:
            response = ollama.chat(model="llama3.2-vision", messages=messages)
            return response.get("message", {}).get("content", "Failed to get response from Ollama.")
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return f"Error connecting to Ollama or model not found. Ensure Ollama server is running and 'llama3.2-vision' model is pulled. Error: {e}"
        finally:
            # Clean up temporary image files
            for path in image_paths:
                if os.path.exists(path):
                    os.remove(path)

# Define the lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_system
    try:
        rag_system = ColPaliRAG()
        # Optionally index a default dataset on startup if not already indexed
        # rag_system.index_documents()
    except ConnectionError as e:
        print(f"Startup failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to initialize RAG system: {e}. Is Qdrant running?"
        )
    except Exception as e:
        print(f"Unexpected error during startup: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during system initialization: {e}"
        )
    yield
    # Cleanup logic can go here (e.g., closing Qdrant client connection if necessary)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="ColPali RAG API",
    description="API for a Vision-Driven RAG system using ColPali and Qdrant.",
    lifespan=lifespan # Register the lifespan event handler
)


# --- API Models ---
class QueryRequest(BaseModel):
    query_text: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_documents: List[Dict]

class DeleteRequest(BaseModel):
    file_hash: str # The hash of the file to be deleted

# --- API Endpoints ---

@app.get("/")
async def read_root():
    return {"message": "Welcome to the ColPali RAG API! Use /docs for API documentation."}

@app.post("/index_pdf/", summary="Index a PDF file for RAG")
async def index_pdf_file(file: UploadFile = File(...)):
    """
    Uploads a PDF file, extracts its pages as images, and indexes them into Qdrant.
    Checks for duplicates using file hash.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are accepted.")
    
    # Read file content into BytesIO for hashing and PDF parsing
    file_content_bytes = await file.read()
    file_content_buffer = BytesIO(file_content_bytes)

    return rag_system.index_pdf(file_content_buffer, file.filename)

@app.post("/delete_pdf/", summary="Delete a PDF file from RAG by its hash")
async def delete_pdf_file(request: DeleteRequest):
    """
    Deletes all indexed pages associated with a specific PDF file hash.
    """
    file_hash_to_delete = request.file_hash
    return rag_system.delete_documents_by_hash(file_hash_to_delete)

@app.post("/query/", response_model=QueryResponse, summary="Query the RAG system")
async def perform_query(request: QueryRequest):
    """
    Performs a RAG query using the provided text and retrieves relevant documents.
    Then, it generates an answer using an Ollama-hosted LLM (e.g., Llama3.2-vision).
    """
    query_text = request.query_text
    retrieved_docs = rag_system.query_documents(query_text)

    # Extract base64 images for Ollama
    retrieved_image_b64s = [doc["image_b64"] for doc in retrieved_docs if doc.get("image_b64")]

    final_answer = rag_system.generate_response_with_ollama(query_text, retrieved_image_b64s)

    # Remove the 'image' PIL object from the retrieved_docs before sending as JSON response
    # It's better to send 'image_b64' if the client needs the image data
    for doc in retrieved_docs:
        if 'image' in doc:
            del doc['image'] # Remove PIL Image object for JSON serialization

    return QueryResponse(
        query=query_text,
        answer=final_answer,
        retrieved_documents=retrieved_docs
    )

@app.post("/index_file/", summary="Index a file (PDF, image, TIFF) for RAG")
async def index_file(file: UploadFile = File(...)):
    """
    Uploads a file (PDF, image, TIFF), extracts images, and indexes them into Qdrant.
    Checks for duplicates using file hash.
    """
    ext = os.path.splitext(file.filename)[1].lower()
    supported_types = [".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif"]
    if ext not in supported_types:
        raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF, image, and TIFF files are accepted.")
    file_content_bytes = await file.read()
    file_content_buffer = BytesIO(file_content_bytes)
    try:
        images = extract_images_from_file(file, file_content_buffer)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract images: {e}")
    # Use file hash for deduplication
    file_hash = calculate_file_hash(file_content_buffer)
    success = rag_system._upsert_images_to_qdrant(images, file_hash, file.filename)
    if success:
        return {"file_name": file.filename, "file_hash": file_hash, "indexed_pages": len(images), "message": f"File indexed successfully."}
    else:
        raise HTTPException(status_code=500, detail="Failed to upsert embeddings to Qdrant.")

# --- Main script execution for local testing (optional) ---
if __name__ == "__main__":
    # This block will only run if main.py is executed directly, not via uvicorn
    # For API, you'd typically run `uvicorn main:app --host 0.0.0.0 --port 8000`
    print("This script is now an API. Run it using: uvicorn main:app --host 0.0.0.0 --port 8000")
    print("Access API docs at http://localhost:8000/docs")

    # Example of how you might test components without FastAPI:
    # rag_system_instance = ColPaliRAG()
    # rag_system_instance.index_pdf(BytesIO(open("path/to/your/document.pdf", "rb").read()), "your_document.pdf")
    # query_results = rag_system_instance.query_documents("What is the main topic?")
    # print(query_results)
