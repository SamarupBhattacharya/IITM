import os
import numpy as np
import httpx
import base64
import json
from io import BytesIO
# Removed: from PIL import Image # Pillow library no longer used for image validation in this version
# Removed: from dotenv import load_dotenv # python-dotenv no longer used for local env loading

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl
import time
import traceback # Import for full traceback in logging

# Load environment variables (Note: load_dotenv() call is removed here,
# as Vercel natively provides these and for local dev you'd set them in your shell
# or use a dedicated local setup without python-dotenv for this version.)
# It's assumed OPENAI_API_KEY and GOOGLE_API_KEY will be set in Vercel's dashboard.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Configuration ---
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

LLM_MODEL = "gpt-4o-mini"
GEMINI_VISION_MODEL = "gemini-2.0-flash"

AI_PROXY_OPENAI_BASE_URL = "https://aiproxy.sanand.workers.dev/openai/v1"
GOOGLE_GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# --- KNOWLEDGE BASE PATH (GLOBAL VARIABLE) ---
# This assumes course_with_id.npz is in the same directory as main.py (inside 'api/')
current_script_dir = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_NPZ_PATH = os.path.join(current_script_dir, 'course_with_id.npz')

# Retrieval Parameters
NUM_SIMILAR_CHUNKS = 5

app = FastAPI(
    title="Course Q&A API",
    description="API endpoint for student questions, leveraging RAG with GPT-4o-mini and Gemini Vision. Returns answer and the top relevant link.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

openai_proxy_client = httpx.AsyncClient(base_url=AI_PROXY_OPENAI_BASE_URL, timeout=30.0)
google_gemini_client = httpx.AsyncClient(base_url=GOOGLE_GEMINI_API_BASE_URL, timeout=30.0)

course_chunk_url_keys = []
course_embeddings = None

@app.on_event("startup")
async def load_knowledge_base():
    """
    Loads the (chunk_text, url) pairs and embeddings from the .npz file when the FastAPI app starts.
    Includes robust path checking and detailed logging to debug deployment issues.
    """
    global course_chunk_url_keys, course_embeddings
    print(f"--- STARTUP: Attempting to load knowledge base ---")
    print(f"DEBUG: Calculated KNOWLEDGE_BASE_NPZ_PATH: '{KNOWLEDGE_BASE_NPZ_PATH}'")

    try:
        # Check if the file exists at the calculated path
        if not os.path.exists(KNOWLEDGE_BASE_NPZ_PATH):
            # If not found, log extensive debugging information
            current_working_dir = os.getcwd()
            
            try:
                files_in_cwd = os.listdir(current_working_dir)
            except OSError as e:
                files_in_cwd = [f"Error listing CWD: {e}"]

            try:
                files_in_script_dir = os.listdir(current_script_dir)
            except OSError as e:
                files_in_script_dir = [f"Error listing script dir: {e}"]
            
            print(f"ERROR: Knowledge base file NOT FOUND at: '{KNOWLEDGE_BASE_NPZ_PATH}'")
            print(f"DEBUG: Current Working Directory (os.getcwd()): '{current_working_dir}'")
            print(f"DEBUG: Files in Current Working Directory: {files_in_cwd}")
            print(f"DEBUG: Script's Directory (current_script_dir): '{current_script_dir}'")
            print(f"DEBUG: Files in Script's Directory: {files_in_script_dir}")
            
            raise FileNotFoundError(f"Knowledge base file not found at: {KNOWLEDGE_BASE_NPZ_PATH}. Check logs for details.")

        # If file exists, attempt to load it with numpy
        print(f"DEBUG: File '{KNOWLEDGE_BASE_NPZ_PATH}' found. Attempting to load with numpy.")
        npz_data = np.load(KNOWLEDGE_BASE_NPZ_PATH, allow_pickle=True)
        
        # Verify expected keys are present
        if 'chunk_url_keys' not in npz_data:
            raise ValueError(f"'{KNOWLEDGE_BASE_NPZ_PATH}' missing 'chunk_url_keys' array.")
        if 'embeddings' not in npz_data:
            raise ValueError(f"'{KNOWLEDGE_BASE_NPZ_PATH}' missing 'embeddings' array.")

        course_chunk_url_keys = npz_data['chunk_url_keys'].tolist()
        course_embeddings = npz_data['embeddings']

        if not course_chunk_url_keys or course_embeddings is None or course_embeddings.shape[0] == 0:
            raise ValueError("Loaded knowledge base is empty or malformed after numpy load.")
        if len(course_chunk_url_keys) != course_embeddings.shape[0]:
            raise ValueError("Mismatched count between chunk_url_keys and embeddings in NPZ file after load.")

        print(f"SUCCESS: Knowledge base loaded successfully.")
        print(f"SUCCESS: Loaded {len(course_chunk_url_keys)} entries (chunk_text, url).")
        print(f"SUCCESS: Embeddings shape: {course_embeddings.shape}")

    except Exception as e:
        print(f"CRITICAL ERROR during knowledge base loading:")
        print(f"  Error Type: {type(e).__name__}")
        print(f"  Error Message: {e}")
        print(f"  Full Traceback:\n{traceback.format_exc()}")
        
        course_chunk_url_keys = []
        course_embeddings = None
        
        raise HTTPException(status_code=500, detail=f"Knowledge base loading failed: {type(e).__name__}: {e}. Check server logs for full details.")


# --- Pydantic Models for API Request/Response (Unchanged) ---
class QueryRequest(BaseModel):
    question: str
    image: str | None = Field(
        None,
        description="Optional base64-encoded image. WebP format is preferred, but this function attempts to process."
    )

class Link(BaseModel):
    url: HttpUrl
    text: str

class APIResponse(BaseModel):
    answer: str
    links: list[Link]


# --- Helper Function for Cosine Similarity (Custom NumPy Implementation - Unchanged) ---
def calculate_cosine_similarity(vec1: np.ndarray, vec2_matrix: np.ndarray) -> np.ndarray:
    vec1 = vec1.reshape(1, -1)
    dot_product = np.dot(vec1, vec2_matrix.T)
    norm_vec1 = np.linalg.norm(vec1, axis=1, keepdims=True)
    norm_vec2 = np.linalg.norm(vec2_matrix, axis=1, keepdims=True).T
    denominators = norm_vec1 * norm_vec2
    denominators[denominators == 0] = 1e-12
    similarities = dot_product / denominators
    return similarities.flatten()


# --- Helper Functions for API Calls (get_gemini_image_description adjusted) ---
async def get_gemini_image_description(image_base64: str) -> str:
    """
    Sends a base64-encoded image to Google Gemini Vision model for description.
    Removed PIL.Image validation as Pillow is no longer a dependency.
    """
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="Google API Key is not configured.")

    url = f"/models/{GEMINI_VISION_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    headers = {"Content-Type": "application/json"}

    try:
        # Determine MIME type from data URI prefix if present
        if "," in image_base64:
            mime_part = image_base64.split(',')[0]
            if 'image/webp' in mime_part:
                mime_type = 'image/webp'
            elif 'image/png' in mime_part:
                mime_type = 'image/png'
            elif 'image/jpeg' in mime_part:
                mime_type = 'image/jpeg'
            else:
                mime_type = 'application/octet-stream' # Fallback if prefix doesn't specify
            base64_data = image_base64.split(',')[1]
        else:
            mime_type = 'application/octet-stream' # Default if no prefix
            base64_data = image_base64

        # Removed: Image.open(BytesIO(base64.b64decode(base64_data))) # No longer using Pillow for validation
        
    except Exception as e:
        # This catch block might now only catch base64 decode errors, not image format errors
        print(f"Warning: Could not process/decode base64 string for image. Error: {e}")
        # Proceeding with generic mime_type and base64_data, API might still reject.
        mime_type = 'application/octet-stream'
        base64_data = image_base64

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": "Provide a detailed summary for this image in 50 words in paragraph. Focus on objects, context, and any visible text or data points."},
                    {"inline_data": {"mime_type": mime_type, "data": base64_data}}
                ]
            }
        ]
    }

    try:
        response = await google_gemini_client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        response_data = response.json()

        if "candidates" in response_data and response_data["candidates"]:
            text_part = response_data["candidates"][0].get("content", {}).get("parts", [{}])[0].get("text", "")
            return text_part.strip()
        else:
            print(f"Gemini response missing candidates or content: {response_data}")
            return "Failed to generate image description due to unexpected API response."

    except httpx.HTTPStatusError as e:
        print(f"Gemini HTTP error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Gemini API error: {e.response.text}")
    except httpx.RequestError as e:
        print(f"Gemini Request error: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini API request failed: {e}")
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON response from Gemini: {e}. Raw response: {response.text if response else 'No response'}")
        raise HTTPException(status_code=500, detail="Failed to parse Gemini API response.")
    except Exception as e:
        print(f"An unexpected error occurred with Gemini API: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error during image processing: {e}")


async def get_openai_embedding(text: str, model: str, dimensions: int = None) -> list[float]:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API Key is not configured.")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    payload = {
        "input": text,
        "model": model,
    }
    if dimensions:
        payload["dimensions"] = dimensions
    try:
        response = await openai_proxy_client.post("/embeddings", headers=headers, json=payload)
        response.raise_for_status()
        response_data = response.json()
        if "data" in response_data and len(response_data["data"]) > 0:
            return response_data["data"][0]["embedding"]
        else:
            raise ValueError(f"Unexpected OpenAI embedding response: {response_data}")
    except httpx.HTTPStatusError as e:
        print(f"OpenAI embedding HTTP error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"OpenAI embedding error: {e.response.text}")
    except httpx.RequestError as e:
        print(f"OpenAI embedding Request error: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI embedding request failed: {e}")
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON response from OpenAI embedding: {e}. Raw response: {response.text if response else 'No response'}")
        raise HTTPException(status_code=500, detail="Failed to parse OpenAI embedding response.")
    except Exception as e:
        print(f"An unexpected error occurred with OpenAI embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error during embedding generation: {e}")


async def get_gpt4o_mini_response(prompt_messages: list[dict]) -> str:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API Key is not configured.")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    payload = {
        "model": LLM_MODEL,
        "messages": prompt_messages,
        "temperature": 0.2,
        "max_tokens": 500,
    }
    try:
        response = await openai_proxy_client.post("/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        response_data = response.json()
        if "choices" in response_data and response_data["choices"]:
            return response_data["choices"][0]["message"]["content"].strip()
        else:
            raise ValueError(f"Unexpected GPT-4o-mini response: {response_data}")
    except httpx.HTTPStatusError as e:
        print(f"GPT-4o-mini HTTP error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"GPT-4o-mini error: {e.response.text}")
    except httpx.RequestError as e:
        print(f"GPT-4o-mini Request error: {e}")
        raise HTTPException(status_code=500, detail=f"GPT-4o-mini request failed: {e}")
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON response from GPT-4o-mini: {e}. Raw response: {response.text if response else 'No response'}")
        raise HTTPException(status_code=500, detail="Failed to parse GPT-4o-mini response.")
    except Exception as e:
        print(f"An unexpected error occurred with GPT-4o-mini: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error during GPT-4o-mini call: {e}")


# --- API Endpoint ---
@app.post("/", response_model=APIResponse)
async def ask_question(request_data: QueryRequest):
    if course_embeddings is None:
        raise HTTPException(status_code=503, detail="Knowledge base is not loaded. Please check server logs for details during startup.")
    if not OPENAI_API_KEY or not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="API Keys not configured. Please ensure OPENAI_API_KEY and GOOGLE_API_KEY environment variables are set.")

    user_question = request_data.question
    image_description = ""
    combined_query_text = user_question

    # 1. Process Image if present
    if request_data.image:
        print("Image detected. Generating description with Gemini...")
        try:
            image_description = await get_gemini_image_description(request_data.image)
            combined_query_text = f"User Question: {user_question}\nImage Context: {image_description}"
            print(f"Gemini image description obtained (first 100 chars): {image_description[:100]}...")
        except HTTPException as e:
            print(f"Error getting image description: {e.detail}. Proceeding without image context.")
            image_description = ""
            combined_query_text = user_question
        except Exception as e:
            print(f"Unexpected error during image processing: {e}. Proceeding without image context.")
            image_description = ""
            combined_query_text = user_question

    # 2. Generate Embedding for combined query text
    print(f"Generating embedding for query: '{combined_query_text[:100]}...'")
    query_embedding_list = await get_openai_embedding(
        text=combined_query_text,
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS
    )
    query_embedding_np = np.array(query_embedding_list).reshape(1, -1)

    # 3. Retrieve Most Similar Chunks
    if course_embeddings.shape[0] == 0:
        raise HTTPException(status_code=500, detail="Knowledge base embeddings are empty. Cannot perform search.")

    print(f"Searching for {NUM_SIMILAR_CHUNKS} most similar chunks...")
    similarities = calculate_cosine_similarity(query_embedding_np, course_embeddings)

    sorted_indices = similarities.argsort()[::-1]
    top_n_indices = sorted_indices[:NUM_SIMILAR_CHUNKS]

    # Retrieve (chunk_text, url) tuples for the top N chunks
    retrieved_chunk_url_pairs = [course_chunk_url_keys[i] for i in top_n_indices]

    # 4. Prepare Link for the response (ONLY THE MOST SIMILAR ONE)
    response_links = []
    if retrieved_chunk_url_pairs:
        most_similar_chunk_text, most_similar_chunk_url = retrieved_chunk_url_pairs[0]
        link_text_display = most_similar_chunk_text
        try:
            response_links.append(Link(url=most_similar_chunk_url, text=link_text_display))
            print(f"Selected top link: URL={most_similar_chunk_url}, Text='{link_text_display}'")
        except Exception as e:
            print(f"Warning: Could not create Link object for URL '{most_similar_chunk_url}': {e}")
    else:
        print("No chunks retrieved, links array will be empty.")


    # 5. Construct Prompt for GPT-4o-mini
    context_string = "\n\n".join([
        f"Context Document {idx+1}: {chunk_text}"
        for idx, (chunk_text, url) in enumerate(retrieved_chunk_url_pairs)
    ])

    user_message_parts = [
        f"Course Materials Context:\n{context_string}\n\n",
        f"Student's Question: {user_question}\n"
    ]

    if image_description:
        user_message_parts.append(f"Image Description: {image_description}\n")

    user_message_parts.append(
        "Please provide a concise answer based ONLY on the provided context. If the context "
        "is insufficient, state so clearly."
    )
    user_message_content = "".join(user_message_parts)

    prompt_messages = [
        {"role": "system", "content": (
            "You are a helpful AI assistant for a university course. "
            "Your knowledge is strictly limited to the provided course materials. "
            "If you cannot find an answer within the provided context, state that you don't have enough information. "
            "Do not make up answers. "
            "Provide a concise answer directly based on the context."
        )},
        {"role": "user", "content": user_message_content}
    ]

    # 6. Call GPT-4o-mini
    print("Calling GPT-4o-mini for answer generation...")
    answer = await get_gpt4o_mini_response(prompt_messages)
    print(f"Generated answer: {answer[:150]}...")

    # 7. Return Response
    print("Returning API response.")
    return APIResponse(answer=answer, links=response_links)

# --- Shutdown Event ---
@app.on_event("shutdown")
async def shutdown_event():
    """Closes HTTPX clients gracefully on shutdown."""
    await openai_proxy_client.aclose()
    await google_gemini_client.aclose()
    print("HTTPX clients closed.")
