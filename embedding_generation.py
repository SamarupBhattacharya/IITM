import os
import numpy as np
import httpx # Import httpx
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time # For rate limiting
import json # For JSON serialization/deserialization

# --- Configuration ---
# IMPORTANT: Replace with your actual OpenAI API Key or set as an environment variable
# It's recommended to set it as an environment variable (OPENAI_API_KEY)
# For example: os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"
# If running in a Canvas environment, the API key might be automatically provided.
# You can leave it as an empty string if Canvas is handling it, but for local testing, set it.
OPENAI_API_KEY = "" # Your OpenAI API Key here

# OpenAI Embedding Model to use
EMBEDDING_MODEL = "text-embedding-3-small"
# The default dimension for text-embedding-3-small is 1536.
# You can optionally set a lower dimension if needed for storage/performance,
# e.g., dimensions=512, but for best performance, use default.
EMBEDDING_DIMENSIONS = 1536 # Default for text-embedding-3-small

# Chunking parameters
# text-embedding-3-small has a max_tokens of 8191.
# A common recommendation is to keep chunks around 250-500 tokens for embeddings.
# Let's target ~500 tokens as a comfortable upper bound for content within a chunk
# to ensure context while staying well within the model's limits.
# Approximate character to token ratio is 4 characters per token.
# So, 500 tokens * 4 chars/token = 2000 characters.
CHUNK_SIZE_CHARS = 2000
CHUNK_OVERLAP_CHARS = int(CHUNK_SIZE_CHARS * 0.20) # 20% overlap

OUTPUT_NPZ_FILENAME = 'course_knowledge_base.npz'

# OpenAI API Endpoint for embeddings
OPENAI_EMBEDDING_URL = "https://api.openai.com/v1/embeddings"

# --- Initialize HTTPX Client ---
# Use a global httpx client for connection pooling
http_client = httpx.Client(timeout=30.0) # Set a reasonable timeout

# --- Helper function to get embeddings using httpx ---
def get_embeddings_with_httpx(texts: list[str], model: str, dimensions: int = None):
    """
    Generates embeddings for a list of texts using OpenAI's API via httpx.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set. Cannot make API calls.")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    payload = {
        "input": texts,
        "model": model,
    }
    if dimensions:
        payload["dimensions"] = dimensions

    try:
        response = http_client.post(OPENAI_EMBEDDING_URL, headers=headers, json=payload)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        response_data = response.json()

        if "data" in response_data and len(response_data["data"]) > 0:
            return [d["embedding"] for d in response_data["data"]]
        else:
            raise ValueError(f"Unexpected API response structure: {response_data}")

    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
        raise
    except httpx.RequestError as e:
        print(f"An error occurred while requesting {e.request.url!r}: {e}")
        raise
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON response: {e}")
        print(f"Raw response text: {response.text}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

# --- Initialize Markdown Text Splitter ---
# We'll use a recursive splitter with Markdown-aware separators to preserve structure.
# This prioritizes splitting by larger semantic units (like headings, code blocks) first.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE_CHARS,
    chunk_overlap=CHUNK_OVERLAP_CHARS,
    length_function=len, # Use character length for splitting
    separators=[
        "\n\n# ",    # Split by H1 headings
        "\n\n## ",   # Split by H2 headings
        "\n\n### ",  # Split by H3 headings
        "\n\n```",   # Split before/after code blocks (important for mixed content)
        "\n\n",      # Paragraph breaks
        "\n",        # Line breaks
        " ",         # Spaces (last resort)
        "",          # Fallback to single character split
    ],
    is_separator_regex=False, # Set to True if separators are regex patterns
)

# --- Process Markdown Files ---
all_chunks = []
# Metadata to store with each chunk (e.g., source filename, original chunk text)
# This will be useful for debugging or future features if you move to a DB.
chunk_metadata = []

print(f"Scanning for Markdown files in: {os.getcwd()}")
md_files_found = 0

for filename in os.listdir('.'):
    if filename.endswith(".md"):
        md_files_found += 1
        filepath = os.path.join('.', filename)
        print(f"Processing file: {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Create chunks for the current file
            file_chunks = text_splitter.split_text(content)
            print(f"  - Generated {len(file_chunks)} chunks for '{filename}'")

            # Add chunks and their metadata to the master list
            for i, chunk in enumerate(file_chunks):
                all_chunks.append(chunk)
                # Store enough information to trace back the chunk if needed
                chunk_metadata.append({
                    "filename": filename,
                    "chunk_index": i,
                    "original_content_start": content.find(chunk) # Approximate start
                })

        except Exception as e:
            print(f"Error processing file '{filename}': {e}")

if md_files_found == 0:
    print("No Markdown files (.md) found in the current directory.")
    print("Please ensure your Markdown files are in the same directory as this script.")
    # Consider exiting here if no files are found to avoid further errors.
    # For now, we'll let it continue to the embedding step, which will also exit.
    # exit()

print(f"\nTotal chunks generated: {len(all_chunks)}")

# --- Generate Embeddings ---
if not all_chunks:
    print("No chunks to embed. Exiting.")
    exit()

print("Generating embeddings for all chunks. This may take a while...")
final_embeddings_list = []
# Batching for efficiency and to reduce API calls
# The text-embedding-3-small model can take up to 8191 tokens, so batching by number of chunks
# (e.g., 50-100 chunks per request) is usually safe as long as total tokens per batch are within limits.
BATCH_SIZE = 50 # Number of chunks to send in one API request

for i in range(0, len(all_chunks), BATCH_SIZE):
    batch_chunks = all_chunks[i : i + BATCH_SIZE]
    print(f"  - Embedding batch {i // BATCH_SIZE + 1} ({len(batch_chunks)} chunks)...")
    try:
        batch_embeddings = get_embeddings_with_httpx(
            texts=batch_chunks,
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMENSIONS
        )
        final_embeddings_list.extend(batch_embeddings)

        # Implement a more robust rate-limiting strategy (e.g., exponential backoff)
        # For simplicity, a fixed sleep is used here. Adjust based on OpenAI's rate limits.
        time.sleep(1) # Sleep for 1 second between batches to respect API limits

    except Exception as e:
        print(f"Error processing batch starting at index {i}: {e}")
        print("Skipping this batch and attempting to continue with subsequent batches.")
        # If an error occurs, the embeddings for this batch won't be added.
        # You might want to handle this more robustly (e.g., saving failed chunks for retry).

print(f"\nSuccessfully generated embeddings for {len(final_embeddings_list)} chunks.")

# It's crucial that the number of embeddings matches the number of chunks processed.
# If an error occurred and a batch was skipped, `len(final_embeddings_list)` might be less than `len(all_chunks)`.
# For the .npz storage, we'll only save chunks for which we successfully got embeddings.
if len(all_chunks) != len(final_embeddings_list):
    print(f"Warning: Number of embeddings ({len(final_embeddings_list)}) does not match number of original chunks ({len(all_chunks)}).")
    print("This might happen if some API calls failed. Only successfully embedded chunks will be saved.")
    # Filter all_chunks to only include those for which we have embeddings
    # (This assumes that if an embedding fails, we don't append it to final_embeddings_list)
    # For a robust solution, you might store chunks and embeddings in a way that
    # maintains a 1:1 mapping, even if some embedding calls fail.
    # For this example, we proceed with the embeddings we have, assuming they correspond
    # to the initial segment of `all_chunks`.
    all_chunks_to_save = all_chunks[:len(final_embeddings_list)]
else:
    all_chunks_to_save = all_chunks

# Convert list of embeddings to a single NumPy array
final_embeddings_array = np.array(final_embeddings_list)

# --- Store Chunks and Embeddings in .npz file ---
try:
    # Use allow_pickle=True because 'chunks' is a list of strings (Python objects)
    np.savez(
        OUTPUT_NPZ_FILENAME,
        chunks=np.array(all_chunks_to_save, dtype=object), # Save chunks as an array of objects
        embeddings=final_embeddings_array
    )
    print(f"\nChunks and embeddings successfully saved to '{OUTPUT_NPZ_FILENAME}'")
    print(f"  - Number of chunks saved: {len(all_chunks_to_save)}")
    print(f"  - Shape of embeddings array: {final_embeddings_array.shape}")
except Exception as e:
    print(f"Error saving to .npz file: {e}")

# Close the httpx client after use
http_client.close()
print("\nProcess completed.")
