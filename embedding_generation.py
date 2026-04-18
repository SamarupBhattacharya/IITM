import os
import numpy as np
import httpx # Import httpx
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
import json 

OPENAI_API_KEY = "" # Your OpenAI API Key here

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536 

CHUNK_SIZE_CHARS = 2000
CHUNK_OVERLAP_CHARS = int(CHUNK_SIZE_CHARS * 0.20) # 20% overlap

OUTPUT_NPZ_FILENAME = 'course_knowledge_base.npz'

OPENAI_EMBEDDING_URL = "https://api.openai.com/v1/embeddings"

http_client = httpx.Client(timeout=30.0)

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
        response.raise_for_status()
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

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE_CHARS,
    chunk_overlap=CHUNK_OVERLAP_CHARS,
    length_function=len,
    separators=[
        "\n\n# ",    
        "\n\n## ",  
        "\n\n### ",  
        "\n\n```",  
        "\n\n",      
        "\n",       
        " ",        
        "",     
    ],
    is_separator_regex=False,
)

all_chunks = []
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

            file_chunks = text_splitter.split_text(content)
            print(f"  - Generated {len(file_chunks)} chunks for '{filename}'")

            for i, chunk in enumerate(file_chunks):
                all_chunks.append(chunk)
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

print(f"\nTotal chunks generated: {len(all_chunks)}")

if not all_chunks:
    print("No chunks to embed. Exiting.")
    exit()

print("Generating embeddings for all chunks. This may take a while...")
final_embeddings_list = []
BATCH_SIZE = 50 

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
        time.sleep(1)

    except Exception as e:
        print(f"Error processing batch starting at index {i}: {e}")
        print("Skipping this batch and attempting to continue with subsequent batches.")

print(f"\nSuccessfully generated embeddings for {len(final_embeddings_list)} chunks.")

if len(all_chunks) != len(final_embeddings_list):
    print(f"Warning: Number of embeddings ({len(final_embeddings_list)}) does not match number of original chunks ({len(all_chunks)}).")
    print("This might happen if some API calls failed. Only successfully embedded chunks will be saved.")
    all_chunks_to_save = all_chunks[:len(final_embeddings_list)]
else:
    all_chunks_to_save = all_chunks
final_embeddings_array = np.array(final_embeddings_list)

try:
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

http_client.close()
print("\nProcess completed.")
