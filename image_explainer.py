import requests
import json
import base64
import time
from datetime import datetime # Though not directly used in this specific script, kept for consistency

# --- Configuration ---
INPUT_FILENAME = "processed_posts.json"
OUTPUT_FILENAME = "processed_posts_with_explanations.json"

# Gemini API configuration
# IMPORTANT: As per instructions, leave API_KEY as an empty string.
# The Canvas environment will automatically provide it at runtime.
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
API_KEY = "AIzaSyDqHqkYWWr_VlNmsL-SYK4wKl4tPElJmhw" 

# --- Helper Functions ---

def get_mime_type(url_content_type):
    """
    Derives a suitable MIME type for the Gemini API from the Content-Type header.
    Gemini-2.0-Flash primarily supports image/png, image/jpeg, and image/webp.
    """
    if 'image/jpeg' in url_content_type:
        return 'image/jpeg'
    elif 'image/png' in url_content_type:
        return 'image/png'
    elif 'image/webp' in url_content_type:
        return 'image/webp'
    # Return None if the MIME type is not explicitly supported or recognized
    return None

def explain_image_with_gemini(base64_image_data, mime_type, prompt_text="Explain and caption this image in detail."):
    """
    Sends a Base64 encoded image to Gemini-2.0-Flash for explanation.
    
    Args:
        base64_image_data (str): The Base64 encoded image string.
        mime_type (str): The MIME type of the image (e.g., 'image/jpeg').
        prompt_text (str): The text prompt for the model.
        
    Returns:
        str: The generated explanation from the model, or an error message if the API call fails.
    """
    if not mime_type:
        return "Error: Unsupported image MIME type provided for Gemini API."

    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt_text},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": base64_image_data
                        }
                    }
                ]
            }
        ]
    }

    try:
        # Construct the API URL with the dynamically provided API_KEY
        api_call_url = f"{GEMINI_API_URL}?key={API_KEY}"
        
        response = requests.post(api_call_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        
        result = response.json()

        # Navigate through the JSON response to extract the text explanation
        if result.get('candidates') and result['candidates'][0].get('content') and \
           result['candidates'][0]['content'].get('parts') and result['candidates'][0]['content']['parts'][0].get('text'):
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            # If the expected structure is not found, return a detailed error
            return f"Error: Unexpected Gemini API response structure. Raw response: {json.dumps(result, indent=2)}"
    except requests.exceptions.Timeout:
        return "Error: Gemini API request timed out."
    except requests.exceptions.RequestException as e:
        return f"Error calling Gemini API: {e}"
    except json.JSONDecodeError:
        return "Error: Could not decode JSON response from Gemini API."
    except Exception as e:
        return f"An unexpected error occurred during Gemini API call: {e}"

def process_image_urls_in_posts(posts_data):
    """
    Processes each image URL within the loaded posts data:
    1. Checks if the image URL is accessible.
    2. Downloads and Base64 encodes the image.
    3. Sends the encoded image to Gemini-2.0-Flash for detailed explanation.
    4. Replaces the original image URL with the generated explanation (or error message).
    
    Args:
        posts_data (list): The loaded data from processed_posts.json,
                           expected to be a list of lists of JSON objects.
        
    Returns:
        list: The modified posts data with image URLs replaced by explanations.
    """
    processed_posts_data = []
    total_image_urls_processed = 0
    
    # Iterate through the outer list (list of post lists)
    for list_index, post_list in enumerate(posts_data):
        processed_inner_list = []
        # Iterate through the inner list (list of individual post JSONs)
        for post_index, post_json in enumerate(post_list):
            
            # Check if 'image_url' field exists and is a list
            if 'image_url' in post_json and isinstance(post_json['image_url'], list):
                original_image_urls = post_json['image_url']
                new_image_explanations = [] # List to store explanations for this post

                # Iterate through each image URL in the 'image_url' list
                for url_index, img_url in enumerate(original_image_urls):
                    print(f"--- Processing Image {url_index + 1} for Post {post_index + 1} in Batch {list_index + 1} ---")
                    print(f"Attempting to access: {img_url}")
                    explanation_or_error = "" # Default value if processing fails
                    
                    try:
                        # Attempt to download the image content
                        # Set a reasonable timeout for network requests
                        img_response = requests.get(img_url, timeout=30)
                        img_response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

                        # Get Content-Type from headers to determine MIME type
                        content_type = img_response.headers.get('Content-Type', '')
                        mime_type = get_mime_type(content_type)
                        
                        # Check for supported MIME type
                        if not mime_type:
                            explanation_or_error = f"Image URL '{img_url}' has an unsupported MIME type ('{content_type}'). Skipping explanation."
                            print(explanation_or_error)
                        # Check image size against Gemini's 7MB limit
                        elif img_response.content and len(img_response.content) <= (7 * 1024 * 1024):
                            # Base64 encode the image content
                            base64_encoded_image = base64.b64encode(img_response.content).decode('utf-8')
                            print(f"  Image successfully downloaded and Base64 encoded. Size: {len(img_response.content) / (1024*1024):.2f} MB")
                            
                            # Send to Gemini for explanation
                            print(f"  Sending image to GenAI-2.0-Flash for explanation...")
                            explanation_or_error = explain_image_with_gemini(base64_encoded_image, mime_type)
                            print(f"  Explanation (first 100 chars): {explanation_or_error[:100]}...") # Print a snippet
                            total_image_urls_processed += 1
                        else:
                            # Handle images that are too large or have no content
                            size_mb = len(img_response.content) / (1024*1024) if img_response.content else 0
                            explanation_or_error = f"Image URL '{img_url}' is too large ({size_mb:.2f}MB > 7MB limit) or has no content. Skipping explanation."
                            print(explanation_or_error)

                    except requests.exceptions.Timeout:
                        explanation_or_error = f"Error: Network request timed out while accessing image URL '{img_url}'."
                        print(explanation_or_error)
                    except requests.exceptions.RequestException as e:
                        explanation_or_error = f"Error: Could not access image URL '{img_url}'. Details: {e}"
                        print(explanation_or_error)
                    except Exception as e:
                        explanation_or_error = f"An unexpected error occurred while processing image '{img_url}'. Details: {e}"
                        print(explanation_or_error)
                    
                    new_image_explanations.append(explanation_or_error)
                    time.sleep(1) # Pause for 1 second to avoid hitting API rate limits or overwhelming the image server
                
                # Replace the original list of image URLs with the new list of explanations
                post_json['image_url'] = new_image_explanations
            processed_inner_list.append(post_json)
        processed_posts_data.append(processed_inner_list)
    
    print(f"\n--- Image Processing Complete ---")
    print(f"Total image URLs successfully processed and explained: {total_image_urls_processed}")
    return processed_posts_data

# --- Main Execution Block ---
if __name__ == "__main__":
    print(f"Attempting to load data from '{INPUT_FILENAME}'...")
    try:
        with open(INPUT_FILENAME, 'r', encoding='utf-8') as f:
            posts_data = json.load(f)
        print(f"Data loaded successfully from '{INPUT_FILENAME}'.")
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILENAME}' not found. Please ensure it exists in the same directory.")
        exit(1) # Exit with an error code
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{INPUT_FILENAME}'. Please check if the file content is valid JSON.")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading the file: {e}")
        exit(1)

    print("\nStarting the process of accessing images and generating explanations with GenAI-2.0-Flash...")
    modified_posts_data = process_image_urls_in_posts(posts_data)

    print(f"\nSaving processed data to '{OUTPUT_FILENAME}'...")
    try:
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            # Use indent for pretty-printing the JSON output
            json.dump(modified_posts_data, f, ensure_ascii=False, indent=4)
        print(f"Processed data successfully saved to '{OUTPUT_FILENAME}'.")
    except IOError as e:
        print(f"Error saving processed data to file: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while saving the file: {e}")
        exit(1)
