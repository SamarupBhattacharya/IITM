import json

# Configuration
INPUT_JSON_FILE = "processed_posts_with_exp.json"
OUTPUT_FILE_PREFIX = "d"

def save_batches_to_markdown():
    """Load JSON, extract strings from each batch, and save to separate Markdown files."""
    try:
        # Read JSON file
        with open(INPUT_JSON_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Verify data is a list
        if not isinstance(data, list):
            print(f"Error: {INPUT_JSON_FILE} does not contain a list.")
            return
        
        # Process each batch
        for batch_idx, batch in enumerate(data, 1):
            if not isinstance(batch, list):
                print(f"Warning: Batch {batch_idx} is not a list, skipping.")
                continue
            
            # Extract strings from text and image_url fields
            batch_strings = []
            for json_obj in batch:
                if not isinstance(json_obj, dict):
                    print(f"Warning: Invalid JSON in batch {batch_idx}, skipping.")
                    continue
                
                # Get text field
                text = json_obj.get("text", "")
                if isinstance(text, str) and text:
                    batch_strings.append(text)
                else:
                    print(f"Warning: Invalid or empty text in batch {batch_idx}: {json_obj}")
                
                # Get image_url field
                image_urls = json_obj.get("image_url", [])
                if isinstance(image_urls, list):
                    for url in image_urls:
                        if isinstance(url, str) and url:
                            batch_strings.append(url)
                        else:
                            print(f"Warning: Invalid URL in batch {batch_idx}: {url}")
                else:
                    print(f"Warning: Invalid image_url in batch {batch_idx}: {image_urls}")
            
            if not batch_strings:
                print(f"Warning: No valid strings found in batch {batch_idx}, skipping file creation.")
                continue
            
            # Write to Markdown file
            output_file = f"{OUTPUT_FILE_PREFIX}{batch_idx}.md"
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    # Write each string on a new line
                    for s in batch_strings:
                        f.write(f"{s}\n")
                print(f"Saved {len(batch_strings)} strings to {output_file}")
            except Exception as e:
                print(f"Error saving {output_file}: {e}")
        
        print(f"Processed {len(data)} batches, created {batch_idx} Markdown files.")
    
    except FileNotFoundError:
        print(f"Error: {INPUT_JSON_FILE} not found.")
    except json.JSONDecodeError:
        print(f"Error: {INPUT_JSON_FILE} contains invalid JSON.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    save_batches_to_markdown()
