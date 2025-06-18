import requests
import json
import os
from datetime import datetime, timezone
from bs4 import BeautifulSoup

# Configuration
DISCOURSE_URL = "https://discourse.onlinedegree.iitm.ac.in"
CATEGORY_ID = 34  # TDS-KB
RAW_OUTPUT_FILE = "raw_topic_posts.json"
PROCESSED_OUTPUT_FILE = "processed_posts.json"
START_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2025, 5, 31, 23, 59, 59, 999999, tzinfo=timezone.utc)

# Cookies extracted from browser
browser_cookies = {
    "_t": "iIu8AfQwWMR4BLZHQiM8XqsKZQsuLPKVY8gwSLbg87t0AyZCf1qQwUEjYctAzFIxtlUFWsbZ5xH6INb%2BXvVD685h%2F9ZKZhilSy9iaAd%2F07ezSbUicGqv8Nfo4pO8RnX9i24%2ByHxx1VKS1YGiKk%2BEv%2BBtz367qKpX%2FBt5O47vIlGODgdg5BnGFogfuFoVi%2FlK539xLsYQwTzOLxguzoWOupAw%2F2mjYUmSZu5nJv%2BD%2FCYiDOyczhLlt7UgeiFEX%2FbIGgaYpz%2FvIbIIP%2BQpJUTuoJdot3vi6O14nILxUHmgImsYnDjO2xCLCMxqq7slDqC6--P%2BqGtbkbbok5JvW5--p%2BHY1mNEVTc%2BZ3wKza5Pvg%3D%3D",
    "_forum_session": "7SLVk4lcOqoQN%2BJk3HsGwWlKHfqTkSBoRJIJX84xqwg39Rd9xTCzQyvhAfi%2FBAPzcdFCiFOx7v2t8Pzm%2FUU89uCgNXeYOa2%2BX1UtDoRmxiVC%2Fm1W5B0yxQZrqd59cIZz%2FUDtxNFXPQrwxpXB99AwLKfF1jKNGGyoH0nQW8UL3VpESD6Cd3f6AIl8z%2FGKtyr0h2zVBUEUjzU%2BlPMpciWVZKeYoVgewE%2Begg%2FSSc1TlYeRUyKi1hqJqlR6XW4OihsyCTXqtqq08r9XqmWFSlaOg9uj8RGPXzRti2rwiuuK74mvPGERPAcAcPUcFlS30aHhkY4K%2F2ha%2B14zxgRQumrfLxzGBFlU1EkAWT5s%2F1AdU1Cati9UtdJ4PoeyTOpbqUmSBuup9ygmLB3TQuXYGWA%2BeCaER1BDsf2spZcT26yWmSVD%2FUnLZdA2wL1PxsPPWLhLC5UuUcrmW19GdCqS6bvKD5PvGSdHvZnvFwpeNeh7JMWxaOBprjEEzRMxb%2B7sog%3D%3D--HcKLH64V31Sfwm40--vPCumtdgFPuh%2BV4b6fGhKQ%3D%3D"
}

def create_session_with_browser_cookies(discourse_url, cookies):
    """Create a requests session with browser cookies."""
    session = requests.Session()
    for name, value in cookies.items():
        session.cookies.set(name, value, domain=discourse_url.split("//")[1])
    return session

def parse_date(date_str):
    """Convert ISO date string to datetime."""
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except ValueError as e:
        print(f"Invalid date format: {date_str}, error: {e}")
        return None

def clean_html(text):
    """Remove HTML tags and convert to plain text."""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ", strip=True)

def extract_image_urls(cooked):
    """Extract all image URLs from cooked HTML."""
    soup = BeautifulSoup(cooked, "html.parser")
    img_tags = soup.find_all("img")
    return [img["src"] for img in img_tags if img.get("src")]

def fetch_topic_ids(session):
    """Fetch topic IDs from category 34 within date range."""
    page = 1
    topic_ids = []
    
    while True:
        url = f"{DISCOURSE_URL}/c/{CATEGORY_ID}.json?page={page}"
        try:
            response = session.get(url)
            response.raise_for_status()
            data = response.json()
            topics = data.get("topic_list", {}).get("topics", [])
            
            if not topics:
                break
            
            for topic in topics:
                created_at_str = topic.get("created_at")
                topic_id = topic.get("id")
                if not created_at_str or not topic_id:
                    print(f"Skipping topic with missing data: {topic.get('title', 'unknown')}")
                    continue
                
                created_at = parse_date(created_at_str)
                if created_at and START_DATE <= created_at <= END_DATE:
                    topic_ids.append(topic_id)
                    print(f"Added topic ID {topic_id}: {topic.get('title')} ({created_at_str})")
                else:
                    print(f"Skipped topic ID {topic_id}: {created_at_str} outside date range")
            
            page += 1
            # Avoid rate limits
            time.sleep(0.5)
        
        except requests.RequestException as e:
            print(f"Error fetching page {page}: {e}")
            break
    
    return topic_ids

def fetch_topic_posts(session, topic_id):
    """Fetch all post JSONs for a topic, handling pagination."""
    page = 1
    topic_jsons = []
    
    while True:
        url = f"{DISCOURSE_URL}/t/{topic_id}.json?page={page}"
        try:
            response = session.get(url)
            response.raise_for_status()
            data = response.json()
            posts = data.get("post_stream", {}).get("posts", [])
            
            if not posts:
                break
            
            topic_jsons.append(data)
            print(f"Fetched {len(posts)} posts for topic {topic_id}, page {page}")
            
            if len(posts) < 20:  # Default posts per page
                break
            page += 1
            time.sleep(0.5)
        
        except requests.RequestException as e:
            print(f"Error fetching posts for topic {topic_id}, page {page}: {e}")
            break
    
    return topic_jsons

def process_posts(topic_jsons):
    """Extract text and image URLs from posts."""
    processed_posts = []
    
    for data in topic_jsons:
        posts = data.get("post_stream", {}).get("posts", [])
        topic_posts = []
        
        for post in posts:
            cooked = post.get("cooked", "")
            topic_posts.append({
                "text": clean_html(cooked),
                "image_url": extract_image_urls(cooked)
            })
        
        processed_posts.append(topic_posts)
    
    return processed_posts

def main():
    # Create authenticated session
    session = create_session_with_browser_cookies(DISCOURSE_URL, browser_cookies)
    
    # Verify authentication
    try:
        response = session.get(f"{DISCOURSE_URL}/session/current.json")
        response.raise_for_status()
        user_data = response.json()
        print(f"Authenticated as: {user_data['current_user']['username']}")
    except (requests.RequestException, KeyError) as e:
        print(f"Authentication failed: {e}")
        return
    
    # Fetch topic IDs
    print("Fetching topic IDs from category 34...")
    topic_ids = fetch_topic_ids(session)
    
    if not topic_ids:
        print("No topics found in the date range.")
        return
    
    print(f"Retrieved {len(topic_ids)} topic IDs.")
    
    # Fetch and process posts
    all_raw_jsons = []
    all_processed_posts = []
    
    for topic_id in topic_ids:
        print(f"Processing topic ID {topic_id}")
        topic_jsons = fetch_topic_posts(session, topic_id)
        if not topic_jsons:
            print(f"No posts found for topic {topic_id}")
            continue
        
        all_raw_jsons.extend(topic_jsons)
        processed_posts = process_posts(topic_jsons)
        all_processed_posts.extend(processed_posts)
    
    # Save raw JSONs
    try:
        with open(RAW_OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(all_raw_jsons, f, indent=2, ensure_ascii=False)
        print(f"Saved raw JSONs to {RAW_OUTPUT_FILE}")
    except Exception as e:
        print(f"Error saving raw JSONs: {e}")
    
    # Save processed posts
    try:
        with open(PROCESSED_OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(all_processed_posts, f, indent=2, ensure_ascii=False)
        print(f"Saved processed posts to {PROCESSED_OUTPUT_FILE}")
    except Exception as e:
        print(f"Error saving processed posts: {e}")

if __name__ == "__main__":
    try:
        import time
        main()
    except Exception as e:
        print(f"Error: {e}")
