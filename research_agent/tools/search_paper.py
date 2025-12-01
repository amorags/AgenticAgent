import requests
from typing import List, Dict, Optional, Literal
from datetime import datetime
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Retry Configuration ---
MAX_RETRY_TIME = 20  # seconds
INITIAL_BACKOFF = 1  # seconds
BACKOFF_FACTOR = 2   # exponential backoff factor
MAX_RETRIES = 5      # Limit number of retries to prevent long initial wait times

def search_papers(
    topic: str,
    year: Optional[int] = None,
    min_citations: Optional[int] = None,
    year_filter: Literal["exact", "before", "after"] = "exact",
    limit: int = 10
) -> List[Dict]:
    """
    Search for research papers using Semantic Scholar API with retry logic.
    
    Args:
        topic: The research topic to search for
        year: The year to filter by (optional)
        min_citations: Minimum number of citations (optional)
        year_filter: How to apply year filter - "exact", "before", or "after"
        limit: Maximum number of results to return (default 10)
    
    Returns:
        List of papers with their metadata (title, authors, year, citations, etc.)
    
    Example:
        papers = search_papers("machine learning", year=2020, min_citations=100)
    """
    
    # Base API endpoint
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    # Build query parameters
    params = {
        "query": topic,
        "limit": limit,
        "fields": "title,authors,year,citationCount,abstract,url,venue,publicationDate"
    }
    
    # Add year filter if specified
    if year is not None:
        if year_filter == "exact":
            params["year"] = str(year)
        elif year_filter == "before":
            params["year"] = f"-{year}"  # e.g., "-2020" means up to 2020
        elif year_filter == "after":
            params["year"] = f"{year}-"  # e.g., "2020-" means from 2020 onwards
    
    # Add minimum citation count if specified
    if min_citations is not None:
        params["minCitationCount"] = min_citations
    
    # Prepare headers with API key if available
    headers = {}
    
    # --- Retry Loop Setup ---
    total_wait_time = 0
    current_backoff = INITIAL_BACKOFF
    
    for attempt in range(MAX_RETRIES):
        try:
            # 1. Add delay to respect rate limits (important for unauthenticated requests)
            # This is done BEFORE the request. If the request fails, the backoff handles the next delay.
            if attempt == 0:
                 time.sleep(1.5)  # Initial delay for rate limiting
            else:
                 # Sleep using calculated backoff time
                 time.sleep(current_backoff)
                 total_wait_time += current_backoff
                 print(f"Retrying request (Attempt {attempt + 1}/{MAX_RETRIES}). Waiting {current_backoff:.2f}s. Total wait: {total_wait_time:.2f}s...")
            
            # 2. Check if total retry time limit is exceeded
            if total_wait_time >= MAX_RETRY_TIME and attempt > 0:
                print(f"Total retry wait time exceeded {MAX_RETRY_TIME} seconds. Aborting retry.")
                break # Exit the loop and let the final attempt (or the exception handling below) take over

            # 3. Make the API request
            response = requests.get(base_url, params=params, headers=headers, timeout=10)
            response.raise_for_status()  # Raise error for bad status codes (4xx or 5xx)
            
            # 4. Success: Process response and return
            data = response.json()
            papers = data.get("data", [])
            
            # Format the results
            formatted_papers = []
            for paper in papers:
                formatted_paper = {
                    "title": paper.get("title", "Unknown"),
                    "authors": [author.get("name", "Unknown") for author in paper.get("authors", [])],
                    "year": paper.get("year"),
                    "citations": paper.get("citationCount", 0),
                    "abstract": paper.get("abstract", "No abstract available"),
                    "url": paper.get("url", ""),
                    "venue": paper.get("venue", "Unknown venue"),
                    "publication_date": paper.get("publicationDate", "Unknown")
                }
                formatted_papers.append(formatted_paper)
            
            return formatted_papers
        
        except requests.exceptions.RequestException as e:
            # Check for recoverable errors (e.g., 5xx status codes, timeouts, connection errors)
            # Semantic Scholar recommends retrying on 429 (Too Many Requests) and all 5xx.
            # requests.raise_for_status() handles the 4xx/5xx part, RequestException handles connection/timeout.
            
            is_recoverable = True
            if response is not None and response.status_code < 500 and response.status_code != 429:
                # If it's a client error (e.g., 400, 404, 403) and not 429, don't retry.
                is_recoverable = False

            if not is_recoverable or attempt == MAX_RETRIES - 1:
                # Last attempt or non-recoverable error. Exit loop/raise error.
                print(f"Error fetching papers after {attempt + 1} attempts: {e}")
                return []

            # 5. Prepare for next retry with exponential backoff
            print(f"Request failed ({e}). Retrying...")
            current_backoff *= BACKOFF_FACTOR
            
            # Ensure the next wait time doesn't exceed the overall limit
            if total_wait_time + current_backoff >= MAX_RETRY_TIME:
                # Adjust backoff time to hit the limit exactly for the final retry
                current_backoff = MAX_RETRY_TIME - total_wait_time 
                if current_backoff <= 0:
                    print(f"Total retry wait time exceeded {MAX_RETRY_TIME} seconds. Aborting retry.")
                    return []
        
        except Exception as e:
            # Handle unexpected errors
            print(f"Unexpected error: {e}")
            return []

    # If the loop finishes without returning, it means all retries failed or the time limit was hit
    return []


def get_paper_details(paper_id: str) -> Optional[Dict]:
    """
    Get detailed information about a specific paper by its Semantic Scholar ID.
    (Retry logic not added to this function as per prompt, but included for completeness)
    
    Args:
        paper_id: The Semantic Scholar paper ID
    
    Returns:
        Dictionary with paper details or None if not found
    """
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
    params = {
        "fields": "title,authors,year,citationCount,abstract,url,venue,publicationDate,references,citations"
    }
    
    try:
        headers = {}
        
        # Add delay for rate limiting
        time.sleep(5)
            
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching paper details: {e}")
        return None