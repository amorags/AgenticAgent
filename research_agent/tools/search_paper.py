import requests
from typing import List, Dict, Optional, Literal
from datetime import datetime
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def search_papers(
    topic: str,
    year: Optional[int] = None,
    min_citations: Optional[int] = None,
    year_filter: Literal["exact", "before", "after"] = "exact",
    limit: int = 5
) -> List[Dict]:
    """
    Search for research papers using Semantic Scholar API.
    
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
    
    try:
        # Prepare headers with API key if available
        headers = {}
        
        # Add delay to respect rate limits (important for unauthenticated requests)
        time.sleep(1.5)  # 1.5 seconds between requests to stay well under limit
        
        # Make the API request
        response = requests.get(base_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()  # Raise error for bad status codes
        
        data = response.json()
        
        # Extract papers from response
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
        print(f"Error fetching papers: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []


def get_paper_details(paper_id: str) -> Optional[Dict]:
    """
    Get detailed information about a specific paper by its Semantic Scholar ID.
    
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
        time.sleep(1.5)
            
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching paper details: {e}")
        return None