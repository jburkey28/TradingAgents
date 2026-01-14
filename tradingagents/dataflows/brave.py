import os
import requests
from datetime import datetime, timedelta
from typing import Annotated


def get_global_news_brave(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "Number of days to look back"] = 7,
    limit: Annotated[int, "Maximum number of articles to return"] = 5,
) -> str:
    """
    Retrieve global news using Brave Search API.
    
    Requires BRAVE_API_KEY environment variable.
    """
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        raise ValueError("BRAVE_API_KEY environment variable not set")
    
    # Calculate date range
    end_date = datetime.strptime(curr_date, "%Y-%m-%d")
    start_date = end_date - timedelta(days=look_back_days)
    freshness = f"{start_date.strftime('%Y-%m-%d')}to{end_date.strftime('%Y-%m-%d')}"
    
    url = "https://api.search.brave.com/res/v1/news/search"
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key
    }
    params = {
        "q": "global macroeconomic financial market news",
        "count": min(limit, 50),  # Brave max is 50
        "country": "US",
        "search_lang": "en",
        "freshness": freshness,
        "safesearch": "moderate"
    }
    
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    
    data = response.json()
    results = data.get("results", [])
    
    if not results:
        return ""
    
    news_str = ""
    for article in results[:limit]:
        title = article.get("title", "No title")
        description = article.get("description", "")
        pub_date = article.get("age", "")
        
        news_str += f"### {title}"
        if pub_date:
            news_str += f" ({pub_date})"
        news_str += f"\n\n{description}\n\n"
    
    return f"## Global News (Brave Search), from {start_date.strftime('%Y-%m-%d')} to {curr_date}:\n{news_str}"


