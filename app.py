"""
FastAPI Wrapper for Misinformation Detection System

This API provides endpoints for:
- Single claim verification
- Trending topic fetching
- Trending topic analysis
- Verification history
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import os
from openai import OpenAI
import feedparser
import requests
import json
import time
import re
from collections import defaultdict

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

# Initialize FastAPI app
app = FastAPI(
    title="Misinformation Detection API",
    description="AI-powered misinformation detection and verification system",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this based on your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY", "")
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable not set")

openai_client = OpenAI(api_key=openai_api_key)

# In-memory storage for history (use database in production)
verification_history_store = []

# ==================== PYDANTIC MODELS ====================

class ClaimVerificationRequest(BaseModel):
    claim: str = Field(..., description="Claim to verify", min_length=10)
    search_engines: List[str] = Field(
        default=["DuckDuckGo"],
        description="Search engines to use: DuckDuckGo, Serper API, SerpAPI"
    )
    min_sources: int = Field(default=5, ge=3, le=20, description="Minimum sources for verification")
    temperature: float = Field(default=0.1, ge=0.0, le=1.0, description="AI temperature")

class SourceInfo(BaseModel):
    source: str
    url: str

class VerificationResult(BaseModel):
    classification: str = Field(..., description="True, False, or Unverified")
    confidence: float = Field(..., description="Confidence score 0.0-1.0")
    reasoning: str = Field(..., description="Explanation of the verdict")
    sources_analyzed: int
    supporting_sources: List[SourceInfo] = []
    contradicting_sources: List[SourceInfo] = []
    support_score: Optional[float] = None
    contradict_score: Optional[float] = None

class VerificationResponse(BaseModel):
    claim: str
    timestamp: str
    verification: VerificationResult
    search_count: int

class TrendingItem(BaseModel):
    title: str
    source: str
    url: Optional[str] = ""
    published: Optional[str] = None
    engagement: Optional[int] = None
    views: Optional[int] = None

class TrendingResponse(BaseModel):
    google_trends: List[TrendingItem]
    google_news: List[TrendingItem]
    wikipedia: List[TrendingItem]
    twitter: List[TrendingItem]
    reddit: List[TrendingItem]
    hackernews: List[TrendingItem]
    bbc_news: List[TrendingItem]
    reuters: List[TrendingItem]
    cnn: List[TrendingItem]
    techcrunch: List[TrendingItem]
    aljazeera: List[TrendingItem]
    times_of_india: List[TrendingItem]
    ndtv: List[TrendingItem]
    thehindu: List[TrendingItem]

class AnalyzeTrendingRequest(BaseModel):
    topics: List[str] = Field(..., description="List of topics to analyze", min_items=1)
    search_engines: List[str] = Field(default=["DuckDuckGo"])
    min_sources: int = Field(default=5, ge=3, le=20)

class TrendAnalysisResult(BaseModel):
    topic: str
    claim: str
    verification: VerificationResult
    risk_score: int = Field(..., ge=0, le=100)
    scanned_at: str

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str

# ==================== CORE FUNCTIONS (from app.py) ====================

def get_api_key(key_name):
    """Try to get API key from environment"""
    return os.getenv(key_name, "")

def call_openai(prompt, system_message="You are a helpful assistant.", model="gpt-4o-mini", temp=0.1):
    """Call OpenAI Chat Completion with error handling"""
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=temp,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def search_duckduckgo(query):
    """Free DuckDuckGo search"""
    results = []
    
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            search_results = ddgs.text(query, max_results=10)
            for r in search_results:
                results.append({
                    'title': r.get('title', ''),
                    'snippet': r.get('body', ''),
                    'url': r.get('href', ''),
                    'source': 'DuckDuckGo'
                })
        if results:
            return results
    except Exception:
        pass
    
    try:
        import urllib.parse
        encoded_query = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json"
        
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if data.get('AbstractText'):
            results.append({
                'title': data.get('Heading', 'DuckDuckGo Answer'),
                'snippet': data.get('AbstractText'),
                'url': data.get('AbstractURL', ''),
                'source': 'DuckDuckGo'
            })
        
        for item in data.get('RelatedTopics', [])[:5]:
            if isinstance(item, dict) and 'Text' in item:
                results.append({
                    'title': item.get('Text', '')[:100],
                    'snippet': item.get('Text', ''),
                    'url': item.get('FirstURL', ''),
                    'source': 'DuckDuckGo'
                })
        
        if results:
            return results
    except Exception:
        pass
    
    return results

def search_serper(query):
    """Serper API search"""
    serper_api_key = get_api_key("SERPER_API_KEY")
    if not serper_api_key:
        return []
    
    try:
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query, "num": 10})
        headers = {
            'X-API-KEY': serper_api_key,
            'Content-Type': 'application/json'
        }
        response = requests.post(url, headers=headers, data=payload, timeout=10)
        data = response.json()
        
        results = []
        for item in data.get('organic', []):
            results.append({
                'title': item.get('title', ''),
                'snippet': item.get('snippet', ''),
                'url': item.get('link', ''),
                'source': 'Google (Serper)'
            })
        return results
    except Exception:
        return []

def search_serpapi(query):
    """SerpAPI search"""
    serpapi_key = get_api_key("SERPAPI_API_KEY")
    if not serpapi_key:
        return []
    
    try:
        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": serpapi_key,
            "num": 10
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        results = []
        for item in data.get('organic_results', []):
            results.append({
                'title': item.get('title'),
                'snippet': item.get('snippet'),
                'url': item.get('link'),
                'source': 'SerpAPI'
            })
        return results
    except Exception:
        return []

def multi_source_search(claim, engines):
    """Search across multiple engines"""
    all_results = []
    
    for engine in engines:
        if engine == "DuckDuckGo":
            results = search_duckduckgo(claim)
            all_results.extend(results)
        elif engine == "Serper API":
            results = search_serper(claim)
            all_results.extend(results)
        elif engine == "SerpAPI":
            results = search_serpapi(claim)
            all_results.extend(results)
    
    # If no results, use AI fallback
    if not all_results:
        ai_prompt = f"""You are a search engine. Find factual information about this claim:
        
CLAIM: {claim}

Generate 5 realistic search results that would help verify this claim. Include:
- Trusted news sources (Reuters, BBC, AP, etc.)
- Official sources (.gov, .edu)
- Fact-checking sites

Format as JSON array:
[
  {{
    "title": "Article title",
    "snippet": "Relevant excerpt about the claim (100-150 words)",
    "url": "https://example.com/article",
    "source": "Source name"
  }}
]

Only return valid JSON array."""
        
        response = call_openai(ai_prompt, "You are a helpful search assistant. Return only JSON.", temp=0.3)
        
        try:
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0].strip()
            elif '```' in response:
                response = response.split('```')[1].split('```')[0].strip()
            
            ai_results = json.loads(response)
            
            for r in ai_results:
                all_results.append({
                    'title': r.get('title', ''),
                    'snippet': r.get('snippet', ''),
                    'url': r.get('url', 'https://example.com'),
                    'source': 'AI Knowledge Base'
                })
        except Exception:
            pass
    
    # Remove duplicates by URL
    seen_urls = set()
    unique_results = []
    for r in all_results:
        url = r.get('url', '')
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(r)
    
    return unique_results

def extract_domain(url):
    """Extract domain from URL"""
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        return domain.replace('www.', '')
    except:
        return url

def assess_source_credibility(domain):
    """Rate source credibility"""
    trusted_domains = {
        'reuters.com': 0.95,
        'apnews.com': 0.95,
        'bbc.com': 0.90,
        'bbc.co.uk': 0.90,
        'nytimes.com': 0.85,
        'wsj.com': 0.85,
        'theguardian.com': 0.85,
        'npr.org': 0.85,
        'cnn.com': 0.75,
        'foxnews.com': 0.70,
        'wikipedia.org': 0.70,
        'gov': 0.90,
        'edu': 0.85,
    }
    
    for trusted, score in trusted_domains.items():
        if trusted in domain:
            return score
    
    if '.gov' in domain:
        return 0.90
    if '.edu' in domain:
        return 0.85
    
    return 0.50

def verify_claim_with_ai(claim, search_results, min_sources):
    """AI-powered verification"""
    if len(search_results) < min_sources:
        return {
            'classification': 'Insufficient Data',
            'confidence': 0.0,
            'reasoning': f'Only found {len(search_results)} sources. Need at least {min_sources} for reliable verification.',
            'sources_analyzed': len(search_results),
            'supporting_sources': [],
            'contradicting_sources': []
        }

    source_analyses = []

    for idx, result in enumerate(search_results[:15]):
        domain = extract_domain(result.get('url',''))
        credibility = assess_source_credibility(domain)

        prompt = (
            "Analyze if this search result supports, contradicts, or is neutral about the claim.\n\n"
            "CLAIM TO VERIFY: " + str(claim) + "\n\n"
            "SEARCH RESULT:\n"
            "Source: " + str(domain) + "\n"
            "Title: " + str(result.get('title','')) + "\n"
            "Content: " + str(result.get('snippet','')) + "\n\n"
            "YOUR TASK:\n"
            "1. Does this content SUPPORT the claim (provides evidence the claim is true)?\n"
            "2. Does this content CONTRADICT the claim (provides evidence the claim is false)?\n"
            "3. Is it NEUTRAL (mentions the topic but doesn't confirm or deny)?\n"
            "4. Is it UNCLEAR (not relevant enough)?\n\n"
            "IMPORTANT:\n"
            "- If the content confirms the claim happened, stance = \"supports\"\n"
            "- If the content denies the claim or says it didn't happen, stance = \"contradicts\"\n"
            "- If it just mentions related topics without confirming/denying, stance = \"neutral\"\n"
            "- Rate relevance HIGH (0.8-1.0) if directly about this claim\n\n"
            "Respond ONLY with valid JSON:\n"
            "{\n"
            "    \"stance\": \"supports\",\n"
            "    \"relevance\": 0.9,\n"
            "    \"key_facts\": \"Brief summary of what this source says about the claim\"\n"
            "}\n\n"
            "Return ONLY the JSON, nothing else."
        )

        response = call_openai(
            prompt,
            "You are a fact-checking expert. Be decisive - if content confirms a claim happened, mark it as 'supports'. If it denies it, mark as 'contradicts'. Return only JSON.",
            temp=0.1
        )

        try:
            response_clean = response.strip()
            
            if not response_clean or response_clean.lower().startswith("error:"):
                continue
            
            if '```json' in response_clean:
                response_clean = response_clean.split('```json', 1)[1].split('```', 1)[0].strip()
            elif '```' in response_clean:
                response_clean = response_clean.split('```', 1)[1].split('```', 1)[0].strip()
            
            if not response_clean:
                continue

            analysis = json.loads(response_clean)

            if not analysis.get('stance') or not isinstance(analysis.get('relevance'), (int, float)):
                continue

            analysis['source'] = domain
            analysis['credibility'] = credibility
            analysis['url'] = result.get('url','')
            analysis['title'] = result.get('title','')
            source_analyses.append(analysis)

        except (json.JSONDecodeError, Exception):
            continue

    if not source_analyses:
        return {
            'classification': 'Analysis Failed',
            'confidence': 0.0,
            'reasoning': 'Unable to analyze any search results.',
            'sources_analyzed': 0,
            'supporting_sources': [],
            'contradicting_sources': []
        }

    # Categorize evidence
    supporting = [s for s in source_analyses if s.get('stance') == 'supports' and s.get('relevance', 0) > 0.3]
    contradicting = [s for s in source_analyses if s.get('stance') == 'contradicts' and s.get('relevance', 0) > 0.3]
    neutral = [s for s in source_analyses if s.get('stance') in ['neutral', 'unclear']]

    # Weight by credibility
    support_score = sum(float(s['credibility']) * float(s.get('relevance', 0)) for s in supporting)
    contradict_score = sum(float(s['credibility']) * float(s.get('relevance', 0)) for s in contradicting)

    # Build evidence summary
    supporting_lines = []
    for s in supporting[:5]:
        supporting_lines.append("- [{}] {}".format(s.get('source',''), s.get('key_facts','N/A')))
    supporting_text = "\n".join(supporting_lines) if supporting_lines else "None found"

    contradicting_lines = []
    for s in contradicting[:5]:
        contradicting_lines.append("- [{}] {}".format(s.get('source',''), s.get('key_facts','N/A')))
    contradicting_text = "\n".join(contradicting_lines) if contradicting_lines else "None found"

    neutral_lines = []
    for s in neutral[:3]:
        neutral_lines.append("- [{}] {}".format(s.get('source',''), (s.get('key_facts','N/A') or "")[:100]))
    neutral_text = "\n".join(neutral_lines) if neutral_lines else "None"

    evidence_summary = (
        "TOTAL SOURCES ANALYZED: {}\n\n"
        "SUPPORTING EVIDENCE ({} sources, weighted score: {}):\n{}\n\n"
        "CONTRADICTING EVIDENCE ({} sources, weighted score: {}):\n{}\n\n"
        "NEUTRAL/UNCLEAR ({} sources):\n{}"
    ).format(
        len(source_analyses),
        len(supporting),
        format(support_score, '.2f'),
        supporting_text,
        len(contradicting),
        format(contradict_score, '.2f'),
        contradicting_text,
        len(neutral),
        neutral_text
    )

    final_prompt = (
        "You are a professional fact-checker. Based on the evidence from {} sources, determine if this claim is True, False, or Unverified.\n\n"
        "CLAIM: {}\n\n"
        "{}\n\n"
        "DECISION RULES:\n"
        "1. **True**: Multiple credible sources confirm it, minimal/no contradictions\n"
        "   - Confidence 0.7-0.9: Several sources confirm\n"
        "   - Confidence 0.9+: Many credible sources + official confirmation\n\n"
        "2. **False**: Sources contradict/debunk it, or claim clearly didn't happen\n"
        "   - Confidence 0.7-0.9: Several sources contradict\n"
        "   - Confidence 0.9+: Widely debunked + official denial\n\n"
        "3. **Unverified**: Conflicting evidence, only neutral mentions, or insufficient data\n"
        "   - Confidence 0.3-0.6: Mixed evidence or unclear\n"
        "   - Confidence 0.0-0.3: No real evidence either way\n\n"
        "ANALYZE:\n"
        "- Support score: {}\n"
        "- Contradict score: {}\n"
        "- Ratio: {}\n\n"
        "If support_score is significantly higher than contradict_score (>2x), lean toward True.\n"
        "If contradict_score is significantly higher than support_score (>2x), lean toward False.\n"
        "If scores are close or both low, mark as Unverified.\n\n"
        "Respond ONLY with valid JSON:\n"
        "{{\n"
        "    \"classification\": \"True/False/Unverified\",\n"
        "    \"confidence\": 0.0-1.0,\n"
        "    \"reasoning\": \"2-3 sentence explanation based on the evidence\"\n"
        "}}\n\n"
        "BE DECISIVE. If you have clear evidence, don't say Unverified. Return ONLY JSON."
    ).format(
        len(source_analyses),
        str(claim),
        evidence_summary,
        format(support_score, '.2f'),
        format(contradict_score, '.2f'),
        format((support_score / (contradict_score + 0.01)), '.2f')
    )

    response = call_openai(
        final_prompt,
        "You are an expert fact-checker. Be conservative with high confidence ratings. Return only JSON.",
        temp=0.05
    )

    try:
        response_clean = response.strip()
        
        if not response_clean or response_clean.lower().startswith("error:"):
            return {
                'classification': 'Analysis Error',
                'confidence': 0.0,
                'reasoning': 'AI returned an error for final verification',
                'sources_analyzed': len(source_analyses),
                'supporting_sources': [],
                'contradicting_sources': []
            }
        
        if '```json' in response_clean:
            response_clean = response_clean.split('```json', 1)[1].split('```', 1)[0].strip()
        elif '```' in response_clean:
            response_clean = response_clean.split('```', 1)[1].split('```', 1)[0].strip()
        
        if not response_clean:
            return {
                'classification': 'Analysis Error',
                'confidence': 0.0,
                'reasoning': 'No valid JSON content in AI response',
                'sources_analyzed': len(source_analyses),
                'supporting_sources': [],
                'contradicting_sources': []
            }

        result = json.loads(response_clean)

        if result.get('classification') not in ['True', 'False', 'Unverified']:
            result['classification'] = 'Unverified'

        if not isinstance(result.get('confidence'), (int, float)):
            result['confidence'] = 0.5

        # Force a decision if we have clear evidence
        if result['classification'] == 'Unverified':
            if support_score > 2.0 and support_score > contradict_score * 2:
                result['classification'] = 'True'
                result['confidence'] = min(0.85, 0.5 + support_score/10)
                result['reasoning'] = "Override: Strong supporting evidence from {} sources. ".format(len(supporting)) + result.get('reasoning','')
            elif contradict_score > 2.0 and contradict_score > support_score * 2:
                result['classification'] = 'False'
                result['confidence'] = min(0.85, 0.5 + contradict_score/10)
                result['reasoning'] = "Override: Strong contradicting evidence from {} sources. ".format(len(contradicting)) + result.get('reasoning','')

        # Add metadata
        result['sources_analyzed'] = len(source_analyses)
        result['supporting_sources'] = [{'source': s['source'], 'url': s['url']} for s in supporting]
        result['contradicting_sources'] = [{'source': s['source'], 'url': s['url']} for s in contradicting]
        result['support_score'] = support_score
        result['contradict_score'] = contradict_score

        return result

    except (json.JSONDecodeError, Exception) as e:
        return {
            'classification': 'Analysis Error',
            'confidence': 0.0,
            'reasoning': f'Error parsing final verdict: {str(e)}',
            'sources_analyzed': len(source_analyses),
            'supporting_sources': [],
            'contradicting_sources': []
        }

# ==================== TRENDING FUNCTIONS ====================

def fetch_google_trends(top_k=5, geo='IN'):
    """Return top_k results from Google Trends RSS"""
    trends = []
    try:
        feed = feedparser.parse(f'https://trends.google.com/trends/trendingsearches/daily/rss?geo={geo}')
        for entry in feed.entries[:top_k]:
            trends.append({
                'title': entry.title,
                'source': 'Google Trends',
                'url': entry.link,
                'published': entry.get('published', datetime.now().isoformat())
            })
    except Exception:
        pass
    return trends

def fetch_google_news(top_k=5, hl='en-IN', gl='IN', ceid='IN:en'):
    """Return top_k results from Google News RSS"""
    trends = []
    try:
        feed = feedparser.parse(f'https://news.google.com/rss?hl={hl}&gl={gl}&ceid={ceid}')
        for entry in feed.entries[:top_k]:
            trends.append({
                'title': entry.title,
                'source': 'Google News',
                'url': entry.link,
                'published': entry.get('published', datetime.now().isoformat())
            })
    except Exception:
        pass
    return trends

def fetch_wikipedia_trending(top_k=5):
    """Return top_k Wikipedia most-viewed articles"""
    trends = []
    try:
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y/%m/%d')
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/{yesterday}"
        response = requests.get(url, timeout=10)
        data = response.json()
        for article in data['items'][0]['articles'][:top_k]:
            if article['article'] not in ['Main_Page', 'Special:Search']:
                trends.append({
                    'title': article['article'].replace('_', ' '),
                    'source': 'Wikipedia Trending',
                    'url': f"https://en.wikipedia.org/wiki/{article['article']}",
                    'views': article.get('views', 0)
                })
    except Exception:
        pass
    return trends

def fetch_trends24(country_slug='india', top_k=5):
    """Scrape trends24.in for Twitter trends"""
    trends = []
    try:
        url = f'https://trends24.in/{country_slug}/'
        headers = {'User-Agent': 'MisinfoDetector/1.0'}
        resp = requests.get(url, headers=headers, timeout=10)
        html = resp.text
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            items = []
            for sec in soup.select('.trend-card')[:3]:
                for li in sec.select('ol.trend-list li')[:30]:
                    text = li.get_text(strip=True)
                    link = li.find('a')
                    href = link['href'] if link and link.get('href') else ''
                    items.append({'title': text, 'source': 'Twitter (Trends24)', 'url': href})
            seen = set()
            unique = []
            for it in items:
                if it['title'] not in seen:
                    seen.add(it['title'])
                    unique.append(it)
            return unique[:top_k]
        except Exception:
            matches = re.findall(r'>(#?[^<\n]{2,80})</a>', html)
            for m in matches[:top_k]:
                trends.append({'title': m.strip(), 'source': 'Twitter (Trends24)', 'url': ''})
            return trends
    except Exception:
        return trends

def fetch_reddit_top_from_subreddit(subreddit='all', limit=50, min_ups=20, top_k=5):
    """Fetch top_k posts from a subreddit"""
    trends = []
    try:
        s = subreddit.strip().lstrip('r/').strip()
        url = f'https://www.reddit.com/r/{s}/hot.json'
        headers = {'User-Agent': 'MisinfoDetector/1.0 (by /u/yourusername)'}
        params = {'limit': limit}
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        data = resp.json()
        for post in data.get('data', {}).get('children', [])[:limit]:
            p = post.get('data', {})
            ups = p.get('ups', 0) or 0
            if ups >= min_ups:
                trends.append({
                    'title': p.get('title', ''),
                    'source': f"r/{s}",
                    'url': f"https://reddit.com{p.get('permalink','')}",
                    'engagement': ups,
                    'published': datetime.fromtimestamp(p.get('created_utc', time.time())).isoformat()
                })
    except Exception:
        pass
    trends_sorted = sorted(trends, key=lambda x: x.get('engagement', 0), reverse=True)
    return trends_sorted[:top_k]

def fetch_hackernews(top_k=5):
    """Fetch top stories from Hacker News"""
    trends = []
    try:
        url = "https://hacker-news.firebaseio.com/v0/topstories.json"
        response = requests.get(url, timeout=10)
        story_ids = response.json()[:top_k * 2]
        
        for story_id in story_ids[:top_k]:
            try:
                story_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
                story_resp = requests.get(story_url, timeout=5)
                story = story_resp.json()
                if story and story.get('title'):
                    trends.append({
                        'title': story.get('title', ''),
                        'source': 'Hacker News',
                        'url': story.get('url') or f"https://news.ycombinator.com/item?id={story_id}",
                        'engagement': story.get('score', 0),
                        'published': datetime.fromtimestamp(story.get('time', time.time())).isoformat()
                    })
            except Exception:
                continue
    except Exception:
        pass
    return trends[:top_k]

def fetch_bbc_news(top_k=5):
    """Fetch from BBC News RSS"""
    trends = []
    try:
        feed = feedparser.parse('http://feeds.bbci.co.uk/news/rss.xml')
        for entry in feed.entries[:top_k]:
            trends.append({
                'title': entry.title,
                'source': 'BBC News',
                'url': entry.link,
                'published': entry.get('published', datetime.now().isoformat())
            })
    except Exception:
        pass
    return trends

def fetch_reuters_news(top_k=5):
    """Fetch from Reuters RSS"""
    trends = []
    try:
        feed = feedparser.parse('https://www.reuters.com/rssFeed/worldNews')
        for entry in feed.entries[:top_k]:
            trends.append({
                'title': entry.title,
                'source': 'Reuters',
                'url': entry.link,
                'published': entry.get('published', datetime.now().isoformat())
            })
    except Exception:
        pass
    return trends

def fetch_cnn_news(top_k=5):
    """Fetch from CNN RSS"""
    trends = []
    try:
        feed = feedparser.parse('http://rss.cnn.com/rss/edition.rss')
        for entry in feed.entries[:top_k]:
            trends.append({
                'title': entry.title,
                'source': 'CNN',
                'url': entry.link,
                'published': entry.get('published', datetime.now().isoformat())
            })
    except Exception:
        pass
    return trends

def fetch_techcrunch(top_k=5):
    """Fetch from TechCrunch RSS"""
    trends = []
    try:
        feed = feedparser.parse('https://techcrunch.com/feed/')
        for entry in feed.entries[:top_k]:
            trends.append({
                'title': entry.title,
                'source': 'TechCrunch',
                'url': entry.link,
                'published': entry.get('published', datetime.now().isoformat())
            })
    except Exception:
        pass
    return trends

def fetch_aljazeera_news(top_k=5):
    """Fetch from Al Jazeera RSS"""
    trends = []
    try:
        feed = feedparser.parse('https://www.aljazeera.com/xml/rss/all.xml')
        for entry in feed.entries[:top_k]:
            trends.append({
                'title': entry.title,
                'source': 'Al Jazeera',
                'url': entry.link,
                'published': entry.get('published', datetime.now().isoformat())
            })
    except Exception:
        pass
    return trends

def fetch_times_of_india(top_k=5):
    """Fetch from Times of India RSS"""
    trends = []
    try:
        feed = feedparser.parse('https://timesofindia.indiatimes.com/rssfeedstopstories.cms')
        for entry in feed.entries[:top_k]:
            trends.append({
                'title': entry.title,
                'source': 'Times of India',
                'url': entry.link,
                'published': entry.get('published', datetime.now().isoformat())
            })
    except Exception:
        pass
    return trends

def fetch_ndtv_news(top_k=5):
    """Fetch from NDTV RSS"""
    trends = []
    try:
        feed = feedparser.parse('https://feeds.feedburner.com/ndtvnews-top-stories')
        for entry in feed.entries[:top_k]:
            trends.append({
                'title': entry.title,
                'source': 'NDTV',
                'url': entry.link,
                'published': entry.get('published', datetime.now().isoformat())
            })
    except Exception:
        pass
    return trends

def fetch_thehindu_news(top_k=5):
    """Fetch from The Hindu RSS"""
    trends = []
    try:
        feed = feedparser.parse('https://www.thehindu.com/news/national/feeder/default.rss')
        for entry in feed.entries[:top_k]:
            trends.append({
                'title': entry.title,
                'source': 'The Hindu',
                'url': entry.link,
                'published': entry.get('published', datetime.now().isoformat())
            })
    except Exception:
        pass
    return trends

def calculate_risk_score(verification, base_risk=30):
    """Calculate risk score (0-100)"""
    risk = base_risk
    
    cls = verification.get('classification', 'Unverified')
    confidence = float(verification.get('confidence', 0.0)) if isinstance(verification.get('confidence', 0.0), (int, float)) else 0.0
    support_score = float(verification.get('support_score', 0.0)) if verification.get('support_score') else 0.0
    contradict_score = float(verification.get('contradict_score', 0.0)) if verification.get('contradict_score') else 0.0
    
    if cls == 'True':
        risk += 10 + int(confidence * 10)
    elif cls == 'False':
        risk -= 20 + int(confidence * 10)
    else:
        risk += 5 + int((support_score + contradict_score) * 2)
    
    risk = max(0, min(100, int(risk)))
    return risk

# ==================== API ENDPOINTS ====================

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with API info"""
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/verify-claim", response_model=VerificationResponse)
async def verify_claim(request: ClaimVerificationRequest):
    """
    Verify a single claim using AI-powered analysis
    
    - **claim**: The claim to verify (minimum 10 characters)
    - **search_engines**: List of search engines to use (DuckDuckGo, Serper API, SerpAPI)
    - **min_sources**: Minimum number of sources required for verification (3-20)
    - **temperature**: AI temperature for analysis (0.0-1.0)
    """
    try:
        # Validate search engines
        valid_engines = {"DuckDuckGo", "Serper API", "SerpAPI"}
        if not all(engine in valid_engines for engine in request.search_engines):
            raise HTTPException(status_code=400, detail="Invalid search engine specified")
        
        # Search for evidence
        search_results = multi_source_search(request.claim, request.search_engines)
        
        if not search_results:
            raise HTTPException(status_code=404, detail="No search results found for the claim")
        
        # Verify claim
        verification = verify_claim_with_ai(request.claim, search_results, request.min_sources)
        
        # Create response
        response = {
            "claim": request.claim,
            "timestamp": datetime.now().isoformat(),
            "verification": verification,
            "search_count": len(search_results)
        }
        
        # Store in history
        verification_history_store.insert(0, response)
        if len(verification_history_store) > 100:
            verification_history_store.pop()
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")

@app.get("/api/trending", response_model=TrendingResponse)
async def get_trending(
    top_k: int = Query(default=5, ge=1, le=20, description="Number of items per platform"),
    include_twitter: bool = Query(default=True, description="Include Twitter trends"),
    reddit_subreddits: Optional[str] = Query(default="all", description="Comma-separated subreddits"),
    reddit_min_ups: int = Query(default=50, ge=0, description="Minimum Reddit upvotes")
):
    """
    Fetch trending topics from multiple platforms
    
    - **top_k**: Number of top items to fetch per platform (1-20)
    - **include_twitter**: Whether to include Twitter trends
    - **reddit_subreddits**: Comma-separated list of subreddits to monitor
    - **reddit_min_ups**: Minimum upvotes for Reddit posts
    """
    try:
        # Fetch from all platforms
        results = {
            'google_trends': fetch_google_trends(top_k=top_k),
            'google_news': fetch_google_news(top_k=top_k),
            'wikipedia': fetch_wikipedia_trending(top_k=top_k),
            'twitter': fetch_trends24('india', top_k=top_k) if include_twitter else [],
            'hackernews': fetch_hackernews(top_k=top_k),
            'bbc_news': fetch_bbc_news(top_k=top_k),
            'reuters': fetch_reuters_news(top_k=top_k),
            'cnn': fetch_cnn_news(top_k=top_k),
            'techcrunch': fetch_techcrunch(top_k=top_k),
            'aljazeera': fetch_aljazeera_news(top_k=top_k),
            'times_of_india': fetch_times_of_india(top_k=top_k),
            'ndtv': fetch_ndtv_news(top_k=top_k),
            'thehindu': fetch_thehindu_news(top_k=top_k)
        }
        
        # Fetch Reddit
        reddit_items = []
        if reddit_subreddits:
            subs = [s.strip() for s in reddit_subreddits.split(',') if s.strip()]
            for s in subs:
                reddit_items.extend(fetch_reddit_top_from_subreddit(s, limit=50, min_ups=reddit_min_ups, top_k=top_k))
        else:
            reddit_items = fetch_reddit_top_from_subreddit('all', limit=50, min_ups=max(100, reddit_min_ups), top_k=top_k)
        
        # Dedupe Reddit
        seen = set()
        unique_reddit = []
        for it in sorted(reddit_items, key=lambda x: x.get('engagement', 0), reverse=True):
            u = it.get('url')
            if u and u not in seen:
                seen.add(u)
                unique_reddit.append(it)
            if len(unique_reddit) >= top_k:
                break
        results['reddit'] = unique_reddit
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch trending topics: {str(e)}")

@app.post("/api/analyze-trending", response_model=List[TrendAnalysisResult])
async def analyze_trending(request: AnalyzeTrendingRequest):
    """
    Analyze specific trending topics for misinformation
    
    - **topics**: List of topic titles to analyze
    - **search_engines**: Search engines to use for verification
    - **min_sources**: Minimum sources for verification
    """
    try:
        if not request.topics:
            raise HTTPException(status_code=400, detail="No topics provided")
        
        results = []
        
        for topic in request.topics:
            # Search and verify
            search_results = multi_source_search(topic, request.search_engines)
            verification = verify_claim_with_ai(topic, search_results, request.min_sources)
            
            # Calculate risk
            risk_score = calculate_risk_score(verification)
            
            result = {
                'topic': topic,
                'claim': topic,
                'verification': verification,
                'risk_score': risk_score,
                'scanned_at': datetime.now().isoformat()
            }
            results.append(result)
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/history", response_model=List[VerificationResponse])
async def get_verification_history(
    limit: int = Query(default=10, ge=1, le=100, description="Number of records to return")
):
    """
    Get verification history
    
    - **limit**: Maximum number of records to return (1-100)
    """
    return verification_history_store[:limit]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
