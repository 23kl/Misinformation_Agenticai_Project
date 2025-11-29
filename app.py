import streamlit as st
from openai import OpenAI
import feedparser
import requests
from datetime import datetime, timedelta
import json
import time
import os
import re
from collections import defaultdict

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

# Configure page
st.set_page_config(page_title="Misinformation Detection Agent", layout="wide")
st.title("🔍 Misinformation Detection Agent")
st.markdown("**Verify claims with specialised web search and ai analysis**")
st.markdown("*An Agentic AI Product*")

# ----------------- API Key Management -----------------
def get_api_key(key_name):
    """Try to get API key from multiple sources"""
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except:
        pass
    return os.getenv(key_name, "")

openai_api_key = get_api_key("OPENAI_API_KEY")

if not openai_api_key:
    st.error("❌ OPENAI_API_KEY not found! Add it to .env file or .streamlit/secrets.toml")
    st.stop()

# Initialize OpenAI client (SDK 1.0.0+)
openai_client = OpenAI(api_key=openai_api_key)

# ----------------- Sidebar / Settings -----------------
with st.sidebar:
    st.success("✅ API Key Loaded")
    
    # Mode selection
    st.markdown("### 🎯 Operation Mode")
    mode = st.radio(
        "Select Mode:",
        ["Single Claim Verification", "Trending Misinformation Monitor"],
        help="Single: Verify one claim | Monitor: Auto-detect trending misinformation"
    )
    
    st.divider()
    st.markdown("### Search Configuration")
    
    search_engines = st.multiselect(
        "Active Search Engines:",
        ["DuckDuckGo", "Serper API", "SerpAPI"],
        default=["DuckDuckGo"],
        help="Using multiple search engines improves accuracy"
    )
    
    min_sources = st.slider("Minimum sources for verification:", 3, 10, 5)
    
    st.divider()
    st.markdown("### Model Settings")
    temperature = st.slider("AI Temperature:", 0.0, 1.0, 0.1, help="Lower = more factual")
    
    st.divider()
    st.markdown("### Trending / Platform Settings")
    top_k = st.slider("Top items per platform (K):", 3, 10, 5, help="How many top items to fetch per platform")
    reddit_subs_input = st.text_input("Subreddits to monitor (comma-separated)", value="mumbai")
    reddit_min_ups = st.number_input("Min Reddit ups (filter)", min_value=0, max_value=100000, value=50, step=10)
    include_twitter = st.checkbox("Include Twitter (trends24) — India", value=True)
    
    if mode == "Trending Misinformation Monitor":
        st.divider()
        st.markdown("### 📊 Monitor Settings")
        refresh_interval = st.slider("Refresh interval (minutes):", 5, 60, 15)
        max_trends = st.slider("Max trends to analyze:", 5, 20, 10)

# ----------------- Session state -----------------
if 'verification_history' not in st.session_state:
    st.session_state.verification_history = []

if 'trending_reports' not in st.session_state:
    st.session_state.trending_reports = []

if 'last_scan_time' not in st.session_state:
    st.session_state.last_scan_time = None

# ----------------- OpenAI wrapper -----------------
def call_openai(prompt, system_message="You are a helpful assistant.", model="gpt-4o-mini", temp=0.1):
    """Call OpenAI Chat Completion with error handling (SDK 1.0.0+)"""
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

# ----------------- Search functions -----------------
def search_duckduckgo(query):
    """Free DuckDuckGo search with multiple fallback methods"""
    results = []
    
    # Method 1: Try duckduckgo_search library
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
    
    # Method 2: Try alternative DuckDuckGo API
    try:
        import urllib.parse
        encoded_query = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json"
        
        response = requests.get(url, timeout=10)
        data = response.json()
        
        # Parse DuckDuckGo instant answer API
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

def multi_source_search(claim, engines, silent=False):
    """Search across multiple engines with guaranteed results"""
    all_results = []
    
    if not silent:
        st.info("🔍 Searching... This may take 10-20 seconds")
    
    for engine in engines:
        if not silent:
            with st.spinner(f"Searching {engine}..."):
                if engine == "DuckDuckGo":
                    results = search_duckduckgo(claim)
                    all_results.extend(results)
                    st.success(f"✅ {engine}: Found {len(results)} results")
                elif engine == "Serper API":
                    results = search_serper(claim)
                    all_results.extend(results)
                    if results:
                        st.success(f"✅ {engine}: Found {len(results)} results")
                elif engine == "SerpAPI":
                    results = search_serpapi(claim)
                    all_results.extend(results)
                    if results:
                        st.success(f"✅ {engine}: Found {len(results)} results")
        else:
            if engine == "DuckDuckGo":
                results = search_duckduckgo(claim)
                all_results.extend(results)
            elif engine == "Serper API":
                results = search_serper(claim)
                all_results.extend(results)
            elif engine == "SerpAPI":
                results = search_serpapi(claim)
                all_results.extend(results)
    
    # If no results from any engine, create synthetic search using OpenAI
    if not all_results:
        if not silent:
            st.warning("⚠️ No search results from APIs. Using AI knowledge base as fallback...")
        
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
            
            if not silent:
                st.info(f"ℹ️ Generated {len(all_results)} results from AI knowledge base")
        except Exception:
            if not silent:
                st.error("❌ Unable to find any information about this claim. Try rephrasing.")
            return []
    
    # Remove duplicates by URL
    seen_urls = set()
    unique_results = []
    for r in all_results:
        url = r.get('url', '')
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(r)
    
    return unique_results

# ----------------- Utilities -----------------
def extract_domain(url):
    """Extract domain from URL for source credibility"""
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

# ----------------- Verification with AI -----------------
def verify_claim_with_ai(claim, search_results, min_sources, silent=False):
    """Deep AI-powered verification with multiple analysis steps"""
    if len(search_results) < min_sources:
        return {
            'classification': 'Insufficient Data',
            'confidence': 0.0,
            'reasoning': 'Only found {} sources. Need at least {} for reliable verification.'.format(len(search_results), min_sources),
            'sources_analyzed': len(search_results),
            'supporting_sources': [],
            'contradicting_sources': []
        }

    # Step 1: Analyze each source individually
    source_analyses = []

    if not silent:
        progress_placeholder = st.empty()

    for idx, result in enumerate(search_results[:15]):
        if not silent:
            progress_placeholder.info("🔍 Analyzing source {}/{}: {}".format(idx+1, min(15, len(search_results)), (result.get('title',''))[:50] + "..."))

        domain = extract_domain(result.get('url',''))
        credibility = assess_source_credibility(domain)

        # Build prompt by concatenation to avoid any f-string brace parsing
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
            
            # Check if response is empty or an error message
            if not response_clean:
                if not silent:
                    st.caption(f"⚠️ Empty response from AI for {domain}; skipping.")
                continue
            
            if response_clean.lower().startswith("error:"):
                if not silent:
                    st.caption(f"⚠️ AI error for {domain}: {response_clean[:100]}; skipping.")
                continue
            
            # Clean markdown code fences
            if '```json' in response_clean:
                response_clean = response_clean.split('```json', 1)[1].split('```', 1)[0].strip()
            elif '```' in response_clean:
                response_clean = response_clean.split('```', 1)[1].split('```', 1)[0].strip()
            
            # Ensure we have something to parse
            if not response_clean:
                if not silent:
                    st.caption(f"⚠️ No valid JSON content after cleaning for {domain}; skipping.")
                continue

            analysis = json.loads(response_clean)

            # Validate expected fields
            if not analysis.get('stance') or not isinstance(analysis.get('relevance'), (int, float)):
                # skip invalid analysis
                if not silent:
                    st.caption("⚠️ Source analysis returned invalid structure; skipping.")
                continue

            analysis['source'] = domain
            analysis['credibility'] = credibility
            analysis['url'] = result.get('url','')
            analysis['title'] = result.get('title','')
            source_analyses.append(analysis)

            if not silent:
                st.caption("✓ {}: {} (relevance: {:.2f})".format(domain, analysis['stance'], float(analysis['relevance'])))

        except json.JSONDecodeError as e:
            if not silent:
                st.caption(f"⚠ Failed to parse JSON from {domain}: {str(e)}")
                with st.expander(f"⚠️ Debug: Raw response from {domain} (click to view)"):
                    st.code(response[:500] if response else "Empty response")
            continue
        except Exception as e:
            if not silent:
                st.caption(f"⚠ Failed to analyze {domain}: {str(e)}")
            continue

    if not silent:
        try:
            progress_placeholder.empty()
        except:
            pass

    if not source_analyses:
        return {
            'classification': 'Analysis Failed',
            'confidence': 0.0,
            'reasoning': 'Unable to analyze any search results. All sources failed to parse.',
            'sources_analyzed': 0,
            'supporting_sources': [],
            'contradicting_sources': []
        }

    # Step 2: Categorize evidence
    supporting = [s for s in source_analyses if s.get('stance') == 'supports' and s.get('relevance', 0) > 0.3]
    contradicting = [s for s in source_analyses if s.get('stance') == 'contradicts' and s.get('relevance', 0) > 0.3]
    neutral = [s for s in source_analyses if s.get('stance') in ['neutral', 'unclear']]

    if not silent:
        st.info("📊 Evidence Breakdown:\n- Supporting: {} sources\n- Contradicting: {} sources\n- Neutral/Unclear: {} sources".format(len(supporting), len(contradicting), len(neutral)))

    # Weight by credibility
    support_score = sum(float(s['credibility']) * float(s.get('relevance', 0)) for s in supporting)
    contradict_score = sum(float(s['credibility']) * float(s.get('relevance', 0)) for s in contradicting)

    # Step 3: Final AI verdict
    # Build evidence_summary carefully without f-strings
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
        
        # Check if response is empty or an error message
        if not response_clean:
            if not silent:
                st.error("⚠️ Empty response from AI for final verification")
            return {
                'classification': 'Analysis Error',
                'confidence': 0.0,
                'reasoning': 'AI returned an empty response for final verification',
                'sources_analyzed': len(source_analyses),
                'supporting_sources': [],
                'contradicting_sources': []
            }
        
        if response_clean.lower().startswith("error:"):
            if not silent:
                st.error(f"⚠️ AI error in final verification: {response_clean[:200]}")
            return {
                'classification': 'Analysis Error',
                'confidence': 0.0,
                'reasoning': f'AI error: {response_clean[:200]}',
                'sources_analyzed': len(source_analyses),
                'supporting_sources': [],
                'contradicting_sources': []
            }
        
        # Clean markdown code fences
        if '```json' in response_clean:
            response_clean = response_clean.split('```json', 1)[1].split('```', 1)[0].strip()
        elif '```' in response_clean:
            response_clean = response_clean.split('```', 1)[1].split('```', 1)[0].strip()
        
        # Final check after cleaning
        if not response_clean:
            if not silent:
                st.error("⚠️ No valid JSON content after cleaning final verdict")
            return {
                'classification': 'Analysis Error',
                'confidence': 0.0,
                'reasoning': 'No valid JSON content in AI response after cleaning',
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

    except json.JSONDecodeError as e:
        error_msg = f'JSON parsing error in final verdict: {str(e)}'
        if not silent:
            st.error(f"❌ {error_msg}")
            with st.expander("🔍 Debug: Raw AI response (click to view)"):
                st.code(response[:1000] if response else "Empty response")
        return {
            'classification': 'Analysis Error',
            'confidence': 0.0,
            'reasoning': error_msg,
            'sources_analyzed': len(source_analyses),
            'supporting_sources': [],
            'contradicting_sources': []
        }
    except Exception as e:
        error_msg = f'Error parsing final verdict: {str(e)}'
        if not silent:
            st.error(f"❌ {error_msg}")
        return {
            'classification': 'Analysis Error',
            'confidence': 0.0,
            'reasoning': error_msg,
            'sources_analyzed': len(source_analyses),
            'supporting_sources': [],
            'contradicting_sources': []
        }


# ----------------- Robust JSON parsing helper -----------------
def safe_parse_json(raw_text):
    """
    Try several heuristics to extract JSON from a model output string.
    Returns Python object on success, or None on failure.
    """
    if not raw_text or not isinstance(raw_text, str):
        return None

    text = raw_text.strip()

    # 1) direct json
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) markdown fenced code block (```json ... ``` or ``` ... ```) 
    for fence in ('```json', '```'):
        if fence in text:
            try:
                body = text.split(fence, 1)[1]
                if '```' in body:
                    body = body.split('```', 1)[0]
                body = body.strip()
                return json.loads(body)
            except Exception:
                continue

    # 3) first {...} or [...] substring via regex
    try:
        m = re.search(r'(\[.*\]|\{.*\})', text, flags=re.S)
        if m:
            candidate = m.group(1)
            return json.loads(candidate)
    except Exception:
        pass

    return None

# ----------------- TREND / PLATFORM FETCHERS (Top-K) -----------------
def fetch_google_trends(top_k=5, geo='IN'):
    """Return top_k results from Google Trends RSS (country via geo)."""
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
    """Return top_k results from Google News RSS for India by default."""
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
    """Return top_k Wikipedia most-viewed articles (English)."""
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
    """Scrape trends24.in to return top_k Twitter trending items for a country."""
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
            # dedupe preserving order
            seen = set(); unique = []
            for it in items:
                if it['title'] not in seen:
                    seen.add(it['title']); unique.append(it)
            return unique[:top_k]
        except Exception:
            # fallback regex parse
            matches = re.findall(r'>(#?[^<\n]{2,80})</a>', html)
            for m in matches[:top_k]:
                trends.append({'title': m.strip(), 'source': 'Twitter (Trends24)', 'url': ''})
            return trends
    except Exception:
        return trends

def fetch_reddit_top_from_subreddit(subreddit='all', limit=50, min_ups=20, top_k=5):
    """
    Fetch top_k posts from a given subreddit (hot). Uses public reddit JSON endpoint.
    - subreddit: 'mumbai' or 'all' etc.
    - min_ups: filter low-engagement posts (set low for small subs)
    """
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
    # sort by ups and return top_k
    trends_sorted = sorted(trends, key=lambda x: x.get('engagement', 0), reverse=True)
    return trends_sorted[:top_k]

def fetch_hackernews(top_k=5):
    """Fetch top stories from Hacker News API."""
    trends = []
    try:
        url = "https://hacker-news.firebaseio.com/v0/topstories.json"
        response = requests.get(url, timeout=10)
        story_ids = response.json()[:top_k * 2]  # Fetch extra in case some fail
        
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
    """Fetch from BBC News RSS feed."""
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
    """Fetch from Reuters RSS feed."""
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
    """Fetch from CNN RSS feed."""
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
    """Fetch from TechCrunch RSS feed."""
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
    """Fetch from Al Jazeera RSS feed."""
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
    """Fetch from Times of India RSS feed."""
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
    """Fetch from NDTV RSS feed."""
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
    """Fetch from The Hindu RSS feed."""
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

def fetch_top_per_platform(top_k=5, include_twitter=True, reddit_subreddits=None, reddit_min_ups=20):
    """
    Returns dict with top_k lists per platform.
    """
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
    reddit_items = []
    if reddit_subreddits:
        for s in reddit_subreddits:
            reddit_items.extend(fetch_reddit_top_from_subreddit(s, limit=50, min_ups=reddit_min_ups, top_k=top_k))
    else:
        reddit_items = fetch_reddit_top_from_subreddit('all', limit=50, min_ups=max(100, reddit_min_ups), top_k=top_k)
    # dedupe reddit by url and keep top_k
    seen = set(); unique_reddit = []
    for it in sorted(reddit_items, key=lambda x: x.get('engagement', 0), reverse=True):
        u = it.get('url')
        if u and u not in seen:
            seen.add(u); unique_reddit.append(it)
        if len(unique_reddit) >= top_k:
            break
    results['reddit'] = unique_reddit
    return results

# ----------------- Clustering (robust) -----------------
def cluster_misinformation(trends):
    """Identify misinformation clusters using AI (robust to non-JSON responses)."""
    st.info("🧠 Analyzing trends for potential misinformation...")

    if not trends:
        st.warning("No trends provided to analyze.")
        return []

    trends_text = "\n".join([f"- {t.get('title','')} (from {t.get('source','')})" for t in trends])

    prompt = f"""Analyze these trending topics and identify which ones are most likely to contain misinformation, conspiracy theories, or false claims.

TRENDING TOPICS:
{trends_text}

TASK:
1. Identify topics with high misinformation risk
2. Extract specific verifiable claims from each
3. Rate risk level (HIGH/MEDIUM/LOW)

Respond with a JSON array like:
[
  {{
    "topic": "Original topic title",
    "claim": "Specific verifiable claim extracted",
    "risk_level": "HIGH/MEDIUM/LOW",
    "reason": "Why this might be misinformation"
  }}
]

IMPORTANT:
- Return ONLY valid JSON (an array). Do not include any extra commentary.
- If you cannot find any high-risk topics, return an empty array: []
"""

    raw_response = call_openai(
        prompt,
        "You are a misinformation detection expert. Identify high-risk claims. Return only JSON.",
        temp=0.2
    )

    # If model errored or returned nothing
    if not raw_response or (isinstance(raw_response, str) and raw_response.lower().startswith("error:")):
        st.error("⚠️ Model returned an error or empty response while clustering. See model output below.")
        st.write(raw_response)
        return []

    # Try robust parsing
    parsed = safe_parse_json(raw_response)

    if parsed is None:
        st.warning("⚠️ Unable to parse model JSON on first attempt. Showing raw model output for debugging.")
        with st.expander("Raw model output (clustering)"):
            st.code(raw_response[:4000])

        # Fallback: ask model to return only an array of topic strings
        fallback_prompt = f"""You will be given a list of trending topics. Return ONLY a JSON array of topic titles (strings) that look likely to contain misinformation.
Return NOTHING else.

TRENDING TOPICS:
{trends_text}

Example output:
["Topic 1", "Topic 2"]

Return an empty array [] if none."""
        fallback_raw = call_openai(
            fallback_prompt,
            "You are a strict assistant. Return only JSON array of topic titles.",
            temp=0.15
        )

        fallback_parsed = safe_parse_json(fallback_raw)
        if fallback_parsed:
            clusters = []
            for t in fallback_parsed:
                clusters.append({
                    'topic': t,
                    'claim': t,
                    'risk_level': 'MEDIUM',
                    'reason': 'Selected by fallback clustering',
                    'trend_sources': [x for x in trends if t.lower() in x.get('title','').lower()][:5]
                })
            st.info(f"✅ Fallback clustering produced {len(clusters)} clusters.")
            return clusters
        else:
            st.error("❌ Fallback also failed to produce valid JSON. Returning empty cluster list.")
            return []

    # Normalize parsed content into list of cluster dicts
    if isinstance(parsed, dict):
        # If model returned {'clusters': [...]} or similar
        if 'clusters' in parsed and isinstance(parsed['clusters'], list):
            parsed = parsed['clusters']
        else:
            # try to find first list value inside
            extracted = None
            for v in parsed.values():
                if isinstance(v, list):
                    extracted = v; break
            if extracted is None:
                st.error("Unexpected JSON structure from model; returning empty.")
                return []
            parsed = extracted

    if not isinstance(parsed, list):
        st.error("Parsed model output is not a list. Returning empty clusters.")
        return []

    clusters = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        topic = item.get('topic') or item.get('claim') or item.get('title')
        claim = item.get('claim') or topic
        risk = item.get('risk_level') or item.get('risk') or 'MEDIUM'
        reason = item.get('reason') or item.get('why') or ''
        matching_trends = [t for t in trends if topic and (topic.lower() in t.get('title','').lower() or t.get('title','').lower() in str(topic).lower())][:5] if topic else []
        clusters.append({
            'topic': topic or claim,
            'claim': claim,
            'risk_level': risk,
            'reason': reason,
            'trend_sources': matching_trends
        })

    st.success(f"✅ Identified {len(clusters)} potential misinformation clusters")
    return clusters

# ----------------- Evidence timeline & risk score -----------------
def build_evidence_timeline(claim, verification):
    """Build timeline of evidence"""
    timeline = []
    
    # Add supporting evidence
    for source in verification.get('supporting_sources', []):
        timeline.append({
            'timestamp': datetime.now().isoformat(),
            'event': 'Supporting Evidence Found',
            'source': source.get('source', ''),
            'url': source.get('url', ''),
            'type': 'support'
        })
    
    # Add contradicting evidence
    for source in verification.get('contradicting_sources', []):
        timeline.append({
            'timestamp': datetime.now().isoformat(),
            'event': 'Contradicting Evidence Found',
            'source': source.get('source', ''),
            'url': source.get('url', ''),
            'type': 'contradict'
        })
    
    # Add final decision
    timeline.append({
        'timestamp': datetime.now().isoformat(),
        'event': f"Final classification: {verification.get('classification','Unknown')}",
        'source': 'Agent',
        'url': '',
        'type': 'decision'
    })
    
    return timeline

def calculate_risk_score(cluster, verification):
    """Calculate risk score (0-100)"""
    risk = 0
    
    # Base risk from cluster analysis
    if cluster.get('risk_level') == 'HIGH':
        risk += 40
    elif cluster.get('risk_level') == 'MEDIUM':
        risk += 25
    else:
        risk += 10
    
    # Add risk based on trend virality (sum engagement/views if available)
    virality = 0
    for t in cluster.get('trend_sources', []):
        # use 'engagement' or 'views' fields when available
        if 'engagement' in t:
            virality += int(t.get('engagement', 0)) / 1000.0  # scale down
        elif 'views' in t:
            virality += int(t.get('views', 0)) / 1000.0
        else:
            # small bump if present in any source
            virality += 1.0
    # Normalize virality to a 0-30 scale (rough)
    virality_score = min(30, virality * 2)
    risk += virality_score
    
    # Adjust based on verification outcome
    cls = verification.get('classification', 'Unverified')
    confidence = float(verification.get('confidence', 0.0)) if isinstance(verification.get('confidence', 0.0), (int, float)) else 0.0
    support_score = float(verification.get('support_score', 0.0)) if verification.get('support_score') else 0.0
    contradict_score = float(verification.get('contradict_score', 0.0)) if verification.get('contradict_score') else 0.0
    
    # If verified TRUE with high confidence → risk is high (it's true but may be harmful)
    if cls == 'True':
        risk += 10 + int(confidence * 10)
    # If verified FALSE with high confidence → risk decreases (debunked)
    elif cls == 'False':
        # lower the risk depending on how strongly debunked
        risk -= 20 + int(confidence * 10)
    else:
        # Unverified — increase risk modestly
        risk += 5 + int((support_score + contradict_score) * 2)
    
    # Clamp between 0 and 100
    risk = max(0, min(100, int(risk)))
    return risk

# ----------------- Run trending scan -----------------
def run_trending_scan(max_trends=10, selected_topics=None):
    """Run a full trending scan and verification pass"""
    trends = fetch_top_per_platform(top_k=max_trends, include_twitter=include_twitter, reddit_subreddits=[s.strip() for s in reddit_subs_input.split(',') if s.strip()], reddit_min_ups=reddit_min_ups)
    # if selected topics passed in (strings), wrap them into clusters
    clusters = []
    if selected_topics:
        for t in selected_topics:
            clusters.append({
                'topic': t,
                'claim': t,
                'risk_level': 'MEDIUM',
                'reason': 'User-selected trending topic',
                'trend_sources': []
            })
    else:
        # flatten all platform items into list for clustering
        flattened = []
        for k, v in trends.items():
            flattened.extend(v if isinstance(v, list) else [])
        clusters = cluster_misinformation(flattened)
    
    reports = []
    for c in clusters:
        claim = c.get('claim') or c.get('topic')
        # perform web search
        search_results = multi_source_search(claim, search_engines, silent=True)
        # verify claim
        verification = verify_claim_with_ai(claim, search_results, min_sources, silent=True)
        # build timeline
        timeline = build_evidence_timeline(claim, verification)
        # risk score
        risk = calculate_risk_score(c, verification)
        
        report = {
            'topic': c.get('topic'),
            'claim': claim,
            'risk_level': c.get('risk_level'),
            'reason': c.get('reason'),
            'trend_sources': c.get('trend_sources', []),
            'verification': verification,
            'timeline': timeline,
            'risk_score': risk,
            'scanned_at': datetime.now().isoformat()
        }
        reports.append(report)
    
    # store in session
    st.session_state.trending_reports.insert(0, {'scanned_at': datetime.now().isoformat(), 'reports': reports})
    st.session_state.last_scan_time = datetime.now().isoformat()
    return reports

# ----------------- UI FLOWS -----------------
if mode == "Single Claim Verification":
    st.header("🔎 Single Claim Verification")
    claim_input = st.text_area("Enter claim to verify:", height=120, placeholder="Example: The Eiffel Tower was completed in 1889")
    if st.button("🔍 Verify Claim") and claim_input.strip():
        claim = claim_input.strip()
        with st.spinner("🔎 Searching across multiple sources..."):
            search_results = multi_source_search(claim, search_engines)
        st.info(f"📊 Found {len(search_results)} results from {len(search_engines)} search engine(s)")
        
        if len(search_results) < min_sources:
            st.error("⚠️ Insufficient search results for reliable verification. Try enabling more search engines.")
        else:
            with st.expander("🔍 Search Results Preview", expanded=False):
                for i, result in enumerate(search_results[:5]):
                    st.markdown(f"**{i+1}. [{result.get('title','')}]({result.get('url','')})**")
                    st.caption(f"Source: {extract_domain(result.get('url',''))} | {result.get('snippet','')[:150]}...")
                    st.divider()
            
            with st.spinner(f"🤖 Analyzing {len(search_results)} sources with AI..."):
                verification = verify_claim_with_ai(claim, search_results, min_sources)
                
                # Store in history
                verification_record = {
                    'claim': claim,
                    'timestamp': datetime.now().isoformat(),
                    'verification': verification,
                    'search_count': len(search_results)
                }
                st.session_state.verification_history.insert(0, verification_record)
                
                # Display results
                st.markdown("---")
                st.subheader("📋 Verification Results")
                
                classification = verification['classification']
                confidence = verification['confidence']
                
                if classification == 'True':
                    st.success(f"✅ **VERIFIED TRUE** (Confidence: {confidence:.0%})")
                elif classification == 'False':
                    st.error(f"❌ **VERIFIED FALSE** (Confidence: {confidence:.0%})")
                elif classification == 'Unverified':
                    st.warning(f"⚠️ **UNVERIFIED** (Confidence: {confidence:.0%})")
                else:
                    st.info(f"ℹ️ **{classification.upper()}**")
                
                st.progress(confidence)
                st.markdown("### 📝 Analysis")
                st.write(verification.get('reasoning', 'No reasoning available.'))
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Sources Analyzed", verification.get('sources_analyzed', 0))
                col2.metric("Supporting", len(verification.get('supporting_sources', [])))
                col3.metric("Contradicting", len(verification.get('contradicting_sources', [])))
                col4.metric("Search Results", len(search_results))
                
                if verification.get('supporting_sources'):
                    with st.expander("✅ Supporting Sources"):
                        for s in verification['supporting_sources']:
                            st.markdown(f"- [{s['source']}]({s['url']})")
                
                if verification.get('contradicting_sources'):
                    with st.expander("❌ Contradicting Sources"):
                        for s in verification['contradicting_sources']:
                            st.markdown(f"- [{s['source']}]({s['url']})")
                
                # Timeline
                timeline = build_evidence_timeline(claim, verification)
                if timeline:
                    with st.expander("📅 Evidence Timeline"):
                        for ev in timeline:
                            st.write(f"- {ev['timestamp']}: {ev['event']} ({ev.get('source')})")
                
elif mode == "Trending Misinformation Monitor":
    st.header("📈 Trending Misinformation Monitor")
    st.write("Fetch Top-K per platform, then select topics to verify.")
    
    # Refresh / fetch top-k lists
    if st.button("Refresh Top Lists"):
        subs = [s.strip() for s in reddit_subs_input.split(',') if s.strip()]
        st.session_state._latest_topk = fetch_top_per_platform(top_k=top_k, include_twitter=include_twitter, reddit_subreddits=subs, reddit_min_ups=reddit_min_ups)
        st.success("Trend lists refreshed")
    
    top_lists = st.session_state.get('_latest_topk') or fetch_top_per_platform(top_k=top_k, include_twitter=include_twitter, reddit_subreddits=[s.strip() for s in reddit_subs_input.split(',') if s.strip()], reddit_min_ups=reddit_min_ups)
    
    st.markdown("### 🔥 Top items per platform")
    # show each platform separately
    for platform_label, display_name in [
        ('google_trends', '🔥 Google Trends'),
        ('google_news',  '📰 Google News'),
        ('wikipedia',    '📚 Wikipedia'),
        ('twitter',      '🐦 Twitter (Trends24)'),
        ('reddit',       '👥 Reddit'),
        ('hackernews',   '💻 Hacker News'),
        ('bbc_news',     '📡 BBC News'),
        ('reuters',      '🌐 Reuters'),
        ('cnn',          '📺 CNN'),
        ('techcrunch',   '💼 TechCrunch'),
        ('aljazeera',    '🌍 Al Jazeera'),
        ('times_of_india', '🇮🇳 Times of India'),
        ('ndtv',         '📰 NDTV'),
        ('thehindu',     '📖 The Hindu')
    ]:
        items = top_lists.get(platform_label, [])
        if items:
            st.markdown(f"**{display_name}**")
            for i, it in enumerate(items, start=1):
                url = it.get('url') or ''
                extra = ""
                if it.get('views'):
                    extra = f" (views:{it.get('views')})"
                if it.get('engagement'):
                    extra = f" (ups:{it.get('engagement')})"
                if url:
                    st.markdown(f"{i}. [{it.get('title')}]({url}) — {it.get('source')}{extra}")
                else:
                    st.markdown(f"{i}. {it.get('title')} — {it.get('source')}{extra}")
            st.markdown("---")
    
    # allow selection and verification
    combined = []
    for plat in ['google_trends','google_news','wikipedia','twitter','reddit','hackernews','bbc_news','reuters','cnn','techcrunch','aljazeera','times_of_india','ndtv','thehindu']:
        for item in top_lists.get(plat, []):
            combined.append({'label': f"{item.get('title')} — ({item.get('source')})", 'title': item.get('title')})
    labels = [c['label'] for c in combined]
    selected = st.multiselect("Select topics from above to verify:", options=labels)
    selected_topics = [s.split(' — (')[0] for s in selected]
    
    if st.button("Run verification on selected") and selected_topics:
        for topic in selected_topics:
            with st.spinner(f"Searching and analyzing: {topic[:80]}"):
                search_results = multi_source_search(topic, search_engines)
                verification = verify_claim_with_ai(topic, search_results, min_sources)
                st.markdown(f"### {topic}")
                cls = verification.get('classification', 'Unknown')
                conf = verification.get('confidence', 0.0)
                if cls == 'True':
                    st.success(f"✅ {cls} ({conf:.0%})")
                elif cls == 'False':
                    st.error(f"❌ {cls} ({conf:.0%})")
                elif cls == 'Unverified':
                    st.warning(f"⚠️ {cls} ({conf:.0%})")
                else:
                    st.info(f"ℹ️ {cls} ({conf:.0%})")
                st.write(verification.get('reasoning', 'No reasoning available.'))
                if verification.get('supporting_sources'):
                    st.write("Supporting sources:")
                    for s in verification['supporting_sources']:
                        st.markdown(f"- [{s.get('source')}]({s.get('url')})")
                if verification.get('contradicting_sources'):
                    st.write("Contradicting sources:")
                    for s in verification['contradicting_sources']:
                        st.markdown(f"- [{s.get('source')}]({s.get('url')})")
                # record history
                st.session_state.verification_history.insert(0, {'claim': topic, 'timestamp': datetime.now().isoformat(), 'verification': verification, 'search_count': len(search_results)})
    
# ----------------- Verification History -----------------
if st.session_state.verification_history:
    st.markdown("---")
    st.header("📚 Verification History")
    
    for i, record in enumerate(st.session_state.verification_history[:10]):
        classification = record['verification']['classification']
        confidence = record['verification']['confidence']
        
        if classification == 'True':
            icon = "✅"
        elif classification == 'False':
            icon = "❌"
        elif classification == 'Unverified':
            icon = "⚠️"
        else:
            icon = "ℹ️"
        
        with st.expander(f"{icon} {classification} ({confidence:.0%}) - {record['claim'][:60]}..."):
            st.write(f"**Claim:** {record['claim']}")
            st.write(f"**Result:** {classification} ({confidence:.0%})")
            st.write(f"**Reasoning:** {record['verification'].get('reasoning','')}")
            st.caption(f"Verified: {record['timestamp']} | Sources: {record['search_count']}")

# ----------------- Footer -----------------
st.markdown("---")
st.caption("⚠️ **Note:** This is an AI-powered tool. Always verify critical information from authoritative sources.")
st.caption("💡 **Tip:** Enable multiple search engines in sidebar for better accuracy")
