import os
import re
import time
import logging
from urllib.parse import urlparse
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import requests as http_requests
from bs4 import BeautifulSoup
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
from google.genai import types

load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
client = genai.Client(api_key=GEMINI_API_KEY)

def _call_with_retry(fn, max_retries=3):
    """Call fn(), retrying on 429 RESOURCE_EXHAUSTED with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 10 * (2 ** attempt)  # 10s, 20s, 40s
                logger.warning("Rate limited (attempt %d/%d), waiting %ds...", attempt + 1, max_retries, wait)
                time.sleep(wait)
            else:
                raise
    # Final attempt without catching
    return fn()


# Custom embedding function for searching queries using Gemini embeddings
class GeminiQueryEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        def _embed():
            return client.models.embed_content(
                model="gemini-embedding-001",
                contents=input,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY"
                )
            )
        result = _call_with_retry(_embed)
        return [e.values for e in result.embeddings]

app = FastAPI(title="SHL Assessment Recommendation API")

collection = None
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_gemini_data")
    gemini_ef = GeminiQueryEmbeddingFunction()
    collection = chroma_client.get_collection(name="shl_assessments_gemini", embedding_function=gemini_ef)
except Exception as e:
    logger.warning("Could not load ChromaDB. Error: %s", e)

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

def _is_url(text: str) -> bool:
    """Check if the input looks like a URL."""
    text = text.strip()
    try:
        parsed = urlparse(text)
        return parsed.scheme in ('http', 'https') and bool(parsed.netloc)
    except Exception:
        return False


def _extract_text_from_url(url: str, max_chars: int = 5000) -> str:
    """Fetch a URL and extract visible text content."""
    resp = http_requests.get(url, timeout=15, headers={
        "User-Agent": "Mozilla/5.0 (compatible; SHL-Recommender/1.0)"
    })
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, 'html.parser')
    # Remove script/style elements
    for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
        tag.decompose()
    text = soup.get_text(separator=' ', strip=True)
    return text[:max_chars]


def sanitize_query(query: str) -> str:
    """Strip control characters and obvious prompt injection patterns from user input."""
    query = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', query)
    return query.strip()

def generate_search_queries(query: str) -> dict:
    """Use LLM to generate multiple targeted search queries + identify types."""
    query = sanitize_query(query)
    prompt = f"""You are an expert HR assessment consultant for SHL. Given a job description or hiring query, you must:

1. Generate 3-5 SHORT search queries (each 5-15 words) to find relevant assessments. Cover different aspects:
   - Technical/domain skills mentioned (e.g. "Java programming test", "SQL database assessment")
   - Cognitive abilities if relevant (e.g. "numerical reasoning ability test", "verbal comprehension assessment")  
   - Personality/behavioral if relevant (e.g. "personality questionnaire workplace behavior", "interpersonal communication skills")
   - Role-specific simulations if relevant (e.g. "coding simulation debugging", "data entry simulation")
2. Identify relevant SHL test types: A(Ability/Aptitude), B(Biodata/SJT), C(Competencies), D(Development/360), E(Exercises), K(Knowledge/Skills), P(Personality/Behavior), S(Simulations)

Query: "{query}"

Respond STRICTLY in this format (no extra text):
Queries: query1 | query2 | query3 | query4
Types: K, P, A"""
    try:
        def _gen():
            return client.models.generate_content(
                model='gemini-2.5-flash-lite',
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0),
            )
        response = _call_with_retry(_gen)
        text = response.text
        search_queries = [query]  # always include original
        types_list = []
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('Queries:'):
                raw = line.replace('Queries:', '').strip()
                search_queries = [q.strip() for q in raw.split('|') if q.strip()]
                search_queries.insert(0, query)  # original first
            elif line.startswith('Types:'):
                type_str = line.replace('Types:', '').strip()
                types_list = [t.strip().upper() for t in type_str.split(',')
                              if t.strip().upper() in ['A','B','C','D','E','K','P','S']]

        return {"search_queries": search_queries[:6], "types": types_list}
    except Exception as e:
        logger.error("LLM Error: %s", e)
        return {"search_queries": [query], "types": ['K', 'P']}


def rerank_with_llm(query: str, candidates: list) -> list:
    """Use LLM to pick the 10 most relevant assessments from candidates."""
    query = sanitize_query(query)
    # Build a numbered list of candidates for the LLM
    candidate_lines = []
    for i, c in enumerate(candidates):
        name = c.get('name', '')
        desc = c.get('description', '')[:150]
        test_types = c.get('test_type', '')
        candidate_lines.append(f"{i}. {name} [Types: {test_types}] - {desc}")

    candidates_text = "\n".join(candidate_lines)

    prompt = f"""You are an expert HR assessment consultant. Given the hiring query and a list of candidate SHL assessments, select the 10 MOST RELEVANT assessments.

Consider:
- Direct skill match (technical skills in the query should map to matching knowledge tests)
- Cognitive abilities needed for the role (numerical, verbal, inductive reasoning)
- Personality/behavioral fit requirements
- Simulations relevant to the job (coding, data entry, etc.)
- Balance between hard skills (K) and soft skills (P, A) when the query implies both

Hiring Query: "{query}"

Candidate Assessments:
{candidates_text}

Return ONLY the index numbers of the 10 most relevant assessments, ordered from most to least relevant.
Format: 0, 5, 12, 3, 8, 1, 15, 7, 20, 11"""

    try:
        def _gen():
            return client.models.generate_content(
                model='gemini-2.5-flash-lite',
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0),
            )
        response = _call_with_retry(_gen)
        text = response.text.strip()
        # Parse indices from response — extract all numbers via regex for robustness
        indices = []
        for match in re.finditer(r'\b(\d+)\b', text):
            idx = int(match.group(1))
            if 0 <= idx < len(candidates) and idx not in indices:
                indices.append(idx)
            if len(indices) >= 10:
                break
        # Fallback: if parsing produced nothing, return distance-sorted top 10
        if not indices:
            indices = list(range(min(10, len(candidates))))
        return indices
    except Exception as e:
        logger.error("Rerank LLM Error: %s", e)
        return list(range(min(10, len(candidates))))


TYPE_SEARCH_NAMES = {
    'A': 'ability aptitude reasoning test',
    'B': 'biodata situational judgement test',
    'C': 'competency assessment',
    'D': 'development 360 feedback',
    'E': 'assessment exercise',
    'K': 'knowledge skills technical test',
    'P': 'personality behavior workplace questionnaire OPQ',
    'S': 'simulation work sample test',
}


def _meta_to_result(meta: dict) -> dict:
    """Convert a ChromaDB metadata dict to an API response item."""
    return {
        "url": meta.get('url', ""),
        "name": meta.get('name', ""),
        "adaptive_support": meta.get('adaptive_support', "No"),
        "description": meta.get('description', ""),
        "duration": int(meta.get('duration', 0)),
        "remote_support": meta.get('remote_support', "No"),
        "test_type": [t.strip() for t in meta.get('test_type', '').split(',') if t.strip()],
    }


def _merge_query_results(candidates: dict, results: dict, type_filter: str = None):
    """Merge ChromaDB query results into the candidates dict (url -> (meta, best_dist)).
    If type_filter is set, only include results whose test_type contains that letter."""
    for meta, dist in zip(
        results.get('metadatas', [[]])[0],
        results.get('distances', [[]])[0],
    ):
        url = meta.get('url', '')
        if type_filter and type_filter not in meta.get('test_type', ''):
            continue
        if url not in candidates or dist < candidates[url][1]:
            candidates[url] = (meta, dist)


def _retrieve_candidates(search_queries: list, target_types: list) -> dict:
    """multi-query retrieval with type-aware gap filling."""
    candidates = {}  # url -> (metadata, best_distance)

    # run each search query
    for sq in search_queries:
        try:
            results = collection.query(query_texts=[sq], n_results=10,
                                       include=["metadatas", "distances"])
            _merge_query_results(candidates, results)
        except Exception:
            pass

    # fill gaps for underrepresented target types
    type_counts = {}
    for meta, _ in candidates.values():
        for t in meta.get('test_type', '').split(','):
            t = t.strip()
            if t:
                type_counts[t] = type_counts.get(t, 0) + 1

    for target in target_types:
        if type_counts.get(target, 0) < 3:
            try:
                results = collection.query(
                    query_texts=[TYPE_SEARCH_NAMES.get(target, target)],
                    n_results=10, include=["metadatas", "distances"])
                _merge_query_results(candidates, results, type_filter=target)
            except Exception:
                pass

    return candidates


def _is_technical_dominant(target_types: list, reranked: list) -> bool:
    """Detect if the query is primarily technical/knowledge-focused.
    Returns True when hard-skill types (K, S) make up the vast majority
    of both the requested types AND the re-ranked results, meaning
    forced diversification into soft-skill types would hurt relevance."""
    hard_types = {'K', 'S'}
    soft_types = {'P', 'A', 'B', 'C', 'D', 'E'}
    # If targets are all hard-skill types, definitely technical
    if set(target_types).issubset(hard_types):
        return True
    # If >=70% of re-ranked items are hard-skill, treat as technical-dominant
    if reranked:
        hard_count = sum(1 for item in reranked
                         if any(t in hard_types for t in item["test_type"]))
        if hard_count / len(reranked) >= 0.7:
            return True
    return False


def _balance_results(reranked: list, candidates: dict, target_types: list) -> list:
    """Proportional type balancing across target types.
    Skips forced round-robin when the query is predominantly technical,
    so pure-skill queries are not diluted with irrelevant personality tests."""
    if len(target_types) <= 1:
        return reranked[:10]

    # For technical-dominant queries, trust the LLM re-ranking order
    if _is_technical_dominant(target_types, reranked):
        return reranked[:10]

    final, used_urls = [], set()

    # Bucket by target type
    type_buckets = {t: [] for t in target_types}
    other_bucket = []
    for item in reranked:
        placed = False
        for t in target_types:
            if t in item["test_type"]:
                type_buckets[t].append(item)
                placed = True
                break
        if not placed:
            other_bucket.append(item)

    # Backfill buckets from full candidate pool
    for meta, _ in sorted(candidates.values(), key=lambda x: x[1]):
        item = _meta_to_result(meta)
        for t in target_types:
            if t in item["test_type"] and not any(b["url"] == item["url"] for b in type_buckets[t]):
                type_buckets[t].append(item)
                break

    # Weighted slot allocation — give more slots to types with more re-ranked hits
    type_hit_counts = {t: len(type_buckets[t]) for t in target_types}
    total_hits = sum(type_hit_counts.values()) or 1
    for t in target_types:
        # Proportional slots (min 1, max 7) instead of equal split
        slots = max(1, min(7, round(10 * type_hit_counts[t] / total_hits)))
        count = 0
        for item in type_buckets[t]:
            if count >= slots or len(final) >= 10:
                break
            if item["url"] not in used_urls:
                final.append(item)
                used_urls.add(item["url"])
                count += 1

    # Fill remaining from reranked order, then other bucket
    for item in reranked + other_bucket:
        if len(final) >= 10:
            break
        if item["url"] not in used_urls:
            final.append(item)
            used_urls.add(item["url"])

    return final[:10]


@app.post("/recommend")
def recommend_assessments(request: QueryRequest):
    if collection is None:
        raise HTTPException(status_code=503, detail="Vector database is not available")
    try:
        # If input is a URL, fetch and extract text from it
        query_text = request.query
        if _is_url(query_text):
            try:
                query_text = _extract_text_from_url(query_text)
                print(f"Extracted text from URL for query: {query_text}...")
            except Exception as e:
                logger.warning("Failed to fetch URL: %s", e)
                raise HTTPException(status_code=400, detail="Could not fetch content from the provided URL")

        # LLM query expansion
        analysis = generate_search_queries(query_text)
        print(f"Generated search target types: {analysis['types']}")
        # Retrieval with type gap filling
        candidates = _retrieve_candidates(analysis["search_queries"], analysis["types"])
        if not candidates:
            return {"recommended_assessments": []}

        # LLM re-ranking
        top30 = sorted(candidates.values(), key=lambda x: x[1])[:30]
        candidate_metas = [meta for meta, _ in top30]
        ranked_indices = rerank_with_llm(query_text, candidate_metas)
        reranked = [_meta_to_result(candidate_metas[i]) for i in ranked_indices]

        # Type balancing
        final = _balance_results(reranked, candidates, analysis["types"])
        return {"recommended_assessments": final}

    except Exception as e:
        logger.error("Recommendation error: %s", e)
        raise HTTPException(status_code=500, detail="Internal Server Error")