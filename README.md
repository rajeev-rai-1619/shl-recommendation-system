# SHL Assessment Recommendation System

An AI-powered recommendation engine that takes a natural language query, job description, or job posting URL and returns the top 10 most relevant SHL assessments from the product catalog.

## Architecture

The system uses a **4-stage RAG (Retrieval-Augmented Generation) pipeline**:

1. **LLM Query Expansion** — Gemini 2.5 Flash analyzes the input and generates multiple targeted search queries + identifies relevant test types (K, P, A, S, etc.)
2. **Multi-Query Vector Retrieval** — Each query is searched against a ChromaDB vector store built with Gemini embeddings (768-dim, asymmetric retrieval). Type-aware gap filling ensures underrepresented assessment types get additional targeted searches.
3. **LLM Re-Ranking** — Top 30 candidates are re-ranked by Gemini 2.5 Flash using domain knowledge about SHL assessments.
4. **Proportional Type Balancing** — Results are balanced across identified types via round-robin slot allocation (e.g., 5 Knowledge + 5 Personality for mixed queries).

## Tech Stack

- **Backend**: FastAPI + Uvicorn
- **Vector Store**: ChromaDB (persistent)
- **Embeddings**: Google `gemini-embedding-001` (asymmetric RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY)
- **LLM**: Google Gemini 2.5 Flash
- **Frontend**: Streamlit
- **Scraping**: requests + BeautifulSoup4

## Project Structure

```
app.py                 # FastAPI backend — recommendation API
frontend.py            # Streamlit web UI
web_scraper.py         # SHL product catalog scraper
build_vectordb.py      # Builds ChromaDB from scraped CSV
test.py                # End-to-end evaluation (Mean Recall@10)
test_retrieval.py      # Retrieval-stage evaluation (vector search only)
generate_predictions.py# Generates predictions.csv for test set
generate_approach_doc.py # Generates Approach.docx
shl_assessments.csv    # Scraped assessment data (377+ rows)
predictions.csv        # Model predictions on test set
requirements.txt       # Pinned Python dependencies
```

## Setup

### 1. Install dependencies

```bash
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file:

```
GEMINI_API_KEY=your_google_gemini_api_key
```

### 3. Build the vector database

```bash
python build_vectordb.py
```

This embeds all 377+ assessments from `shl_assessments.csv` into ChromaDB.

### 4. Start the API server

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

### 5. Launch the frontend (optional)

```bash
streamlit run frontend.py
```

## API Endpoints

### `GET /health`

Returns `{"status": "healthy"}`.

### `POST /recommend`

**Request:**
```json
{
  "query": "Looking for a Java developer assessment with collaboration skills"
}
```

Or with a URL:
```json
{
  "query": "https://example.com/job-posting"
}
```

**Response:**
```json
{
  "recommended_assessments": [
    {
      "url": "https://www.shl.com/products/product-catalog/view/java-8/",
      "name": "Java 8",
      "adaptive_support": "No",
      "description": "...",
      "duration": 20,
      "remote_support": "Yes",
      "test_type": ["K"]
    }
  ]
}
```

## Evaluation

Two evaluation scripts measure system performance at different stages:

| Script | Stage | What it measures |
|--------|-------|-----------------|
| `test_retrieval.py` | Retrieval | Vector search quality before LLM re-ranking |
| `test.py` | End-to-end | Full pipeline output (requires running server) |

```bash
# Retrieval-stage evaluation (no server needed)
python test_retrieval.py

# End-to-end evaluation (server must be running)
python test.py
```

Both use **Mean Recall@10** against labeled training data from `data/Gen_AI Dataset.xlsx`.

## Re-scraping (optional)

To refresh the assessment catalog:

```bash
python web_scraper.py      # Scrapes SHL catalog → shl_assessments.csv
python build_vectordb.py   # Rebuilds vector DB from new CSV
```
