"""
Retrieval-Stage Evaluation
--------------------------
Measures Recall@K directly on ChromaDB vector search results,
before any LLM re-ranking or balancing.  
"""

import os
from dotenv import load_dotenv
import pandas as pd
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
client = genai.Client(api_key=GEMINI_API_KEY)


class GeminiQueryEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=input,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return [e.values for e in result.embeddings]


def normalize_url(url):
    """Extracts the unique assessment slug from a URL for fair comparison."""
    return str(url).split("?")[0].strip("/").split("/")[-1]


def recall_at_k(recommended_urls, ground_truth_urls, k=10):
    top_k = recommended_urls[:k]
    norm_recs = set(normalize_url(u) for u in top_k)
    norm_truth = set(normalize_url(u) for u in ground_truth_urls)
    if not norm_truth:
        return 0.0
    return len(norm_recs & norm_truth) / len(norm_truth)


def evaluate_retrieval(n_results=10):
 
    chroma_client = chromadb.PersistentClient(path="./chroma_gemini_data")
    gemini_ef = GeminiQueryEmbeddingFunction()
    collection = chroma_client.get_collection(
        name="shl_assessments_gemini", embedding_function=gemini_ef
    )

    # Load ground-truth train data
    print("Loading Train Set from Excel...")
    df_train = pd.read_excel("data/Gen_AI Dataset.xlsx", sheet_name="Train-Set")
    ground_truth = df_train.groupby("Query")["Assessment_url"].apply(list).to_dict()

    num_queries = len(ground_truth)
    total_recall = 0.0

    print(f"Evaluating retrieval stage on {num_queries} queries  (n_results={n_results})\n")

    for i, (query, correct_urls) in enumerate(ground_truth.items()):
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["metadatas"],
        )

        retrieved_urls = [
            meta.get("url", "")
            for meta in results.get("metadatas", [[]])[0]
        ]

        r = recall_at_k(retrieved_urls, correct_urls, k=n_results)
        total_recall += r

        print(f"Query {i + 1}/{num_queries}: Recall@{n_results} = {r:.2f}")

    mean_recall = total_recall / num_queries if num_queries else 0
    print(f"\n{'=' * 50}")
    print(f"RETRIEVAL STAGE — Mean Recall@{n_results} = {mean_recall:.4f}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    evaluate_retrieval(n_results=10)
