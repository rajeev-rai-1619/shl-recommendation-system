import os
from dotenv import load_dotenv
import pandas as pd
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
from google.genai import types
import time

load_dotenv()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
client = genai.Client(api_key=GEMINI_API_KEY)

class GeminiDocumentEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=input,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                title="SHL Assessment Catalog"
            )
        )
        return [e.values for e in result.embeddings]

def build_vector_db():
    print("Loading assessment data...")
    try:
        df = pd.read_csv("shl_assessments.csv")
    except FileNotFoundError:
        print("Error: 'shl_assessments.csv' not found.")
        return
    
    chroma_client = chromadb.PersistentClient(path="./chroma_gemini_data")
    gemini_ef = GeminiDocumentEmbeddingFunction()
    
    try:
        chroma_client.delete_collection(name="shl_assessments_gemini")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name="shl_assessments_gemini",
        embedding_function=gemini_ef
    )
    
    documents = []
    metadatas = []
    ids = []
    
    print("Preparing documents for Gemini Vectorization...")
    for index, row in df.iterrows():
        name = str(row['name'])
        desc = str(row['description'])
        test_types_raw = str(row['test_type'])
        # Convert "['K', 'P']" list-string to clean comma-separated "K,P"
        try:
            import ast
            test_types_list = ast.literal_eval(test_types_raw)
            if isinstance(test_types_list, list):
                test_types_str = ",".join(test_types_list)
            else:
                test_types_str = str(test_types_list)
        except (ValueError, SyntaxError):
            test_types_str = test_types_raw
        
        rich_text = f"Assessment Name: {name}\nTest Types: {test_types_str}\nDescription: {desc}"
        documents.append(rich_text)
        
        metadatas.append({
            "name": name,
            "url": str(row['url']),
            "description": desc,
            "duration": int(row['duration']) if pd.notna(row['duration']) else 0,
            "remote_support": str(row['remote_support']),
            "adaptive_support": str(row['adaptive_support']),
            "test_type": test_types_str
        })
        
        ids.append(f"assessment_{index}")

    print(f"Upserting {len(documents)} records via Gemini API. Processing carefully to respect rate limits...")
    
    batch_size = 50 
    
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        success = False
        retries = 0
        
        # Retry loop in case we hit the rate limit
        while not success and retries < 5:
            try:
                collection.upsert(
                    documents=batch_docs,
                    metadatas=batch_meta,
                    ids=batch_ids
                )
                success = True
                print(f"Successfully processed batch {i} to {min(i+batch_size, len(documents))}...")
                
                # Rest for 5 seconds between successful batches
                time.sleep(5) 
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    print(f"Rate limit hit at batch {i}! Sleeping for 10 seconds before retrying...")
                    time.sleep(10)
                    retries += 1
                else:
                    print(f"Unexpected Error: {e}")
                    raise e
                    
        if not success:
            print("Failed to process batch after 5 retries. Exiting.")
            return
            
    print("\nPhase 2 Complete! Gemini Vector database saved to './chroma_gemini_data'.")

if __name__ == "__main__":
    build_vector_db()