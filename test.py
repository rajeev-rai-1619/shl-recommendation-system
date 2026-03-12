import requests

# # Test Health Endpoint
# health = requests.get("http://127.0.0.1:8000/health")
# print("Health Check:", health.json())

# # Test Recommendation Endpoint
# payload = {"query": "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes."}
# response = requests.post("http://127.0.0.1:8000/recommend", json=payload)

# print("\nRecommendations Found:", len(response.json()['recommended_assessments']))
# print(response.json())

# Check Accuracy on the testimport requests
import pandas as pd

API_URL = "http://127.0.0.1:8000/recommend"

def normalize_url(url):
    """Extracts the unique assessment name from the URL for fair comparison."""
    return str(url).split('?')[0].strip('/').split('/')[-1]

def calculate_recall_at_k(recommended_urls, ground_truth_urls, k=10):
    """Calculates Recall@K for a single query."""
    top_k_recs = recommended_urls[:k]
    
    # Normalize both lists
    norm_recs = set([normalize_url(url) for url in top_k_recs])
    norm_truths = set([normalize_url(url) for url in ground_truth_urls])
    
    relevant_retrieved = len(norm_recs.intersection(norm_truths))
    total_relevant = len(norm_truths)
    
    if total_relevant == 0:
        return 0.0
        
    return relevant_retrieved / total_relevant

def evaluate_model():
    print("Loading Train Set from Excel...")
    try:
        df_train = pd.read_excel("data/Gen_AI Dataset.xlsx", sheet_name="Train-Set")
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Group by Query to get a list of correct URLs for each unique query
    ground_truth = df_train.groupby('Query')['Assessment_url'].apply(list).to_dict()
    
    total_recall = 0.0
    num_queries = len(ground_truth)
    
    print(f"Evaluating {num_queries} unique queries against the API...\n")
    
    for i, (query, correct_urls) in enumerate(ground_truth.items()):
        try:
            response = requests.post(API_URL, json={"query": query})
            response.raise_for_status()
            
            data = response.json()
            recommendations = data.get("recommended_assessments", [])
            
            recommended_urls = [rec['url'] for rec in recommendations]
            
            recall = calculate_recall_at_k(recommended_urls, correct_urls, k=10)
            total_recall += recall
            
            print(f"Query {i+1}/{num_queries}: Recall@10 = {recall:.2f}")
            
        except Exception as e:
            print(f"Error processing query {i+1}: {e}")
            
    mean_recall = total_recall / num_queries if num_queries > 0 else 0
    print(f"\n====================================")
    print(f"FINAL SCORE: Mean Recall@10 = {mean_recall:.4f}")
    print(f"====================================")

if __name__ == "__main__":
    evaluate_model()