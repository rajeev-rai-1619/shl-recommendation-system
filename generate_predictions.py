import os
import requests
import pandas as pd

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/recommend")

def generate_predictions():
    print("Loading Test Set...")
    try:
        # Load the test dataset you uploaded earlier
        df_test = pd.read_excel("data/Gen_AI Dataset.xlsx", sheet_name="Test-Set")
    except Exception as e:
        print(f"Error loading Test Set: {e}")
        return
        
    test_queries = df_test['Query'].tolist()
    submission_data = []
    
    print(f"Processing {len(test_queries)} test queries through the API...\n")
    
    for i, query in enumerate(test_queries):
        try:
            response = requests.post(API_URL, json={"query": query})
            response.raise_for_status()
            
            data = response.json()
            recommendations = data.get("recommended_assessments", [])
            
            # The instructions ask for a minimum of 5 and maximum of 10.
            urls_added = 0
            for rec in recommendations[:10]:
                submission_data.append({
                    "Query": query,
                    "Assessment_url": rec["url"]
                })
                urls_added += 1
                
            print(f"Query {i+1} processed: Generated {urls_added} recommendations.")
            
        except Exception as e:
            print(f"Error processing query {i+1}: {e}")

    # Create the final DataFrame
    df_predictions = pd.DataFrame(submission_data)
    
    # Save exactly in the required 2-column format
    df_predictions.to_csv("predictions.csv", index=False)
    print(f"\nSuccess! Generated 'predictions.csv' with {len(df_predictions)} total rows.")
    print("This file is perfectly formatted for your final submission!")

if __name__ == "__main__":
    generate_predictions()