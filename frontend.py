import streamlit as st
import requests
import pandas as pd
import os

# Check Streamlit secrets first (for Streamlit Cloud), then env var, then localhost
try:
    API_URL = st.secrets["API_URL"]
except (KeyError, FileNotFoundError):
    API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/recommend")

st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")

st.title(" SHL Assessment Recommendation System")
st.markdown("""
Welcome! Paste a **Natural Language Query**, **Job Description**, or a **Job Description URL** below,  
and our AI will recommend the 5-10 most relevant SHL assessments.
""")

# Text input for the user
user_query = st.text_area("Enter Job Description, Query, or URL:", height=200, 
                          placeholder="e.g., Looking for a mid-level Java Developer with good communication skills...\nor paste a URL: https://example.com/job-posting")

if st.button("Get Recommendations", type="primary"):
    if not user_query.strip():
        st.warning("Please enter a query first.")
    else:
        with st.spinner("Analyzing query and finding the best assessments..."):
            try:
                # Send the query to your FastAPI backend
                response = requests.post(API_URL, json={"query": user_query})
                
                if response.status_code == 200:
                    data = response.json()
                    recommendations = data.get("recommended_assessments", [])
                    
                    if recommendations:
                        st.success(f"Found {len(recommendations)} relevant assessments!")
                        
                        # Convert to a Pandas DataFrame for a nice table display
                        df = pd.DataFrame(recommendations)
                        
                        # Reorder columns to make it look nicer
                        cols = ["name", "test_type", "duration", "remote_support", "adaptive_support", "description", "url"]
                        df = df[[c for c in cols if c in df.columns]]
                        
                        # Display the table
                        st.dataframe(
                            df,
                            column_config={
                                "name": "Assessment Name",
                                "url": st.column_config.LinkColumn("SHL Link"),
                                "description": "Description",
                                "duration": "Duration (mins)",
                                "test_type": "Test Types",
                                "remote_support": "Remote?",
                                "adaptive_support": "Adaptive?"
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                    else:
                        st.info("No matching assessments found. Try adjusting your query.")
                else:
                    st.error(f"API Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("🚨 Could not connect to the API. Make sure your FastAPI server (`uvicorn app:app`) is running!")
            except Exception as e:
                st.error(f"An error occurred: {e}")