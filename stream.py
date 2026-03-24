import streamlit as st
import requests
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

def get_search_results(query, top_k):
    url = "http://localhost:8000/api/embeddings/search/"
    
    query_req = QueryRequest(query=query, top_k=top_k)
    
    response = requests.post(url, json=query_req.dict())
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Failed to retrieve search results")
        return []

def display_search_results(results):

    if not results:
        st.write("No results found.")
        return
    
    for result in results:
        st.subheader(result.get('metadata', {}).get('source', 'No Source'))
        st.write(f"**Page Content:**\n{result.get('page_content', 'No content available.')}")
        st.write(f"**Type:** {result.get('type', 'Unknown Type')}")
        st.write("---")

def main():
    st.title("Search Embeddings")
    
    query = st.text_input("Enter your search query:")
    
    top_k = st.slider("Number of Results (Top K)", min_value=1, max_value=20, value=5)
    
    if st.button("Search"):
        if query:
            with st.spinner("Searching..."):
                results = get_search_results(query, top_k)
                
                display_search_results(results)
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
