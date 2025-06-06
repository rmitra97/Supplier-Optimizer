"""
ESG Document Search Module
This module provides semantic search functionality for ESG documents using Pinecone vector database
and Sentence Transformers for text embeddings.
"""

# Import required libraries
from sentence_transformers import SentenceTransformer  # For converting text to embeddings
from pinecone import Pinecone  # Vector database for similarity search
import os  # For environment variable access

# Initialize Pinecone client with API key from environment variables
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Connect to the specific Pinecone index for ESG analysis
index = pc.Index("esg-analysis")

# Load the sentence transformer model for generating text embeddings
# all-MiniLM-L6-v2 is a lightweight but effective model for semantic search
model = SentenceTransformer("all-MiniLM-L6-v2")

def search_esg(query_text, top_k=5, category=None):
    """
    Perform semantic search on ESG documents using vector similarity.
    
    Args:
        query_text (str): The search query text
        top_k (int): Number of most similar results to return (default: 5)
        category (str, optional): Filter results by ESG category (e.g., "Scope 1", "Scope 2", etc.)
    
    Returns:
        list: List of matching documents with their metadata and similarity scores
    """
    # Convert the query text to a vector embedding using the transformer model
    vector = model.encode(query_text).tolist()
    
    # Create filter object if category is specified
    # This allows filtering results by ESG category while maintaining semantic search
    filter_obj = {"category": {"$eq": category}} if category else {}
    
    # Query the Pinecone index with the vector and filters
    # include_metadata=True ensures we get back the original text and other metadata
    results = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        filter=filter_obj
    )
    
    # Return the matching documents
    return results["matches"]
