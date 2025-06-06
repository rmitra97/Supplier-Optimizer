import os
import pickle
from pinecone import Pinecone, ServerlessSpec

# Load environment variable (or replace with your actual key and region)
api_key="pcsk_6MyPdd_NCqUbEQySzrC5i3denToMkuhfGcGeqh1D5h7KX6haXzvTDDPQr8Fgxft7GdeZkJ"
environment = os.environ.get("PINECONE_ENV") or "us-east-1"

# Create Pinecone client
pc = Pinecone(api_key=api_key)

# Create index if it doesn’t exist
index_name = "esg-analysis"
dimension = 384  # or 1536 depending on your embedding model

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=environment)
    )

index = pc.Index(index_name)

# Load embeddings from file
with open("/Users/kopalbhatnagar/Documents/Pipeline/BDA/chunk_embeddings.pkl", "rb") as f:
    data = pickle.load(f)

# Upload to Pinecone
vectors = [
    {
        "id": f"chunk-{i}",
        "values": item["embedding"],
        "metadata": {"text": item["text"]}
    }
    for i, item in enumerate(data)
]

index.upsert(vectors=vectors)

print("✅ Uploaded all embeddings to Pinecone.")
