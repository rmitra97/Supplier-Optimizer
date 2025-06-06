from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import pickle

# Load your chunked data (already done via PySpark and saved as CSV)
df = pd.read_csv("/Users/kopalbhatnagar/Documents/Pipeline/BDA/esg_pdf_chunks.csv")  # This CSV must have a column called 'text'

# Load a pre-trained embedding model
model = SentenceTransformer("BAAI/bge-large-en-v1.5")   # Lightweight & fast

# Generate embeddings for each chunk
df["embedding"] = df["text"].apply(lambda x: model.encode(x).tolist())  # Convert np.array to list

# Save embeddings + metadata
embeddings = df[["text", "embedding"]]  


with open("/Users/kopalbhatnagar/Documents/Pipeline/BDA/chunk_embeddings.pkl", "wb") as f:
    pickle.dump(embeddings.to_dict(orient="records"), f)

print("✅ Embeddings generated and saved.")
