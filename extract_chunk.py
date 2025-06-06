from PyPDF2 import PdfReader
from pyspark.sql import SparkSession
from transformers import pipeline

# Initialize zero-shot classifier once
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)  # use GPU if available

candidate_labels = [
    "Scope 1", "Scope 2", "Scope 3",
    "Water Reduction", "Waste Reduction",
    "Sustainable Packaging", "Governance", "Other"
]

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += "\n" + text
    return full_text

def chunk_text(text, chunk_size=4000):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def batch_classify_chunks(chunks, batch_size=8):
    predicted_categories = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        results = classifier(batch, candidate_labels, multi_label=False)
        if isinstance(results, dict):  # If only 1 item
            results = [results]
        predicted_categories.extend([r['labels'][0] for r in results])
    return predicted_categories

def chunk_pdf_with_spark(pdf_path_or_file, spark_output_path="esg_chunks_output"):
    spark = SparkSession.builder.appName("ESG_PDF_Chunking").getOrCreate()
    
    if hasattr(pdf_path_or_file, "read"):
        content = extract_text_from_pdf(pdf_path_or_file)
    else:
        content = extract_text_from_pdf(open(pdf_path_or_file, "rb"))

    chunks = chunk_text(content)
    predictions = batch_classify_chunks(chunks, batch_size=8)

    classified = list(zip(range(len(chunks)), chunks, predictions))

    df_chunks = spark.createDataFrame(classified, ["chunk_id", "text", "predicted_category"])
    df_chunks.write.mode("overwrite").option("header", "true").csv(spark_output_path)
    
    return df_chunks
