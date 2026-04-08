from extract import extract_text
from chunker import chunk_text_fixed
from embedder import embed_text
from retriever import retrieve
from generator import generate_response

def run_pipeline(file_path: str, query: str) -> str:
    pages = extract_text(file_path)
    chunks = chunk_text_fixed(pages)
    texts = [chunk["text"] for chunk in chunks]
    chunk_embeddings = embed_text(texts)
    retrieved_chunks = retrieve(query, chunks, chunk_embeddings)
    answer = generate_response(query, retrieved_chunks)
    return answer