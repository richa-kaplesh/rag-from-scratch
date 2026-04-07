import numpy as np
from embedder import embed_query
from vector_store import get_top_k

def retrieve(query:str , chunks:list[dict],chunk_embeddings:np.ndarray, k:int =3) -> list[dict]:
    query_vec = embed_query(query)
    top_k_indices=get_top_k(query_vec, chunk_embeddings, k=k)
    return [chunks[i] for i in top_k_indices]