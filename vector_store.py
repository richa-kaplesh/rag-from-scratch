import numpy as np

def cosine_similarity(query_vec:np.ndarray,chunk_vecs:np.ndarray) ->np.ndarray:
    dot_product = np.dot(query_vec,chunk_vecs)
    query_magnitude = np.linalg.norm(query_vec)
    chunk_magnitudes = np.linalg.norm(chunk_vecs, axis=1)
    return  dot_product / (query_magnitude * chunk_magnitudes )

def get_top_k_chunks(query_vec:np.ndarray, chunk_vecs:np.ndarray , k:int=3)->list[int]:
    similarities = cosine_similarity(query_vec, chunk_vecs)
    top_k_indices = np.argsort(similarities)[::-1][:k]
    return top_k_indices.tolist()
