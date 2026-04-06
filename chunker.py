def chunk_text_fixed(text: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    chunks = []
    start = 0
    index = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append({
            "text": text[start:end],
            "chunk_index": index,
            "start_char": start,
            "end_char": end
        })
        start += chunk_size - overlap
        index += 1
    return chunks