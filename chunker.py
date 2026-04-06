def chunk_text_fixed(pages:list[dict], chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    chunks = []
    for page in pages:
        text = page["text"]
        page_number = page["page_number"]
        start =0
        while start <len(text):
            end = start + chunk_size
            chunks.append({
                "text":text[start:end],
                "page_number":page_number,
                "start_char":start,
            })
            start += chunk_size-overlap
    
    return chunks