import fitz

def extract_text(file_path: str) -> list[dict]:
    doc = fitz.open(file_path)
    pages = []
    for page_num, page in enumerate(doc):
        pages.append({
            "text": page.get_text(),
            "page_number": page_num + 1
        })
    return pages