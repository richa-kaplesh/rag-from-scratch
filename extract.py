import fitz  # PyMuPDF

def extract_text(file_path: str) -> str:
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text.strip()