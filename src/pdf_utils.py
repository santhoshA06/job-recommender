from io import BytesIO
from typing import BinaryIO
import pdfplumber

def extract_text_from_pdf(file_obj: BinaryIO) -> str:
    """
    Extract text from a text-based PDF.
    Works with Streamlit uploaded files or normal binary file handles.
    """
    if hasattr(file_obj, "read"):
        raw_bytes = file_obj.read()
    else:
        raise ValueError("Invalid file object")

    text_parts = []
    with pdfplumber.open(BytesIO(raw_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    return "\n".join(text_parts).strip()