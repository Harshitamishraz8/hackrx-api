import os
import fitz  # PyMuPDF
import tempfile
import requests
from PIL import Image
import pytesseract

def download_pdf(url):
    response = requests.get(url)
    if response.status_code == 200:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp_file.write(response.content)
        tmp_file.close()
        return tmp_file.name
    else:
        raise Exception("Failed to download PDF")

def extract_text_from_pdf(file_path):
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            page_text = page.get_text()
            if not page_text.strip():
                # Use OCR fallback
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_text = pytesseract.image_to_string(img)
            text += page_text + "\n"
        doc.close()
    except Exception as e:
        print(f"Text extraction failed: {e}")
    return text
