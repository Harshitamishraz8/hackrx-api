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
            text += page.get_text()
        doc.close()
        # Fallback to OCR if extracted text is too small
        if len(text.strip()) < 50:
            text = extract_text_with_ocr(file_path)
    except Exception as e:
        print(f"Error during PDF text extraction: {e}")
        text = extract_text_with_ocr(file_path)
    return text

def extract_text_with_ocr(file_path):
    print("Falling back to OCR...")
    text = ""
    doc = fitz.open(file_path)
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text += pytesseract.image_to_string(img)
    return text
