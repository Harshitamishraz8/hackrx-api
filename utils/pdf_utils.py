import requests
import fitz  # PyMuPDF
import io

def extract_text_from_url(url: str) -> str:
    try:
        response = requests.get(url)
        response.raise_for_status()
        doc = fitz.open(stream=io.BytesIO(response.content), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""
