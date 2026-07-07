import fitz  # pymupdf
import docx
import pytesseract
from PIL import Image
import io
from typing import List

# Point pytesseract to your Tesseract install (adjust path if yours differs)
import platform

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Users\simpl\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
# On Linux (Streamlit Cloud), tesseract-ocr installs to system PATH automatically — no path needed

MIN_TEXT_LENGTH = 50  # if extracted text is shorter than this, assume it's a scanned PDF

def load_pdf(path: str) -> str:
    doc = fitz.open(path)
    text = "\n".join(page.get_text() for page in doc) 

    if len(text.strip()) < MIN_TEXT_LENGTH:
        print("[Loader] Little/no text found — falling back to OCR...")
        text = ocr_pdf(doc)

    doc.close()
    return text

def ocr_pdf(doc) -> str:
    """Render each PDF page as an image and run OCR on it."""
    ocr_text = []
    for page_num, page in enumerate(doc):
        pix = page.get_pixmap(dpi=300)  # higher dpi = better OCR accuracy
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        page_text = pytesseract.image_to_string(img)
        ocr_text.append(page_text)
        print(f"[OCR] Processed page {page_num + 1}/{len(doc)}")
    return "\n".join(ocr_text)

def load_docx(path: str) -> str:
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def load_document(path: str) -> str:
    if path.endswith(".pdf"):
        return load_pdf(path)
    elif path.endswith(".docx"):
        return load_docx(path)
    elif path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {path}")

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks