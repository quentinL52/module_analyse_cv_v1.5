import re
import fitz  # PyMuPDF
import pymupdf4llm
from PIL import Image
import pytesseract
import io

class DocumentExtractor:
    def __init__(self, tesseract_cmd: str = None):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def is_text_readable(self, text: str) -> bool:
        """
        Test de lisibilité : Vérifie si le texte natif a suffisamment de caractères valides.
        """
        if not text:
            return False
        
        # Nettoyage basique pour le test
        clean_text = re.sub(r'\s+', '', text)
        if len(clean_text) < 100:  # Arbitraire : un CV a généralement plus de 100 caractères
            return False
            
        return True

    def sanitize_text(self, text: str) -> str:
        """
        Nettoyage (Sanitization) : Suppression des artefacts, doubles espaces, caractères invisibles.
        """
        # Remplace les caractères invisibles ou nuls par des espaces
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', ' ', text)
        # Remplace les espaces multiples par un seul
        text = re.sub(r'[ \t]+', ' ', text)
        # Remplace les retours à la ligne multiples par un double retour à la ligne max
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def extract_with_ocr(self, pdf_bytes: bytes) -> str:
        """
        Mode dégradé (Fallback OCR) via pytesseract avec un DPI de 300.
        """
        text_pages = []
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        for page in doc:
            # Rendu de la page à 300 DPI pour Tesseract
            pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            
            # OCR
            text = pytesseract.image_to_string(img, lang="fra+eng")
            text_pages.append(text)
            
        return "\n".join(text_pages)

    def process_pdf(self, pdf_bytes: bytes) -> tuple[str, bool]:
        """
        Extraction native (pymupdf4llm) puis fallback (pytesseract) si échec.
        Retourne (texte_extrait, is_ocr_fallback).
        """
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # Extraction native Markdown
        native_text = pymupdf4llm.to_markdown(doc)
        
        if self.is_text_readable(native_text):
            return self.sanitize_text(native_text), False
            
        # Fallback OCR si le texte n'est pas lisible (ex: CV scanné)
        ocr_text = self.extract_with_ocr(pdf_bytes)
        return self.sanitize_text(ocr_text), True

    def get_first_page_as_base64(self, pdf_bytes: bytes) -> str:
        """
        Rend la première page du PDF sous forme d'image PNG en Base64.
        DPI réduit (Matrix 1.5) pour économiser les tokens Vision.
        """
        import base64
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if len(doc) == 0:
            return ""
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img_bytes = pix.tobytes("png")
        return base64.b64encode(img_bytes).decode("utf-8")

extractor = DocumentExtractor()
