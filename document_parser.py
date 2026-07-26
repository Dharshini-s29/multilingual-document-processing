"""
document_parser.py - Document Text Extraction Layer
Supports: PDF (native + scanned), Images (JPG/PNG), Raw text
Uses pdfplumber for digital PDFs, pytesseract for OCR.
"""
import io
import os
import re
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy imports for optional heavy dependencies
def _import_pdfplumber():
    try:
        import pdfplumber
        return pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed. PDF parsing unavailable.")
        return None

def _import_pytesseract():
    try:
        import pytesseract
        return pytesseract
    except ImportError:
        logger.warning("pytesseract not installed. OCR unavailable.")
        return None

def _import_pil():
    try:
        from PIL import Image
        return Image
    except ImportError:
        logger.warning("Pillow not installed. Image processing unavailable.")
        return None

def _import_pdf2image():
    try:
        from pdf2image import convert_from_bytes
        return convert_from_bytes
    except ImportError:
        logger.warning("pdf2image not installed. Scanned PDF OCR unavailable.")
        return None


@dataclass
class ExtractionResult:
    text: str
    source_type: str        # "pdf_native", "pdf_ocr", "image_ocr", "raw_text"
    page_count: int = 1
    confidence: float = 1.0
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source_type": self.source_type,
            "page_count": self.page_count,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


def extract_from_pdf(file_bytes: bytes, filename: str = "document.pdf") -> ExtractionResult:
    """
    Extract text from PDF. Tries native extraction first,
    falls back to OCR for scanned pages.
    """
    pdfplumber = _import_pdfplumber()
    if pdfplumber is None:
        return ExtractionResult(
            text="[ERROR] pdfplumber not installed",
            source_type="error", confidence=0.0
        )

    pages_text = []
    ocr_pages = []
    total_pages = 0

    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            total_pages = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                text = text.strip()
                if len(text) > 20:  # meaningful text found
                    pages_text.append(text)
                else:
                    ocr_pages.append(i)
                    pages_text.append("")  # placeholder
    except Exception as e:
        logger.error(f"PDF parsing error: {e}")
        return ExtractionResult(
            text=f"[ERROR] Failed to parse PDF: {str(e)}",
            source_type="error", confidence=0.0
        )

    # OCR fallback for pages with no extractable text
    if ocr_pages:
        convert_from_bytes = _import_pdf2image()
        pytesseract = _import_pytesseract()
        if convert_from_bytes and pytesseract:
            try:
                images = convert_from_bytes(file_bytes, dpi=300)
                for page_idx in ocr_pages:
                    if page_idx < len(images):
                        ocr_text = pytesseract.image_to_string(
                            images[page_idx],
                            lang="eng+hin+tam+tel",
                            config="--oem 3 --psm 6"
                        )
                        pages_text[page_idx] = ocr_text.strip()
            except Exception as e:
                logger.warning(f"OCR fallback failed: {e}")

    full_text = "\n\n".join(pages_text).strip()
    source = "pdf_ocr" if ocr_pages else "pdf_native"

    return ExtractionResult(
        text=full_text if full_text else "[No text extracted]",
        source_type=source,
        page_count=total_pages,
        confidence=0.85 if ocr_pages else 0.95,
        metadata={"filename": filename, "ocr_pages": ocr_pages},
    )


def extract_from_image(file_bytes: bytes, filename: str = "image.png") -> ExtractionResult:
    """Extract text from an image using Tesseract OCR."""
    Image = _import_pil()
    pytesseract = _import_pytesseract()

    if Image is None or pytesseract is None:
        return ExtractionResult(
            text="[ERROR] Pillow or pytesseract not installed",
            source_type="error", confidence=0.0
        )

    try:
        img = Image.open(io.BytesIO(file_bytes))
        # Multi-language OCR for Indian documents
        text = pytesseract.image_to_string(
            img, lang="eng+hin+tam+tel",
            config="--oem 3 --psm 6"
        )
        # Get confidence data
        try:
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            confidences = [int(c) for c in data["conf"] if int(c) > 0]
            avg_conf = sum(confidences) / len(confidences) / 100 if confidences else 0.5
        except Exception:
            avg_conf = 0.7

        return ExtractionResult(
            text=text.strip() if text.strip() else "[No text detected in image]",
            source_type="image_ocr",
            confidence=round(avg_conf, 2),
            metadata={"filename": filename, "image_size": img.size},
        )
    except Exception as e:
        logger.error(f"Image OCR error: {e}")
        return ExtractionResult(
            text=f"[ERROR] OCR failed: {str(e)}",
            source_type="error", confidence=0.0
        )


def extract_from_raw_text(text: str) -> ExtractionResult:
    """Wrap raw text input into ExtractionResult."""
    cleaned = text.strip()
    return ExtractionResult(
        text=cleaned if cleaned else "[Empty input]",
        source_type="raw_text",
        confidence=1.0,
        metadata={"char_count": len(cleaned)},
    )


def extract_text(file_bytes: Optional[bytes], filename: str = "",
                 mime_type: str = "", raw_text: str = "") -> ExtractionResult:
    """
    Router: choose extraction method based on input type.
    """
    # Raw text input
    if raw_text:
        return extract_from_raw_text(raw_text)

    if file_bytes is None:
        return ExtractionResult(text="[ERROR] No input provided",
                                source_type="error", confidence=0.0)

    # Determine type from mime or extension
    ext = Path(filename).suffix.lower() if filename else ""
    if mime_type.startswith("application/pdf") or ext == ".pdf":
        return extract_from_pdf(file_bytes, filename)
    elif mime_type.startswith("image/") or ext in (".jpg", ".jpeg", ".png", ".tiff", ".bmp"):
        return extract_from_image(file_bytes, filename)
    else:
        # Try as raw text
        try:
            text = file_bytes.decode("utf-8")
            return extract_from_raw_text(text)
        except UnicodeDecodeError:
            return ExtractionResult(
                text="[ERROR] Unsupported file format",
                source_type="error", confidence=0.0
            )
