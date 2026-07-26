import io
import logging
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our custom modules
from document_parser import extract_text
from phonetic_engine import normalize
from code_mixed import analyze_code_mixing, detect_languages_in_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multilingual IDP API",
    description="Intelligent Document Processing for Code-Mixed & Multilingual text.",
    version="1.0.0"
)

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IDPResponse(BaseModel):
    extraction: Dict[str, Any]
    normalization: Dict[str, Any]
    code_mixing: Dict[str, Any]


@app.get("/")
def read_root():
    return {"message": "Welcome to the Multilingual IDP System API. Use /docs to test endpoints."}


@app.post("/api/process-document", response_model=IDPResponse)
async def process_document(
    file: UploadFile = File(None),
    raw_text: str = Form(None)
):
    """
    Process a document or raw text through the full IDP pipeline:
    1. OCR / Text Extraction
    2. Phonetic Normalization
    3. Code-Mixed Language Analysis
    """
    if not file and not raw_text:
        raise HTTPException(status_code=400, detail="Must provide either a file or raw_text")

    try:
        # 1. Extraction
        if file:
            file_bytes = await file.read()
            extraction_result = extract_text(
                file_bytes=file_bytes,
                filename=file.filename,
                mime_type=file.content_type
            )
        else:
            extraction_result = extract_text(
                file_bytes=None,
                raw_text=raw_text
            )

        extracted_text = extraction_result.text

        # 2. Normalization
        # We normalize the text to handle variations like "wanakkam" -> "vanakkam"
        norm_result = normalize(extracted_text)
        normalized_text = norm_result["normalized_text"]

        # 3. Code-Mixing Analysis
        # Analyze the normalized text for languages and code-mixing index
        cm_analysis = analyze_code_mixing(normalized_text)
        cm_languages = detect_languages_in_text(normalized_text)

        # Merge languages info into the CM response
        cm_analysis["detected_languages_summary"] = cm_languages

        return IDPResponse(
            extraction=extraction_result.to_dict(),
            normalization=norm_result,
            code_mixing=cm_analysis
        )

    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # To run locally: python main.py
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
