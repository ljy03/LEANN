"""MinerU PDF extraction module for LEANN.

MinerU (magic-pdf) extracts text from PDFs with:
- OCR support for scanned documents
- Table detection and markdown conversion
- Formula recognition (LaTeX)
- Multi-language support (84 languages)
"""

from .mineru_reader import MinerUReader, extract_pdf_with_mineru

__all__ = ["MinerUReader", "extract_pdf_with_mineru"]

