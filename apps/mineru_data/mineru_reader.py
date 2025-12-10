"""
MinerU PDF Reader for LEANN.

This module uses MinerU (magic-pdf) to extract text from PDFs.
MinerU handles:
- Text extraction with layout preservation
- OCR for scanned documents
- Table detection → Markdown tables
- Formula recognition → LaTeX
- Multi-language support

Usage:
    from mineru_data import MinerUReader
    
    reader = MinerUReader()
    documents = reader.load_pdf("paper.pdf")
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from llama_index.core import Document


def extract_pdf_with_mineru(pdf_path: str) -> str:
    """
    Extract text from PDF using MinerU RAG API.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text in markdown format
    """
    try:
        from magic_pdf.integrations.rag.api import RagDocumentReader
    except ImportError:
        raise ImportError(
            "MinerU (magic-pdf) is not installed or not configured.\n"
            "Install with: uv pip install magic-pdf\n"
            "Make sure ~/magic-pdf.json config exists."
        )
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    try:
        # Use MinerU RAG reader
        print(f"  Running MinerU extraction...")
        reader = RagDocumentReader(str(pdf_path))
        
        # Get all text from all pages
        text_parts = []
        for page_idx, page in enumerate(reader):
            page_text = page.get_page_text()
            if page_text and page_text.strip():
                text_parts.append(f"## Page {page_idx + 1}\n\n{page_text}")
        
        if text_parts:
            return "\n\n".join(text_parts)
        else:
            print("  Warning: No text extracted, trying fallback...")
            return _fallback_extract(pdf_path)
            
    except Exception as e:
        print(f"  Warning: MinerU extraction failed ({e}), trying fallback...")
        return _fallback_extract(pdf_path)


def _fallback_extract(pdf_path: Path) -> str:
    """
    Fallback extraction using PyMuPDF if MinerU fails.
    """
    try:
        import fitz  # PyMuPDF
        
        text_parts = []
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            if text.strip():
                text_parts.append(f"## Page {page_num + 1}\n\n{text}")
        
        doc.close()
        return "\n\n".join(text_parts)
        
    except Exception as e:
        raise RuntimeError(f"Both MinerU and PyMuPDF extraction failed: {e}")


class MinerUReader:
    """
    PDF Reader using MinerU for high-quality text extraction.
    
    Example:
        reader = MinerUReader()
        docs = reader.load_pdf("paper.pdf")
        
        # Or load all PDFs from a directory
        docs = reader.load_directory("./papers/")
    """
    
    def __init__(self):
        """Initialize MinerU Reader."""
        self._check_installation()
    
    def _check_installation(self):
        """Check if MinerU is installed and configured."""
        try:
            from magic_pdf.integrations.rag.api import RagDocumentReader
            print("✓ MinerU (magic-pdf) is ready")
        except ImportError as e:
            raise ImportError(
                f"MinerU (magic-pdf) is not properly installed: {e}\n"
                "Install with: uv pip install magic-pdf"
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                "MinerU config not found. Create ~/magic-pdf.json with:\n"
                '{\n'
                '  "device-mode": "cpu",\n'
                '  "latex-delimiter-config": {\n'
                '    "display": {"left": "$$", "right": "$$"},\n'
                '    "inline": {"left": "$", "right": "$"}\n'
                '  }\n'
                '}'
            )
    
    def load_pdf(
        self,
        pdf_path: str,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> list[Document]:
        """
        Load a single PDF file.
        
        Args:
            pdf_path: Path to PDF file
            extra_metadata: Additional metadata to include
            
        Returns:
            List containing one LlamaIndex Document
        """
        pdf_path = Path(pdf_path)
        print(f"📄 Processing: {pdf_path.name}")
        
        # Extract text with MinerU
        markdown_text = extract_pdf_with_mineru(str(pdf_path))
        
        # Build metadata
        metadata = {
            "file_path": str(pdf_path.absolute()),
            "file_name": pdf_path.name,
            "file_type": "pdf",
            "extraction_method": "mineru",
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        
        # Create document
        doc = Document(text=markdown_text, metadata=metadata)
        
        print(f"  ✓ Extracted {len(markdown_text)} characters")
        return [doc]
    
    def load_directory(
        self,
        directory: str,
        recursive: bool = True,
        show_progress: bool = True,
    ) -> list[Document]:
        """
        Load all PDFs from a directory.
        
        Args:
            directory: Directory path
            recursive: Search subdirectories
            show_progress: Show progress bar
            
        Returns:
            List of LlamaIndex Documents
        """
        directory = Path(directory)
        
        # Find PDFs
        if recursive:
            pdf_files = list(directory.rglob("*.pdf"))
        else:
            pdf_files = list(directory.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {directory}")
            return []
        
        print(f"📁 Found {len(pdf_files)} PDF files")
        
        documents = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                pdf_files = tqdm(pdf_files, desc="MinerU extraction")
            except ImportError:
                pass
        
        for pdf_file in pdf_files:
            try:
                docs = self.load_pdf(str(pdf_file))
                documents.extend(docs)
            except Exception as e:
                print(f"  ⚠️ Failed to process {pdf_file.name}: {e}")
                continue
        
        print(f"\n✓ Successfully extracted {len(documents)} documents")
        return documents
