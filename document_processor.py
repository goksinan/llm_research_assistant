"""
Handles PDF ingestion, text extraction, and chunking with source tracking.

This module provides robust PDF processing capabilities including:
- PDF text extraction with structure preservation
- Intelligent text chunking with overlap
- Metadata extraction
- Source tracking for chunks
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime

import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Stores metadata for a processed document."""
    title: str
    authors: List[str]
    date: Optional[datetime]
    num_pages: int
    file_path: str
    file_size: int
    processing_date: datetime = datetime.now()

@dataclass
class DocumentChunk:
    """Represents a chunk of processed document text with source tracking."""
    content: str
    page_number: int
    chunk_index: int
    metadata: DocumentMetadata
    section: Optional[str] = None

class DocumentProcessor:
    """Handles the processing of PDF documents into analyzable chunks."""
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 max_file_size_mb: int = 50):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Target size of text chunks in characters
            chunk_overlap: Number of characters to overlap between chunks
            max_file_size_mb: Maximum allowed file size in megabytes
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def process_pdf(self, pdf_path: str) -> Tuple[List[DocumentChunk], DocumentMetadata]:
        """
        Process a PDF file and return chunked content with metadata.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (list of DocumentChunk objects, DocumentMetadata)
            
        Raises:
            ValueError: If file is invalid or too large
            FileNotFoundError: If PDF file doesn't exist
            RuntimeError: If PDF processing fails
        """
        pdf_path = Path(pdf_path)
        
        # Validate file
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        file_size = pdf_path.stat().st_size
        if file_size > self.max_file_size_bytes:
            raise ValueError(
                f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds maximum allowed size "
                f"({self.max_file_size_bytes / 1024 / 1024:.1f}MB)"
            )

        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                metadata = self._extract_metadata(pdf_reader, pdf_path, file_size)
                chunks = self._process_pages(pdf_reader, metadata)
                
            logger.info(f"Successfully processed {pdf_path.name}: {len(chunks)} chunks created")
            return chunks, metadata
            
        except Exception as e:
            raise RuntimeError(f"Failed to process PDF {pdf_path}: {str(e)}") from e

    def _extract_metadata(self, 
                         pdf_reader: pypdf.PdfReader, 
                         pdf_path: Path,
                         file_size: int) -> DocumentMetadata:
        """Extract metadata from PDF."""
        info = pdf_reader.metadata
        
        # Try to extract title from PDF metadata or use filename
        title = info.get('/Title', pdf_path.stem)
        if not title or title.strip() == '':
            title = pdf_path.stem
            
        # Try to extract authors
        authors = []
        if info.get('/Author'):
            authors = [author.strip() for author in info.get('/Author').split(';')]
            
        # Try to extract date
        date = None
        if info.get('/CreationDate'):
            try:
                # Handle common PDF date format like "D:20240220123456"
                date_str = info.get('/CreationDate').replace('D:', '')[:14]
                date = datetime.strptime(date_str, '%Y%m%d%H%M%S')
            except (ValueError, AttributeError):
                pass
                
        return DocumentMetadata(
            title=title,
            authors=authors,
            date=date,
            num_pages=len(pdf_reader.pages),
            file_path=str(pdf_path),
            file_size=file_size
        )

    def _process_pages(self, 
                      pdf_reader: pypdf.PdfReader, 
                      metadata: DocumentMetadata) -> List[DocumentChunk]:
        """Process all pages in the PDF."""
        chunks = []
        current_section = None
        
        for page_num, page in enumerate(pdf_reader.pages, 1):
            text = page.extract_text()
            
            # Try to detect section headers
            section_match = re.search(r'^(?:[\d.]*\s*)?([A-Z][A-Za-z\s]{2,})(?:\n|$)', text)
            if section_match:
                current_section = section_match.group(1).strip()
            
            # Split page text into chunks
            page_chunks = self.text_splitter.split_text(text)
            
            # Create DocumentChunk objects with source tracking
            for chunk_idx, chunk_text in enumerate(page_chunks):
                chunk = DocumentChunk(
                    content=chunk_text,
                    page_number=page_num,
                    chunk_index=chunk_idx,
                    metadata=metadata,
                    section=current_section
                )
                chunks.append(chunk)
                
        return chunks

    def get_document_stats(self, chunks: List[DocumentChunk]) -> Dict:
        """
        Generate statistics about the processed document.
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            Dictionary containing document statistics
        """
        return {
            'total_chunks': len(chunks),
            'total_pages': chunks[0].metadata.num_pages if chunks else 0,
            'avg_chunk_size': sum(len(c.content) for c in chunks) / len(chunks) if chunks else 0,
            'sections': sorted(set(c.section for c in chunks if c.section))
        }
    

