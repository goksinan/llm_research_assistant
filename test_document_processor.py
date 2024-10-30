#!/usr/bin/env python3
"""
Test script for DocumentProcessor implementation.
"""
from rich.console import Console
from rich.table import Table
from pathlib import Path
import sys

from document_processor import DocumentProcessor

console = Console()

def test_process_pdf(pdf_path: str):
    """Process a PDF file and display the results."""
    console.print(f"\n[bold blue]Processing PDF:[/] {pdf_path}")
    
    try:
        # Initialize processor
        processor = DocumentProcessor()
        
        # Process the PDF
        with console.status("[bold green]Processing document..."):
            chunks, metadata = processor.process_pdf(pdf_path)
        
        # Display metadata
        console.print("\n[bold]Document Metadata:[/]")
        console.print(f"Title: {metadata.title}")
        console.print(f"Authors: {', '.join(metadata.authors) if metadata.authors else 'Unknown'}")
        console.print(f"Date: {metadata.date or 'Unknown'}")
        console.print(f"Pages: {metadata.num_pages}")
        console.print(f"File size: {metadata.file_size / 1024 / 1024:.2f}MB")
        
        # Display chunk statistics
        stats = processor.get_document_stats(chunks)
        console.print(f"\n[bold]Processing Statistics:[/]")
        console.print(f"Total chunks: {stats['total_chunks']}")
        console.print(f"Average chunk size: {stats['avg_chunk_size']:.0f} characters")
        
        # Display sections found
        if stats['sections']:
            console.print("\n[bold]Detected Sections:[/]")
            for section in stats['sections']:
                console.print(f"- {section}")
        
        # Display sample chunks in a table
        table = Table(title="\nSample Chunks (first 3)")
        table.add_column("Page", justify="right", style="cyan")
        table.add_column("Section", style="magenta")
        table.add_column("Content Preview", style="green")
        
        for chunk in chunks[:3]:
            content_preview = chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
            table.add_row(
                str(chunk.page_number),
                chunk.section or "N/A",
                content_preview
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     console.print("[bold red]Usage: python test_processor.py <path_to_pdf>[/]")
    #     sys.exit(1)
        
    # pdf_path = sys.argv[1]
    # success = test_process_pdf(pdf_path)
    # sys.exit(0 if success else 1)
    pdf_path = "What Every Programmer Should Know About Memory.pdf"
    test_process_pdf(pdf_path)