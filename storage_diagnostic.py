"""
Diagnostic tool for verifying ChromaDB storage setup and data persistence.
"""
import os
from pathlib import Path
import click
from rich.console import Console
from rich.table import Table

from storage import StorageManager
from document_processor import DocumentProcessor

console = Console()

@click.command()
@click.option('--storage-dir', default='./data/chroma', help='ChromaDB storage directory')
def diagnose_storage(storage_dir: str):
    """Run diagnostics on the ChromaDB storage setup."""
    console.print("\n[bold]Running ChromaDB Storage Diagnostics[/]\n")
    
    try:
        # Create StorageManager instance
        storage = StorageManager(persist_directory=storage_dir)
        
        # Verify storage status
        status = storage.verify_storage()
        
        # Display results in a table
        table = Table(title="Storage Diagnostic Results")
        table.add_column("Check", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="yellow")
        
        table.add_row(
            "Storage Directory",
            "✓" if status["directory_exists"] else "✗",
            status["directory_path"]
        )
        
        table.add_row(
            "Directory Writable",
            "✓" if status["directory_writable"] else "✗",
            "Directory can be written to" if status["directory_writable"] 
            else "Permission denied"
        )
        
        table.add_row(
            "ChromaDB Files",
            "✓" if status["chroma_files"] else "✗",
            f"Found {len(status['chroma_files'])} files" if status["chroma_files"]
            else "No files found"
        )
        
        table.add_row(
            "Document Count",
            "✓" if status["collection_count"] > 0 else "✗",
            f"{status['collection_count']} documents found"
        )
        
        console.print(table)
        
        # Show ChromaDB files if they exist
        if status["chroma_files"]:
            console.print("\n[bold]ChromaDB Files:[/]")
            for file in status["chroma_files"]:
                console.print(f"- {file}")
        
        # Provide recommendations
        console.print("\n[bold]Recommendations:[/]")
        if not status["directory_exists"]:
            console.print("[red]- Create the storage directory[/]")
        if not status["directory_writable"]:
            console.print("[red]- Fix directory permissions[/]")
        if not status["chroma_files"]:
            console.print("[red]- Verify ChromaDB initialization[/]")
        if status["collection_count"] == 0:
            console.print("[red]- Reingest documents[/]")
            
    except Exception as e:
        console.print(f"[red]Error during diagnostics: {str(e)}[/]")

if __name__ == "__main__":
    diagnose_storage()