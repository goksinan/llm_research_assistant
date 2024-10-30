"""
Main CLI interface for the research assistant.
Provides commands for document ingestion, chat interaction,
and research paper management.
"""
import os
from pathlib import Path
from typing import List, Optional, Tuple
import logging
from datetime import datetime

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from document_processor import DocumentProcessor
from conversation import ConversationManager
from recommendations import PaperRecommender
from storage import StorageManager
from llm import LLMManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Rich console
console = Console()

class ResearchAssistant:
    """Manages the research assistant's components and state."""
    
    def __init__(self):
        """Initialize all required components."""
        try:
            # Initialize core components
            self.storage = StorageManager()
            self.llm = LLMManager()
            self.doc_processor = DocumentProcessor()
            self.conversation = ConversationManager(self.llm, self.storage)
            self.recommender = PaperRecommender()
            
            # Validate components
            self._validate_setup()
            
        except Exception as e:
            console.print(f"[red]Error initializing research assistant: {str(e)}[/]")
            raise click.Abort()
            
    def _validate_setup(self):
        """Validate that all components are properly initialized."""
        # Check LLM API keys
        if not self.llm.validate_api_key():
            raise click.UsageError(
                "LLM API key not found or invalid. Please set the appropriate "
                "environment variable (OPENAI_API_KEY or ANTHROPIC_API_KEY)."
            )
            
        # Check storage directory
        storage_dir = Path("./data/chroma")
        storage_dir.mkdir(parents=True, exist_ok=True)

@click.group()
@click.version_option()
def cli():
    """Research Paper Analysis CLI
    
    A tool for analyzing and exploring academic papers through natural conversation.
    """
    pass

@cli.command()
@click.argument('pdf_paths', type=click.Path(exists=True), nargs=-1)
@click.option('--batch', is_flag=True, help="Process multiple PDFs in batch mode")
def ingest(pdf_paths: Tuple[str], batch: bool):
    """Ingest one or more PDF documents for analysis.
    
    Args:
        pdf_paths: One or more paths to PDF files
        batch: Enable batch processing mode
    """
    if not pdf_paths:
        console.print("[red]Error: No PDF files specified[/]")
        return
        
    try:
        assistant = ResearchAssistant()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for pdf_path in pdf_paths:
                task = progress.add_task(
                    f"Processing {Path(pdf_path).name}...",
                    total=None
                )
                
                try:
                    # Process the PDF
                    chunks, metadata = assistant.doc_processor.process_pdf(pdf_path)
                    
                    # Store the processed chunks
                    chunk_ids = assistant.storage.store_document(chunks)
                    
                    progress.remove_task(task)
                    console.print(
                        f"[green]Successfully processed[/] {metadata.title}\n"
                        f"  Pages: {metadata.num_pages}\n"
                        f"  Chunks: {len(chunks)}\n"
                        f"  Size: {metadata.file_size / 1024 / 1024:.1f}MB"
                    )
                    
                except Exception as e:
                    progress.remove_task(task)
                    console.print(f"[red]Error processing {pdf_path}: {str(e)}[/]")
                    if not batch:
                        raise click.Abort()
                        
        # Show collection stats
        stats = assistant.storage.get_collection_stats()
        console.print(
            f"\n[bold]Collection Statistics:[/]\n"
            f"Total documents: {stats['unique_documents']}\n"
            f"Total chunks: {stats['total_chunks']}"
        )
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/]")
        raise click.Abort()

@cli.command()
@click.option('--document', '-d', multiple=True, help="Focus on specific document(s)")
def chat(document: List[str]):
    """Start an interactive chat session for paper analysis."""
    try:
        assistant = ResearchAssistant()
        
        # Set active documents if specified
        if document:
            assistant.conversation.set_active_documents(list(document))
            console.print(
                f"[cyan]Focusing on documents:[/] "
                f"{', '.join(document)}\n"
            )
        
        console.print(
            "[bold]Research Assistant Chat[/]\n"
            "Type your questions about the papers. Use /help for commands.\n"
        )
        
        while True:
            # Get user input
            query = Prompt.ask("\n[bold cyan]You[/]")
            
            # Handle special commands
            if query.startswith('/'):
                if not _handle_command(query, assistant):
                    continue
            
            # Process regular query
            try:
                with console.status("[bold green]Thinking..."):
                    response, metadata = assistant.conversation.process_query(query)
                
                # Display response with source attribution
                console.print("\n[bold green]Assistant[/]")
                console.print(Markdown(response))
                
                # Show source documents if available
                if metadata.get('source_documents'):
                    console.print(
                        "\n[dim]Sources: " +
                        ", ".join(metadata['source_documents']) +
                        "[/]"
                    )
                
                # Offer paper recommendations if relevant
                if _should_recommend_papers(query, response):
                    _show_recommendations(assistant, query)
                    
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/]")
                
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/]")
        raise click.Abort()

def _handle_command(command: str, assistant: ResearchAssistant) -> bool:
    """Handle special CLI commands.
    
    Returns:
        bool: True if should continue chat, False if should exit
    """
    cmd = command.lower()
    
    if cmd == '/help':
        console.print(
            "\n[bold]Available Commands:[/]\n"
            "/help - Show this help message\n"
            "/documents - List active documents\n"
            "/stats - Show conversation statistics\n"
            "/clear - Clear conversation history\n"
            "/focus <doc> - Focus on specific document\n"
            "/quit - Exit the chat"
        )
        
    elif cmd == '/documents':
        stats = assistant.storage.get_collection_stats()
        console.print(f"\n[bold]Available Documents:[/]")
        for doc in assistant.conversation.context.active_documents:
            console.print(f"- {doc}")
        console.print(f"\nTotal documents: {stats['unique_documents']}")
        
    elif cmd == '/stats':
        stats = assistant.conversation.get_conversation_summary()
        console.print(
            f"\n[bold]Conversation Statistics:[/]\n"
            f"Messages: {stats['message_count']}\n"
            f"Active documents: {len(stats['active_documents'])}\n"
            f"Context usage: {stats['context_window_usage']} tokens"
        )
        
    elif cmd == '/clear':
        if Confirm.ask("Clear conversation history?"):
            assistant.conversation.clear_history()
            console.print("[green]Conversation history cleared[/]")
            
    elif cmd.startswith('/focus '):
        doc_title = command[7:].strip()
        assistant.conversation.set_active_documents([doc_title])
        console.print(f"[green]Now focusing on: {doc_title}[/]")
        
    elif cmd == '/quit':
        if Confirm.ask("Exit chat?"):
            console.print("[cyan]Goodbye![/]")
            return False
            
    else:
        console.print(f"[red]Unknown command: {command}[/]")
        
    return True

def _should_recommend_papers(query: str, response: str) -> bool:
    """Determine if paper recommendations would be helpful."""
    # Look for indicators that recommendations might be helpful
    recommendation_triggers = [
        'related work',
        'similar papers',
        'other papers',
        'further reading',
        'state of the art',
        'recent advances'
    ]
    
    return any(trigger in query.lower() or trigger in response.lower() 
              for trigger in recommendation_triggers)

def _show_recommendations(assistant: ResearchAssistant, query: str):
    """Show relevant paper recommendations."""
    try:
        # Get recommendations based on conversation context
        context = {
            'current_topic': query,
            'keywords': assistant.conversation.context.active_documents
        }
        
        recommendations = assistant.recommender.get_recommendations(
            context,
            filter_criteria={'date_range': {'start': '2020-01-01'}}
        )
        
        if recommendations:
            console.print("\n[bold]📚 Related Papers:[/]")
            for i, paper in enumerate(recommendations[:3], 1):
                console.print(
                    f"\n{i}. [bold]{paper.title}[/]\n"
                    f"   Authors: {', '.join(paper.authors)}\n"
                    f"   [dim]{paper.url}[/]"
                )
                
    except Exception as e:
        logger.warning(f"Error getting recommendations: {str(e)}")

if __name__ == '__main__':
    cli()