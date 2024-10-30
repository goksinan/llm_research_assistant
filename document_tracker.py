"""
Utility class to track and verify document ingestion status.
Provides functionality to check document availability and status
in the storage system.
"""
from pathlib import Path
from typing import List, Optional, Set
import logging

from storage import StorageManager
from document_processor import DocumentMetadata

class DocumentTracker:
    """Tracks and verifies document ingestion status."""
    
    def __init__(self, storage_manager: StorageManager):
        """
        Initialize document tracker.
        
        Args:
            storage_manager: StorageManager instance to verify document status
        """
        self.storage = storage_manager
        self.logger = logging.getLogger(__name__)
        
    def verify_documents(self, document_titles: Optional[List[str]] = None) -> dict:
        """
        Verify the status of specified or all documents in storage.
        
        Args:
            document_titles: Optional list of document titles to verify
            
        Returns:
            Dictionary containing verification results
        """
        try:
            # Get all documents if none specified
            if not document_titles:
                collection_stats = self.storage.get_collection_stats()
                if collection_stats['total_chunks'] == 0:
                    return {
                        'status': 'empty',
                        'message': 'No documents found in storage. Please ingest documents first using the `ingest` command.',
                        'available_documents': []
                    }
            
            # Get all available documents
            available_docs = set()
            for metadata in self.storage.collection.get()['metadatas']:
                if metadata.get('title'):
                    available_docs.add(metadata['title'])
            
            if document_titles:
                # Verify specific documents
                missing_docs = [title for title in document_titles if title not in available_docs]
                if missing_docs:
                    return {
                        'status': 'partial',
                        'message': f"Some specified documents not found: {', '.join(missing_docs)}",
                        'available_documents': list(available_docs),
                        'missing_documents': missing_docs
                    }
            
            return {
                'status': 'ready',
                'message': 'Documents verified and ready',
                'available_documents': list(available_docs)
            }
            
        except Exception as e:
            self.logger.error(f"Error verifying documents: {str(e)}")
            return {
                'status': 'error',
                'message': f"Error verifying documents: {str(e)}",
                'available_documents': []
            }

    def get_available_documents(self) -> List[str]:
        """Get list of all available document titles."""
        try:
            available_docs = set()
            for metadata in self.storage.collection.get()['metadatas']:
                if metadata.get('title'):
                    available_docs.add(metadata['title'])
            return list(available_docs)
        except Exception as e:
            self.logger.error(f"Error getting available documents: {str(e)}")
            return []