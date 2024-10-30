"""
Manages document storage and vector embeddings using ChromaDB.
Provides functionality for storing, retrieving, and searching document chunks
with their associated metadata and embeddings.
"""
from typing import List, Dict, Optional, Any, Tuple
import logging
from datetime import datetime
import json
import hashlib
import os

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np

from document_processor import DocumentChunk, DocumentMetadata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StorageManager:
    """Manages document storage and retrieval using ChromaDB."""
    
    def __init__(
        self,
        persist_directory: str = "./data/chroma",
        collection_name: str = "research_papers",
        embedding_function: Optional[Any] = None
    ):
        """
        Initialize the storage manager.
        
        Args:
            persist_directory: Directory for ChromaDB persistence
            collection_name: Name of the ChromaDB collection
            embedding_function: Custom embedding function (optional)
        """
        # Ensure persistence directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client with correct persistence settings
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                allow_reset=True,
                anonymized_telemetry=False
            )
        )
        
        # Set up embedding function (default to OpenAI if none provided)
        self.embedding_function = embedding_function or embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv('OPENAI_API_KEY'),
            model_name="text-embedding-ada-002"
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": "Research paper chunks and metadata"}
        )
        
        logger.info(f"Initialized StorageManager with collection: {collection_name}")
        
    def _generate_chunk_id(self, chunk: DocumentChunk) -> str:
        """Generate a unique ID for a document chunk."""
        # Combine relevant fields to create a unique identifier
        unique_string = f"{chunk.metadata.file_path}_{chunk.page_number}_{chunk.chunk_index}"
        return hashlib.sha256(unique_string.encode()).hexdigest()[:16]
    
    def _clean_metadata_value(self, value: Any) -> Any:
        """
        Clean and validate metadata values for ChromaDB compatibility.
        
        Args:
            value: Any metadata value
            
        Returns:
            Cleaned value compatible with ChromaDB
        """
        if value is None:
            return ""  # Convert None to empty string
        elif isinstance(value, (str, int, float, bool)):
            return value
        elif isinstance(value, list):
            # Convert list to JSON string, defaulting to empty list for None
            return json.dumps([str(item) for item in value if item is not None]) if value else "[]"
        elif isinstance(value, datetime):
            # Convert datetime to ISO format string
            return value.isoformat()
        else:
            # Convert any other type to string representation
            return str(value)
        
    def _serialize_metadata(self, metadata: DocumentMetadata) -> Dict[str, Any]:
        """
        Serialize DocumentMetadata to a dictionary with ChromaDB-compatible values.
        
        Args:
            metadata: DocumentMetadata object
            
        Returns:
            Dictionary with cleaned metadata values
        """
        raw_metadata = {
            "title": metadata.title,
            "authors": metadata.authors,
            "date": metadata.date,
            "num_pages": metadata.num_pages,
            "file_path": metadata.file_path,
            "file_size": metadata.file_size,
            "processing_date": metadata.processing_date
        }
        
        # Clean all metadata values
        return {
            key: self._clean_metadata_value(value)
            for key, value in raw_metadata.items()
        }
        
    def store_document(self, chunks: List[DocumentChunk]) -> List[str]:
        """
        Store document chunks with their metadata and generate embeddings.
        
        Args:
            chunks: List of DocumentChunk objects to store
            
        Returns:
            List of generated chunk IDs
            
        Raises:
            ValueError: If chunks list is empty
            RuntimeError: If storage operation fails
        """
        if not chunks:
            raise ValueError("No chunks provided for storage")
            
        try:
            # Prepare data for batch insertion
            chunk_ids = []
            documents = []
            metadatas = []
            
            for chunk in chunks:
                chunk_id = self._generate_chunk_id(chunk)
                
                # Clean the chunk content
                content = chunk.content if chunk.content else ""
                
                # Get base metadata
                metadata = self._serialize_metadata(chunk.metadata)
                
                # Add chunk-specific metadata with cleaning
                chunk_metadata = {
                    "page_number": self._clean_metadata_value(chunk.page_number),
                    "chunk_index": self._clean_metadata_value(chunk.chunk_index),
                    "section": self._clean_metadata_value(chunk.section)
                }
                metadata.update(chunk_metadata)
                
                chunk_ids.append(chunk_id)
                documents.append(content)
                metadatas.append(metadata)
                
            # Log metadata for debugging
            logger.debug(f"Storing chunks with metadata: {metadatas[0] if metadatas else 'No metadata'}")
            
            # Batch add to collection
            self.collection.add(
                ids=chunk_ids,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Successfully stored {len(chunks)} chunks")
            return chunk_ids
            
        except Exception as e:
            logger.error(f"Error storing chunks: {str(e)}")
            raise RuntimeError(f"Failed to store chunks: {str(e)}") from e
            
    def search_similar(
        self,
        query: str,
        n_results: int = 5,
        filter_criteria: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar documents using query text.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            filter_criteria: Optional filtering criteria
            
        Returns:
            List of similar documents with metadata
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filter_criteria
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': float(results['distances'][0][i]) if 'distances' in results else None
            })
            
        return formatted_results
        
    def get_document_chunks(
        self,
        document_title: str
    ) -> List[Dict]:
        """
        Retrieve all chunks for a specific document.
        
        Args:
            document_title: Title of the document
            
        Returns:
            List of document chunks with metadata
        """
        results = self.collection.get(
            where={"title": document_title}
        )
        
        # Format results
        chunks = []
        for i in range(len(results['ids'])):
            chunks.append({
                'id': results['ids'][i],
                'content': results['documents'][i],
                'metadata': results['metadatas'][i]
            })
            
        return sorted(chunks, key=lambda x: (x['metadata']['page_number'], x['metadata']['chunk_index']))
        
    def delete_document(self, document_title: str) -> int:
        """
        Delete all chunks associated with a document.
        
        Args:
            document_title: Title of the document
            
        Returns:
            Number of chunks deleted
        """
        chunks = self.get_document_chunks(document_title)
        if chunks:
            self.collection.delete(
                ids=[chunk['id'] for chunk in chunks]
            )
        return len(chunks)
        
    def get_collection_stats(self) -> Dict:
        """Get statistics about the stored documents."""
        all_metadata = self.collection.get()['metadatas']
        return {
            'total_chunks': len(all_metadata),
            'unique_documents': len(set(m['title'] for m in all_metadata)) if all_metadata else 0
        }