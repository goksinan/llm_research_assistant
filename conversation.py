"""
Manages conversation context and LLM interactions.
Handles conversation history, context management, and integration with LLM and storage components.
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime
import json

from llm import LLMManager, LLMResponse
from storage import StorageManager
from document_processor import DocumentChunk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConversationContext:
    """Stores the current conversation context."""
    history: List[Dict[str, str]]  # List of message dictionaries
    active_documents: List[str]    # List of active document titles
    last_query_time: datetime
    context_window_size: int = 4000  # Default context window size in tokens

class ConversationManager:
    """Manages conversation flow and context."""
    
    def __init__(
        self,
        llm_manager: LLMManager,
        storage_manager: StorageManager,
        context_window_size: int = 4000
    ):
        """
        Initialize the conversation manager.
        
        Args:
            llm_manager: LLMManager instance for handling LLM interactions
            storage_manager: StorageManager instance for document retrieval
            context_window_size: Maximum size of context window in tokens
        """
        self.llm_manager = llm_manager
        self.storage_manager = storage_manager
        self.context = ConversationContext(
            history=[],
            active_documents=[],
            last_query_time=datetime.now(),
            context_window_size=context_window_size
        )
    
    def process_query(self, query: str) -> Tuple[str, Dict]:
        """
        Process a user query and return a response with metadata.
        
        Args:
            query: User's input query
            
        Returns:
            Tuple of (response text, response metadata)
            
        Raises:
            RuntimeError: If query processing fails
        """
        try:
            # Update conversation context
            self._update_context(query)
            
            # Retrieve relevant document chunks
            relevant_chunks = self._get_relevant_chunks(query)
            
            # Construct prompt with context
            prompt = self._construct_prompt(query, relevant_chunks)
            
            # Generate response using LLM
            llm_response = self.llm_manager.generate_response(
                prompt=prompt,
                context=self._get_llm_context(),
                temperature=0.7
            )
            
            # Update conversation history with response
            self._add_to_history("assistant", llm_response.content)
            
            # Prepare response metadata
            metadata = {
                "used_chunks": len(relevant_chunks),
                "source_documents": list(set(chunk["metadata"]["title"] for chunk in relevant_chunks)),
                "token_usage": llm_response.usage,
                "model": llm_response.model
            }
            
            return llm_response.content, metadata
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise RuntimeError(f"Failed to process query: {str(e)}") from e
    
    def _update_context(self, query: str):
        """Update conversation context with new query."""
        self._add_to_history("user", query)
        self.context.last_query_time = datetime.now()
    
    def _add_to_history(self, role: str, content: str):
        """Add a message to conversation history."""
        self.context.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self._prune_history()
    
    def _prune_history(self):
        """
        Prune conversation history to maintain context window size.
        Removes oldest messages while keeping essential context.
        """
        total_tokens = sum(
            self.llm_manager.get_token_estimate(msg["content"])
            for msg in self.context.history
        )
        
        while (total_tokens > self.context.context_window_size and 
               len(self.context.history) > 2):  # Keep at least the last exchange
            # Remove oldest message
            removed_msg = self.context.history.pop(0)
            total_tokens -= self.llm_manager.get_token_estimate(
                removed_msg["content"]
            )
    
    def _get_relevant_chunks(self, query: str) -> List[Dict]:
        """
        Retrieve relevant document chunks based on query.
        
        Args:
            query: User's input query
            
        Returns:
            List of relevant document chunks with metadata
        """
        # Get similar chunks from storage
        chunks = self.storage_manager.search_similar(
            query=query,
            n_results=5,  # Adjust based on context window size
            filter_criteria=(
                {"title": {"$in": self.context.active_documents}}
                if self.context.active_documents
                else None
            )
        )
        
        return chunks
    
    def _construct_prompt(self, query: str, relevant_chunks: List[Dict]) -> str:
        """
        Construct a prompt combining query and relevant context.
        
        Args:
            query: User's input query
            relevant_chunks: List of relevant document chunks
            
        Returns:
            Constructed prompt string
        """
        # Base system prompt
        prompt = """You are a research assistant helping to analyze academic papers. 
        Use the provided context to answer questions accurately and concisely. 
        If you're unsure or if the context doesn't contain relevant information, 
        say so explicitly.\n\nContext:\n"""
        
        # Add relevant chunks with source information
        for chunk in relevant_chunks:
            prompt += f"\nFrom '{chunk['metadata']['title']}' (Page {chunk['metadata']['page_number']}):\n"
            prompt += chunk['content'] + "\n"
        
        # Add the user's query
        prompt += f"\nQuestion: {query}\n\nAnswer: "
        
        return prompt
    
    def _get_llm_context(self) -> Dict:
        """Get context for LLM request."""
        return {
            "history": [
                {
                    "role": msg["role"],
                    "content": msg["content"]
                }
                for msg in self.context.history[-3:]  # Last 3 messages for context
            ]
        }
    
    def set_active_documents(self, document_titles: List[str]):
        """
        Set the active documents for the conversation.
        
        Args:
            document_titles: List of document titles to set as active
        """
        self.context.active_documents = document_titles
    
    def get_conversation_summary(self) -> Dict:
        """Get a summary of the current conversation state."""
        return {
            "message_count": len(self.context.history),
            "active_documents": self.context.active_documents,
            "last_query_time": self.context.last_query_time.isoformat(),
            "context_window_usage": sum(
                self.llm_manager.get_token_estimate(msg["content"])
                for msg in self.context.history
            )
        }
    
    def clear_history(self):
        """Clear conversation history while maintaining active documents."""
        self.context.history = []
        logger.info("Conversation history cleared")

if __name__ == "__main__":
    # Example usage
    from llm import LLMManager
    from storage import StorageManager
    
    # Initialize components
    llm = LLMManager()
    storage = StorageManager()
    
    # Create conversation manager
    conversation = ConversationManager(llm, storage)
    
    # Set active documents
    conversation.set_active_documents(["Sample Research Paper"])
    
    # Test query
    try:
        response, metadata = conversation.process_query(
            "What are the main findings of the paper?"
        )
        print(f"Response: {response}\n")
        print("Metadata:", json.dumps(metadata, indent=2))
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")