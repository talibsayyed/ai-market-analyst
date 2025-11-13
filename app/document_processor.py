from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Handles document loading and chunking with optimized strategy.
    
    Design Decision - Chunking Strategy:
    - Chunk Size: 500 characters
      Rationale: Balances context preservation with retrieval precision.
      Too large chunks (>1000) may include irrelevant info, reducing retrieval accuracy.
      Too small chunks (<300) may lose context and require more retrievals.
      
    - Overlap: 50 characters (10%)
      Rationale: Ensures important information at chunk boundaries isn't lost.
      10% overlap is optimal - more than 20% creates redundancy, less than 5% risks context loss.
      
    - Separator Strategy: Recursive splitting by ["\n\n", "\n", ". ", " "]
      Rationale: Preserves natural document structure and sentence boundaries,
      improving semantic coherence of chunks.
    """
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            is_separator_regex=False
        )
        
        logger.info(f"DocumentProcessor initialized with chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def load_document(self, file_path: str) -> str:
        """Load document from file path."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Loaded document from {file_path}, length: {len(content)} chars")
            return content
        except Exception as e:
            logger.error(f"Error loading document: {e}")
            raise
    
    def process_document(self, content: str, metadata: dict = None) -> List[Document]:
        """
        Split document into chunks with metadata.
        
        Args:
            content: Raw document text
            metadata: Optional metadata to attach to all chunks
            
        Returns:
            List of Document objects with text and metadata
        """
        if metadata is None:
            metadata = {}
        
        # Create chunks
        chunks = self.text_splitter.split_text(content)
        
        # Convert to Document objects with metadata
        documents = [
            Document(
                page_content=chunk,
                metadata={
                    **metadata,
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk)
                }
            )
            for i, chunk in enumerate(chunks)
        ]
        
        logger.info(f"Processed document into {len(documents)} chunks")
        return documents
    
    def process_file(self, file_path: str, metadata: dict = None) -> List[Document]:
        """Load and process document file in one step."""
        content = self.load_document(file_path)
        if metadata is None:
            metadata = {"source": file_path}
        return self.process_document(content, metadata)