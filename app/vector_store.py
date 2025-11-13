from typing import List, Optional
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    Manages vector store operations for document retrieval.
    
    Design Decision - Vector Database Choice: ChromaDB
    Rationale:
    1. Lightweight: No separate server required, perfect for development and deployment
    2. Performance: Fast similarity search with HNSW indexing
    3. Persistence: Built-in persistence to disk
    4. Python-native: Seamless integration with LangChain
    5. Metadata filtering: Supports complex metadata queries
    
    Alternative considered: Pinecone (cloud), FAISS (no persistence), Weaviate (complex setup)
    
    Design Decision - Embedding Model: text-embedding-3-small
    Rationale:
    1. Cost-effective: 5x cheaper than ada-002
    2. Performance: 1536 dimensions, strong on domain-specific content
    3. Speed: Fast inference time (~50ms per batch)
    4. Quality: Outperforms ada-002 on most benchmarks
    
    Alternative models: sentence-transformers (offline), text-embedding-3-large (overkill for this use case)
    """
    
    def __init__(
        self,
        embedding_model: str = None,
        persist_directory: str = None,
        collection_name: str = None
    ):
        self.embedding_model_name = embedding_model or settings.embedding_model
        self.persist_directory = persist_directory or settings.vector_store_path
        self.collection_name = collection_name or settings.collection_name
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model_name,
            openai_api_key=settings.openai_api_key
        )
        
        self.vector_store: Optional[Chroma] = None
        logger.info(f"VectorStoreManager initialized with {self.embedding_model_name}")
    
    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """
        Create a new vector store from documents.
        
        Args:
            documents: List of Document objects to embed and store
            
        Returns:
            Chroma vector store instance
        """
        logger.info(f"Creating vector store with {len(documents)} documents")
        
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )
        
        # Persist to disk
        self.vector_store.persist()
        logger.info(f"Vector store created and persisted to {self.persist_directory}")
        
        return self.vector_store
    
    def load_vector_store(self) -> Chroma:
        """Load existing vector store from disk."""
        logger.info(f"Loading vector store from {self.persist_directory}")
        
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )
        
        return self.vector_store
    
    def similarity_search(
        self,
        query: str,
        k: int = None,
        filter_dict: dict = None
    ) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of most similar documents
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call create_vector_store or load_vector_store first.")
        
        k = k or settings.top_k_results
        
        results = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter_dict
        )
        
        logger.info(f"Similarity search returned {len(results)} results for query: {query[:50]}...")
        return results
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = None
    ) -> List[tuple[Document, float]]:
        """
        Perform similarity search with relevance scores.
        
        Returns:
            List of (Document, score) tuples
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized.")
        
        k = k or settings.top_k_results
        
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=k
        )
        
        return results
    
    def get_retriever(self, search_kwargs: dict = None):
        """Get a retriever interface for the vector store."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized.")
        
        if search_kwargs is None:
            search_kwargs = {"k": settings.top_k_results}
        
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)