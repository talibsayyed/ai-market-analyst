from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Keys
    openai_api_key: str
    anthropic_api_key: Optional[str] = None
    
    # Model configurations
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4-turbo-preview"
    temperature: float = 0.1
    
    # Chunking configuration
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # Vector store
    vector_store_path: str = "./chroma_db"
    collection_name: str = "innovate_inc_report"
    
    # Retrieval
    top_k_results: int = 4
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()