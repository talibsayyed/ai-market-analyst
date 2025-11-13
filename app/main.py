from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging
import os

from app.config import settings
from app.document_processor import DocumentProcessor
from app.vector_store import VectorStoreManager
from app.tools import MarketAnalystTools
from app.agent import AutonomousMarketAnalystAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Market Analyst Agent",
    description="Multi-functional AI agent for market research analysis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QueryRequest(BaseModel):
    query: str = Field(..., description="User's natural language query")
    force_tool: Optional[str] = Field(None, description="Force specific tool: qa, summarize, or extract")

class QARequest(BaseModel):
    question: str = Field(..., description="Question about the market research document")

class SummarizeRequest(BaseModel):
    focus_area: str = Field(default="overall", description="Area to focus on for summarization")

class ExtractionRequest(BaseModel):
    extraction_type: str = Field(default="all", description="Type of data to extract")

# Global variables for initialized components
vector_store_manager: Optional[VectorStoreManager] = None
tools: Optional[MarketAnalystTools] = None
agent: Optional[AutonomousMarketAnalystAgent] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup."""
    global vector_store_manager, tools, agent
    
    logger.info("Starting AI Market Analyst Agent...")
    
    try:
        # Initialize components
        vector_store_manager = VectorStoreManager()
        
        # Check if vector store exists, otherwise create it
        vector_store_path = settings.vector_store_path
        data_file = "data/market_report.txt"
        
        if os.path.exists(vector_store_path) and os.listdir(vector_store_path):
            logger.info("Loading existing vector store...")
            vector_store_manager.load_vector_store()
        else:
            logger.info("Creating new vector store from document...")
            
            # Process document
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"Market report not found at {data_file}")
            
            processor = DocumentProcessor()
            documents = processor.process_file(
                data_file,
                metadata={"source": "Innovate Inc. Market Research Report Q3 2025"}
            )
            
            # Create vector store
            vector_store_manager.create_vector_store(documents)
        
        # Initialize tools and agent
        tools = MarketAnalystTools(vector_store_manager)
        agent = AutonomousMarketAnalystAgent(tools)
        
        logger.info("AI Market Analyst Agent successfully initialized!")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AI Market Analyst Agent API",
        "version": "1.0.0",
        "endpoints": {
            "/query": "Autonomous routing - agent decides which tool to use",
            "/qa": "Q&A tool - answer specific questions",
            "/summarize": "Summarization tool - generate summaries",
            "/extract": "Extraction tool - extract structured data",
            "/health": "Health check endpoint"
        },
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "vector_store_initialized": vector_store_manager is not None,
        "tools_initialized": tools is not None,
        "agent_initialized": agent is not None
    }

@app.post("/query")
async def autonomous_query(request: QueryRequest) -> Dict[str, Any]:
    """
    Autonomous query endpoint (Bonus Feature).
    
    The agent automatically determines which tool to use based on the query.
    
    Example queries:
    - "What is Innovate Inc's market share?" → Routes to Q&A
    - "Summarize the competitive landscape" → Routes to Summarize
    - "Extract all competitor information" → Routes to Extract
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        result = agent.process_query(request.query, request.force_tool)
        return result
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/explain")
async def explain_routing(request: QueryRequest) -> Dict[str, Any]:
    """
    Explain how a query would be routed without executing it.
    Useful for understanding the agent's decision-making.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        explanation = agent.explain_routing(request.query)
        return explanation
    except Exception as e:
        logger.error(f"Routing explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/qa")
async def qa_endpoint(request: QARequest) -> Dict[str, Any]:
    """
    Q&A Tool endpoint.
    
    Answers specific questions about the market research document.
    Uses retrieval-augmented generation for accurate, grounded responses.
    """
    if tools is None:
        raise HTTPException(status_code=503, detail="Tools not initialized")
    
    try:
        result = tools.qa_tool(request.question)
        return result
    except Exception as e:
        logger.error(f"Q&A error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def summarize_endpoint(request: SummarizeRequest) -> Dict[str, Any]:
    """
    Summarization Tool endpoint.
    
    Generates comprehensive summaries of market research findings.
    Can focus on specific areas or provide overall summary.
    """
    if tools is None:
        raise HTTPException(status_code=503, detail="Tools not initialized")
    
    try:
        result = tools.summarize_tool(request.focus_area)
        return result
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract")
async def extract_endpoint(request: ExtractionRequest) -> Dict[str, Any]:
    """
    Data Extraction Tool endpoint.
    
    Extracts structured data from the document in JSON format.
    Includes company info, market data, competitors, SWOT analysis, and more.
    """
    if tools is None:
        raise HTTPException(status_code=503, detail="Tools not initialized")
    
    try:
        result = tools.extract_tool(request.extraction_type)
        return result
    except Exception as e:
        logger.error(f"Extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tools/descriptions")
async def get_tool_descriptions() -> Dict[str, str]:
    """Get descriptions of all available tools."""
    if tools is None:
        raise HTTPException(status_code=503, detail="Tools not initialized")
    
    return {
        "qa": tools.get_tool_description("qa"),
        "summarize": tools.get_tool_description("summarize"),
        "extract": tools.get_tool_description("extract")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )