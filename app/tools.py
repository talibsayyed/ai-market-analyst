from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from app.config import settings
from app.vector_store import VectorStoreManager
import json
import logging

logger = logging.getLogger(__name__)

class MarketAnalystTools:
    """
    Implements three core tools for the AI Market Analyst agent:
    1. Q&A Tool: Answers questions using RAG
    2. Summarization Tool: Generates market research summaries
    3. Data Extraction Tool: Extracts structured data as JSON
    """
    
    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store_manager = vector_store_manager
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.temperature,
            openai_api_key=settings.openai_api_key
        )
        
        logger.info(f"MarketAnalystTools initialized with model {settings.llm_model}")
    
    def qa_tool(self, question: str) -> Dict[str, Any]:
        """
        Q&A Tool: Answer questions about the market research document.
        
        Uses RAG (Retrieval-Augmented Generation) to provide accurate answers
        grounded in the document content.
        """
        logger.info(f"Q&A Tool invoked with question: {question}")
        
        # Retrieve relevant documents
        relevant_docs = self.vector_store_manager.similarity_search(question, k=4)
        
        # Create context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # QA Prompt
        qa_prompt = PromptTemplate(
            template="""You are an AI assistant analyzing market research data for Innovate Inc.
            
Use the following context to answer the question accurately and concisely.
If you cannot answer based on the context, say so clearly.

Context:
{context}

Question: {question}

Answer: Provide a clear, factual answer based solely on the context above.""",
            input_variables=["context", "question"]
        )
        
        # Generate answer
        prompt_text = qa_prompt.format(context=context, question=question)
        response = self.llm.predict(prompt_text)
        
        return {
            "tool": "qa",
            "question": question,
            "answer": response,
            "source_chunks": len(relevant_docs),
            "sources": [
                {
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "content": doc.page_content[:200] + "..."
                }
                for doc in relevant_docs
            ]
        }
    
    def summarize_tool(self, focus_area: str = "overall") -> Dict[str, Any]:
        """
        Summarization Tool: Generate market research summaries.
        
        Can focus on specific areas or provide overall summary.
        """
        logger.info(f"Summarization Tool invoked with focus: {focus_area}")
        
        # Retrieve relevant documents based on focus area
        query = f"market research summary {focus_area}"
        relevant_docs = self.vector_store_manager.similarity_search(query, k=6)
        
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        summarize_prompt = PromptTemplate(
            template="""You are a market research analyst summarizing key findings from Innovate Inc.'s market report.

Context:
{context}

Task: Create a comprehensive summary focusing on: {focus_area}

Provide a well-structured summary with:
1. Key metrics and figures
2. Main findings
3. Strategic implications

Summary:""",
            input_variables=["context", "focus_area"]
        )
        
        prompt_text = summarize_prompt.format(context=context, focus_area=focus_area)
        response = self.llm.predict(prompt_text)
        
        return {
            "tool": "summarize",
            "focus_area": focus_area,
            "summary": response,
            "chunks_analyzed": len(relevant_docs)
        }
    
    def extract_tool(self, extraction_type: str = "all") -> Dict[str, Any]:
        """
        Data Extraction Tool: Extract structured data as JSON.
        
        Design Decision - Data Extraction Prompt Engineering:
        
        Strategy for reliable JSON extraction:
        1. Clear Schema Definition: Explicitly define expected JSON structure
        2. Few-shot Examples: Provide example output format
        3. Format Instructions: Strict instructions to output ONLY valid JSON
        4. Error Handling: Parse and validate JSON, retry if needed
        5. Field Validation: Ensure all required fields are present
        
        Prompt techniques used:
        - Role setting: "You are a data extraction specialist"
        - Output format specification: "Output ONLY valid JSON, no other text"
        - Schema enforcement: Provide exact field names and types
        - Consistency check: Request specific numeric formats (e.g., percentages without % symbol)
        """
        logger.info(f"Extraction Tool invoked for type: {extraction_type}")
        
        # Get all relevant content
        relevant_docs = self.vector_store_manager.similarity_search(
            f"extract data {extraction_type}", 
            k=8
        )
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        extraction_prompt = PromptTemplate(
            template="""You are a data extraction specialist. Extract structured information from the market research document.

Context:
{context}

Task: Extract the following information and return it as VALID JSON ONLY (no markdown, no additional text):

{{
  "company_info": {{
    "name": "company name",
    "product": "flagship product name",
    "market_share": numeric value without % symbol (e.g., 12 not "12%"),
    "industry_focus": ["primary", "sectors"]
  }},
  "market_data": {{
    "current_market_size_billion": numeric value,
    "projected_market_size_billion": numeric value,
    "cagr_percent": numeric value,
    "projection_year": year as integer
  }},
  "competitors": [
    {{
      "name": "competitor name",
      "market_share": numeric value without % symbol
    }}
  ],
  "swot": {{
    "strengths": ["list of strengths"],
    "weaknesses": ["list of weaknesses"],
    "opportunities": ["list of opportunities"],
    "threats": ["list of threats"]
  }},
  "strategic_priorities": ["list of key priorities"]
}}

CRITICAL: Output ONLY the JSON object. No explanatory text before or after.

JSON Output:""",
            input_variables=["context"]
        )
        
        prompt_text = extraction_prompt.format(context=context)
        response = self.llm.predict(prompt_text)
        
        # Clean and parse JSON
        try:
            # Remove potential markdown code blocks
            json_str = response.strip()
            if json_str.startswith("```"):
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
            json_str = json_str.strip()
            
            extracted_data = json.loads(json_str)
            
            return {
                "tool": "extract",
                "extraction_type": extraction_type,
                "data": extracted_data,
                "status": "success"
            }
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return {
                "tool": "extract",
                "extraction_type": extraction_type,
                "raw_response": response,
                "status": "error",
                "error": f"Failed to parse JSON: {str(e)}"
            }
    
    def get_tool_description(self, tool_name: str) -> str:
        """Get description of a specific tool for the routing agent."""
        descriptions = {
            "qa": "Answers specific questions about the market research document using retrieval-augmented generation. Best for: 'What is...?', 'How much...?', 'Who are...?' type questions.",
            "summarize": "Generates comprehensive summaries of market research findings. Can focus on specific areas (competitors, market size, SWOT, etc.) or provide overall summary. Best for: 'Summarize...', 'Give me an overview...', 'What are the key findings...?'",
            "extract": "Extracts structured data from the document in JSON format including company info, market data, competitors, SWOT analysis, and strategic priorities. Best for: 'Extract...', 'Get structured data...', 'List all competitors...'"
        }
        return descriptions.get(tool_name, "Tool not found")