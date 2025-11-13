from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from app.config import settings
from app.tools import MarketAnalystTools
import logging
import json

logger = logging.getLogger(__name__)

class AutonomousMarketAnalystAgent:
    """
    Autonomous agent that routes user queries to appropriate tools.
    
    Bonus Feature: Autonomous Routing
    
    Design: Uses an LLM-based routing mechanism to classify user intent
    and select the most appropriate tool (Q&A, Summarize, or Extract).
    
    Routing Strategy:
    1. Intent Classification: Analyze query semantics and keywords
    2. Tool Selection: Map intent to one of three tools
    3. Parameter Extraction: Extract relevant parameters for the tool
    4. Execution: Call the selected tool with parameters
    5. Response Formatting: Return unified response format
    
    Benefits over manual routing:
    - Better UX: Users don't need to know which tool to use
    - Natural language: Works with conversational queries
    - Flexible: Can handle ambiguous or compound queries
    """
    
    def __init__(self, tools: MarketAnalystTools):
        self.tools = tools
        self.routing_llm = ChatOpenAI(
            model="gpt-3.5-turbo",  # Faster, cheaper for routing
            temperature=0.0,  # Deterministic routing
            openai_api_key=settings.openai_api_key
        )
        
        logger.info("AutonomousMarketAnalystAgent initialized")
    
    def route_query(self, user_query: str) -> str:
        """
        Determine which tool to use based on user query.
        
        Returns: "qa", "summarize", or "extract"
        """
        routing_prompt = PromptTemplate(
            template="""You are a routing assistant for a market research analysis system.

Available tools:
1. "qa" - Answers specific questions about the document (e.g., "What is the market share?", "Who are the competitors?")
2. "summarize" - Provides summaries and overviews (e.g., "Summarize the report", "What are the key findings?")
3. "extract" - Extracts structured data in JSON format (e.g., "Extract all competitors", "Get structured data", "List SWOT analysis")

User Query: {query}

Based on the query, which tool should be used? Respond with ONLY one word: qa, summarize, or extract

Tool:""",
            input_variables=["query"]
        )
        
        prompt_text = routing_prompt.format(query=user_query)
        response = self.routing_llm.predict(prompt_text).strip().lower()
        
        # Validate response
        if response not in ["qa", "summarize", "extract"]:
            # Default to Q&A for ambiguous queries
            logger.warning(f"Invalid routing response: {response}. Defaulting to 'qa'")
            return "qa"
        
        logger.info(f"Query routed to: {response}")
        return response
    
    def extract_parameters(self, user_query: str, tool: str) -> Dict[str, Any]:
        """Extract relevant parameters from query for the selected tool."""
        
        if tool == "summarize":
            # Extract focus area for summarization
            param_prompt = PromptTemplate(
                template="""Extract the focus area from this summarization request.

Query: {query}

Common focus areas: "overall", "competitors", "market size", "SWOT analysis", "growth projections"

If no specific focus is mentioned, respond with "overall".

Focus area:""",
                input_variables=["query"]
            )
            
            prompt_text = param_prompt.format(query=user_query)
            focus_area = self.routing_llm.predict(prompt_text).strip().lower()
            
            return {"focus_area": focus_area}
        
        elif tool == "extract":
            # Determine extraction type
            return {"extraction_type": "all"}
        
        else:  # qa
            return {"question": user_query}
    
    def process_query(self, user_query: str, force_tool: str = None) -> Dict[str, Any]:
        """
        Process user query with autonomous routing or forced tool selection.
        
        Args:
            user_query: Natural language query from user
            force_tool: Optional - force a specific tool ("qa", "summarize", "extract")
            
        Returns:
            Dict with tool output and metadata
        """
        logger.info(f"Processing query: {user_query}")
        
        # Route to appropriate tool
        if force_tool:
            tool = force_tool
            logger.info(f"Using forced tool: {tool}")
        else:
            tool = self.route_query(user_query)
        
        # Extract parameters
        parameters = self.extract_parameters(user_query, tool)
        
        # Execute tool
        try:
            if tool == "qa":
                result = self.tools.qa_tool(parameters["question"])
            elif tool == "summarize":
                result = self.tools.summarize_tool(parameters["focus_area"])
            elif tool == "extract":
                result = self.tools.extract_tool(parameters["extraction_type"])
            else:
                raise ValueError(f"Unknown tool: {tool}")
            
            # Add routing metadata
            result["routing"] = {
                "selected_tool": tool,
                "autonomous": force_tool is None,
                "parameters": parameters
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error executing tool {tool}: {e}")
            return {
                "error": str(e),
                "tool": tool,
                "query": user_query,
                "status": "failed"
            }
    
    def explain_routing(self, user_query: str) -> Dict[str, Any]:
        """
        Explain why a query would be routed to a specific tool.
        Useful for debugging and transparency.
        """
        tool = self.route_query(user_query)
        parameters = self.extract_parameters(user_query, tool)
        
        explanation_prompt = PromptTemplate(
            template="""Explain in 1-2 sentences why the query "{query}" was routed to the "{tool}" tool.

Explanation:""",
            input_variables=["query", "tool"]
        )
        
        prompt_text = explanation_prompt.format(query=user_query, tool=tool)
        explanation = self.routing_llm.predict(prompt_text)
        
        return {
            "query": user_query,
            "selected_tool": tool,
            "parameters": parameters,
            "explanation": explanation,
            "tool_description": self.tools.get_tool_description(tool)
        }