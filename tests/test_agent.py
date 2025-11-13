import pytest
from unittest.mock import Mock, patch, MagicMock
from app.agent import AutonomousMarketAnalystAgent
from app.tools import MarketAnalystTools
from app.vector_store import VectorStoreManager
from langchain.schema import Document

@pytest.fixture
def mock_vector_store_manager():
    """Create a mock vector store manager."""
    manager = Mock(spec=VectorStoreManager)
    manager.similarity_search.return_value = [
        Document(
            page_content="Innovate Inc. holds a 12% market share.",
            metadata={"chunk_id": 0}
        ),
        Document(
            page_content="Primary competitors are Synergy Systems (18%) and FutureFlow (15%).",
            metadata={"chunk_id": 1}
        )
    ]
    return manager

@pytest.fixture
def mock_tools(mock_vector_store_manager):
    """Create tools with mocked dependencies."""
    with patch('app.tools.ChatOpenAI') as mock_llm:
        mock_llm.return_value.predict.return_value = "Innovate Inc. holds a 12% market share."
        tools = MarketAnalystTools(mock_vector_store_manager)
        return tools

@pytest.fixture
def agent(mock_tools):
    """Create an agent with mocked tools."""
    with patch('app.agent.ChatOpenAI') as mock_routing_llm:
        mock_routing_llm.return_value.predict.return_value = "qa"
        agent = AutonomousMarketAnalystAgent(mock_tools)
        return agent

class TestAutonomousMarketAnalystAgent:
    """Test suite for the autonomous agent."""
    
    def test_route_query_to_qa(self, agent):
        """Test that questions are routed to Q&A tool."""
        with patch.object(agent.routing_llm, 'predict', return_value="qa"):
            tool = agent.route_query("What is the market share?")
            assert tool == "qa"
    
    def test_route_query_to_summarize(self, agent):
        """Test that summary requests are routed to summarization tool."""
        with patch.object(agent.routing_llm, 'predict', return_value="summarize"):
            tool = agent.route_query("Summarize the report")
            assert tool == "summarize"
    
    def test_route_query_to_extract(self, agent):
        """Test that extraction requests are routed to extraction tool."""
        with patch.object(agent.routing_llm, 'predict', return_value="extract"):
            tool = agent.route_query("Extract all competitors")
            assert tool == "extract"
    
    def test_invalid_routing_defaults_to_qa(self, agent):
        """Test that invalid routing responses default to Q&A."""
        with patch.object(agent.routing_llm, 'predict', return_value="invalid"):
            tool = agent.route_query("Some query")
            assert tool == "qa"
    
    def test_process_query_with_qa(self, agent):
        """Test processing a query that routes to Q&A."""
        with patch.object(agent, 'route_query', return_value="qa"):
            with patch.object(agent.tools, 'qa_tool', return_value={"tool": "qa", "answer": "Test answer"}):
                result = agent.process_query("What is the market share?")
                
                assert result['tool'] == 'qa'
                assert 'routing' in result
                assert result['routing']['selected_tool'] == 'qa'
                assert result['routing']['autonomous'] == True
    
    def test_process_query_with_forced_tool(self, agent):
        """Test processing a query with forced tool selection."""
        with patch.object(agent.tools, 'summarize_tool', return_value={"tool": "summarize", "summary": "Test summary"}):
            result = agent.process_query("What is the market share?", force_tool="summarize")
            
            assert result['tool'] == 'summarize'
            assert result['routing']['autonomous'] == False
            assert result['routing']['selected_tool'] == 'summarize'
    
    def test_extract_parameters_for_qa(self, agent):
        """Test parameter extraction for Q&A queries."""
        params = agent.extract_parameters("What is the market share?", "qa")
        
        assert 'question' in params
        assert params['question'] == "What is the market share?"
    
    def test_extract_parameters_for_summarize(self, agent):
        """Test parameter extraction for summarization queries."""
        with patch.object(agent.routing_llm, 'predict', return_value="competitors"):
            params = agent.extract_parameters("Summarize competitors", "summarize")
            
            assert 'focus_area' in params
    
    def test_extract_parameters_for_extract(self, agent):
        """Test parameter extraction for extraction queries."""
        params = agent.extract_parameters("Extract all data", "extract")
        
        assert 'extraction_type' in params
        assert params['extraction_type'] == "all"
    
    def test_explain_routing(self, agent):
        """Test routing explanation feature."""
        with patch.object(agent, 'route_query', return_value="qa"):
            with patch.object(agent.routing_llm, 'predict', return_value="This query asks a specific question."):
                explanation = agent.explain_routing("What is the market share?")
                
                assert 'query' in explanation
                assert 'selected_tool' in explanation
                assert 'explanation' in explanation
                assert explanation['selected_tool'] == 'qa'

class TestMarketAnalystTools:
    """Test suite for individual tools."""
    
    def test_qa_tool_returns_answer(self, mock_tools):
        """Test that Q&A tool returns a valid answer."""
        with patch.object(mock_tools.llm, 'predict', return_value="Innovate Inc. holds 12% market share."):
            result = mock_tools.qa_tool("What is the market share?")
            
            assert result['tool'] == 'qa'
            assert 'answer' in result
            assert 'sources' in result
            assert len(result['sources']) > 0
    
    def test_summarize_tool_returns_summary(self, mock_tools):
        """Test that summarization tool returns a summary."""
        with patch.object(mock_tools.llm, 'predict', return_value="Summary of key findings..."):
            result = mock_tools.summarize_tool("overall")
            
            assert result['tool'] == 'summarize'
            assert 'summary' in result
            assert result['focus_area'] == 'overall'
    
    def test_extract_tool_returns_json(self, mock_tools):
        """Test that extraction tool returns structured data."""
        json_response = '''
        {
            "company_info": {"name": "Innovate Inc.", "market_share": 12},
            "competitors": [{"name": "Synergy Systems", "market_share": 18}]
        }
        '''
        
        with patch.object(mock_tools.llm, 'predict', return_value=json_response):
            result = mock_tools.extract_tool("all")
            
            assert result['tool'] == 'extract'
            assert result['status'] == 'success'
            assert 'data' in result
            assert 'company_info' in result['data']
    
    def test_extract_tool_handles_malformed_json(self, mock_tools):
        """Test that extraction tool handles malformed JSON gracefully."""
        with patch.object(mock_tools.llm, 'predict', return_value="Invalid JSON {"):
            result = mock_tools.extract_tool("all")
            
            assert result['tool'] == 'extract'
            assert result['status'] == 'error'
            assert 'error' in result
    
    def test_get_tool_description(self, mock_tools):
        """Test getting tool descriptions."""
        qa_desc = mock_tools.get_tool_description("qa")
        assert "questions" in qa_desc.lower()
        
        summ_desc = mock_tools.get_tool_description("summarize")
        assert "summar" in summ_desc.lower()
        
        extract_desc = mock_tools.get_tool_description("extract")
        assert "extract" in extract_desc.lower() or "json" in extract_desc.lower()

class TestDocumentProcessing:
    """Test document processing and chunking."""
    
    def test_chunk_size_and_overlap(self):
        """Test that chunking respects size and overlap settings."""
        from app.document_processor import DocumentProcessor
        
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        
        # Create a test document
        test_text = "A" * 300  # 300 character document
        
        chunks = processor.process_document(test_text)
        
        # Should create multiple chunks
        assert len(chunks) > 1
        
        # Each chunk should be around the specified size
        for chunk in chunks[:-1]:  # Exclude last chunk which may be smaller
            assert len(chunk.page_content) <= 120  # chunk_size + some tolerance

    def test_metadata_preservation(self):
        """Test that metadata is preserved in chunks."""
        from app.document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        test_metadata = {"source": "test_document.txt", "category": "market_research"}
        
        chunks = processor.process_document("Test content", metadata=test_metadata)
        
        for chunk in chunks:
            assert chunk.metadata['source'] == "test_document.txt"
            assert chunk.metadata['category'] == "market_research"
            assert 'chunk_id' in chunk.metadata

if __name__ == "__main__":
    pytest.main([__file__, "-v"])