import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="AI Market Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .tool-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .json-output {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        max-height: 500px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📊 AI Market Analyst Agent</h1>', unsafe_allow_html=True)
st.markdown("### Intelligent Market Research Analysis for Innovate Inc.")

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=Innovate+Inc", use_container_width=True)
    st.markdown("---")
    
    st.markdown("### 🎯 Available Tools")
    st.markdown("""
    - **🤖 Autonomous Query**: Agent decides best tool
    - **❓ Q&A**: Answer specific questions
    - **📝 Summarize**: Generate summaries
    - **🗂️ Extract**: Get structured data
    """)
    
    st.markdown("---")
    
    st.markdown("### 📚 Sample Queries")
    st.markdown("""
    **Q&A:**
    - What is the market share?
    - Who are the competitors?
    - What is the CAGR?
    
    **Summarize:**
    - Summarize the report
    - Key findings about competitors
    - Overview of opportunities
    
    **Extract:**
    - Extract all data as JSON
    - Get structured competitor info
    """)

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🤖 Autonomous Agent",
    "❓ Q&A Tool",
    "📝 Summarize Tool",
    "🗂️ Extract Tool",
    "📈 Analytics"
])

# Tab 1: Autonomous Agent
with tab1:
    st.markdown("### 🤖 Autonomous Query (Bonus Feature)")
    st.info("💡 The agent automatically determines which tool to use based on your query!")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_query = st.text_area(
            "Enter your query:",
            placeholder="e.g., What is Innovate Inc's market share?",
            height=100,
            key="auto_query"
        )
    
    with col2:
        st.markdown("#### Options")
        force_tool = st.selectbox(
            "Force specific tool:",
            ["Auto-detect", "Q&A", "Summarize", "Extract"],
            key="force_tool"
        )
        
        explain_mode = st.checkbox("Explain routing", value=False)
    
    if st.button("🚀 Process Query", type="primary", use_container_width=True):
        if user_query:
            with st.spinner("Processing..."):
                try:
                    # Map tool selection
                    force_tool_param = None if force_tool == "Auto-detect" else force_tool.lower()
                    
                    if explain_mode:
                        # Explain routing
                        response = requests.post(
                            f"{API_URL}/query/explain",
                            json={"query": user_query}
                        )
                    else:
                        # Execute query
                        response = requests.post(
                            f"{API_URL}/query",
                            json={"query": user_query, "force_tool": force_tool_param}
                        )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if explain_mode:
                            st.success("✅ Routing Explanation")
                            st.markdown(f"**Selected Tool:** `{result['selected_tool']}`")
                            st.markdown(f"**Explanation:** {result['explanation']}")
                            st.markdown(f"**Tool Description:** {result['tool_description']}")
                            
                            with st.expander("📋 Parameters"):
                                st.json(result['parameters'])
                        else:
                            # Display routing info
                            if 'routing' in result:
                                routing = result['routing']
                                cols = st.columns(3)
                                cols[0].metric("Tool Used", routing['selected_tool'].upper())
                                cols[1].metric("Mode", "Autonomous" if routing['autonomous'] else "Forced")
                                
                            st.success("✅ Query Processed Successfully")
                            
                            # Display result based on tool
                            if result.get('tool') == 'qa':
                                st.markdown(f"**Answer:** {result['answer']}")
                                with st.expander("📚 Source Chunks"):
                                    for i, source in enumerate(result.get('sources', []), 1):
                                        st.markdown(f"**Chunk {i}:** {source['content']}")
                            
                            elif result.get('tool') == 'summarize':
                                st.markdown("**Summary:**")
                                st.markdown(result['summary'])
                            
                            elif result.get('tool') == 'extract':
                                st.markdown("**Extracted Data:**")
                                st.json(result.get('data', {}))
                            
                            # Raw response
                            with st.expander("🔍 Raw Response"):
                                st.json(result)
                    else:
                        st.error(f"❌ Error: {response.status_code} - {response.text}")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Please enter a query")

# Tab 2: Q&A Tool
with tab2:
    st.markdown("### ❓ Q&A Tool")
    st.info("Ask specific questions about the Innovate Inc. market research report")
    
    question = st.text_input(
        "Your question:",
        placeholder="e.g., What is the current market size?",
        key="qa_question"
    )
    
    if st.button("Get Answer", type="primary", key="qa_button"):
        if question:
            with st.spinner("Searching for answer..."):
                try:
                    response = requests.post(
                        f"{API_URL}/qa",
                        json={"question": question}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        st.success("✅ Answer Found")
                        st.markdown(f"**Question:** {result['question']}")
                        st.markdown(f"**Answer:** {result['answer']}")
                        
                        st.markdown(f"*Based on {result['source_chunks']} source chunks*")
                        
                        with st.expander("📚 View Source Chunks"):
                            for i, source in enumerate(result['sources'], 1):
                                st.markdown(f"**Chunk {i} (ID: {source['chunk_id']}):**")
                                st.text(source['content'])
                                st.markdown("---")
                    else:
                        st.error(f"❌ Error: {response.status_code}")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Please enter a question")

# Tab 3: Summarize Tool
with tab3:
    st.markdown("### 📝 Summarization Tool")
    st.info("Generate comprehensive summaries of market research findings")
    
    focus_area = st.selectbox(
        "Focus area:",
        ["overall", "competitors", "market size", "SWOT analysis", "growth projections", "opportunities", "threats"],
        key="summarize_focus"
    )
    
    if st.button("Generate Summary", type="primary", key="summarize_button"):
        with st.spinner(f"Generating summary for '{focus_area}'..."):
            try:
                response = requests.post(
                    f"{API_URL}/summarize",
                    json={"focus_area": focus_area}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success("✅ Summary Generated")
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**Focus Area:** {result['focus_area']}")
                    with col2:
                        st.metric("Chunks Analyzed", result['chunks_analyzed'])
                    
                    st.markdown("---")
                    st.markdown("### 📄 Summary")
                    st.markdown(result['summary'])
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Summary",
                        data=result['summary'],
                        file_name=f"summary_{focus_area}.txt",
                        mime="text/plain"
                    )
                else:
                    st.error(f"❌ Error: {response.status_code}")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Tab 4: Extract Tool
with tab4:
    st.markdown("### 🗂️ Data Extraction Tool")
    st.info("Extract structured data from the document in JSON format")
    
    extraction_type = st.selectbox(
        "Extraction type:",
        ["all", "company_info", "market_data", "competitors", "swot"],
        key="extract_type"
    )
    
    if st.button("Extract Data", type="primary", key="extract_button"):
        with st.spinner("Extracting structured data..."):
            try:
                response = requests.post(
                    f"{API_URL}/extract",
                    json={"extraction_type": extraction_type}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result.get('status') == 'success':
                        st.success("✅ Data Extracted Successfully")
                        
                        # Display formatted JSON
                        st.markdown("### 📊 Extracted Data")
                        st.json(result['data'])
                        
                        # Download button
                        json_str = json.dumps(result['data'], indent=2)
                        st.download_button(
                            label="📥 Download JSON",
                            data=json_str,
                            file_name=f"extracted_data_{extraction_type}.json",
                            mime="application/json"
                        )
                        
                        # Data preview tables
                        data = result['data']
                        
                        if 'company_info' in data:
                            st.markdown("#### 🏢 Company Information")
                            st.table({
                                "Name": [data['company_info']['name']],
                                "Product": [data['company_info']['product']],
                                "Market Share": [f"{data['company_info']['market_share']}%"]
                            })
                        
                        if 'competitors' in data:
                            st.markdown("#### 🏆 Competitors")
                            import pandas as pd
                            df = pd.DataFrame(data['competitors'])
                            df['market_share'] = df['market_share'].apply(lambda x: f"{x}%")
                            st.dataframe(df, use_container_width=True)
                    
                    else:
                        st.error(f"❌ Extraction failed: {result.get('error')}")
                        with st.expander("Raw Response"):
                            st.text(result.get('raw_response'))
                else:
                    st.error(f"❌ Error: {response.status_code}")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Tab 5: Analytics
with tab5:
    st.markdown("### 📈 System Analytics")
    
    # Health check
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            health = response.json()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                status = "🟢 Healthy" if health['status'] == 'healthy' else "🔴 Unhealthy"
                st.metric("System Status", status)
            
            with col2:
                vs_status = "✅ Yes" if health['vector_store_initialized'] else "❌ No"
                st.metric("Vector Store", vs_status)
            
            with col3:
                tools_status = "✅ Yes" if health['tools_initialized'] else "❌ No"
                st.metric("Tools", tools_status)
            
            with col4:
                agent_status = "✅ Yes" if health['agent_initialized'] else "❌ No"
                st.metric("Agent", agent_status)
    
    except Exception as e:
        st.error(f"Unable to connect to API: {str(e)}")
    
    st.markdown("---")
    
    # Tool descriptions
    st.markdown("### 🛠️ Tool Descriptions")
    try:
        response = requests.get(f"{API_URL}/tools/descriptions")
        if response.status_code == 200:
            descriptions = response.json()
            
            for tool, desc in descriptions.items():
                with st.expander(f"**{tool.upper()} Tool**"):
                    st.write(desc)
    except:
        st.warning("Unable to fetch tool descriptions")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "AI Market Analyst Agent v1.0.0 | Built with FastAPI, LangChain, and Streamlit"
    "</div>",
    unsafe_allow_html=True
)