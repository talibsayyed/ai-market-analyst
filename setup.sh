#!/bin/bash

# AI Market Analyst Agent - Setup Script
# This script automates the setup process

echo "================================================"
echo "   AI Market Analyst Agent - Setup Script"
echo "================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.11+ required. Current version: $python_version"
    exit 1
fi
echo "✓ Python version OK: $python_version"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Skipping..."
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Setup environment file
echo "Setting up environment variables..."
if [ -f ".env" ]; then
    echo "⚠️  .env file already exists. Skipping..."
else
    cp .env.example .env
    echo "✓ Created .env file from template"
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env and add your OPENAI_API_KEY"
    echo ""
    read -p "Enter your OpenAI API key (or press Enter to add it later): " api_key
    if [ ! -z "$api_key" ]; then
        sed -i.bak "s/your_openai_api_key_here/$api_key/" .env
        rm .env.bak
        echo "✓ API key added to .env file"
    fi
fi
echo ""

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p data
mkdir -p chroma_db
mkdir -p evaluation
mkdir -p tests
echo "✓ Directories created"
echo ""

# Initialize vector store
echo "Initializing vector store..."
if [ -d "chroma_db" ] && [ "$(ls -A chroma_db)" ]; then
    echo "⚠️  Vector store already exists. Skipping initialization..."
else
    echo "This will be created on first run of the application."
fi
echo ""

echo "================================================"
echo "   Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your OPENAI_API_KEY (if not done)"
echo "2. Start the API server:"
echo "   source venv/bin/activate"
echo "   uvicorn app.main:app --reload"
echo ""
echo "3. In a new terminal, start the UI:"
echo "   source venv/bin/activate"
echo "   streamlit run streamlit_app.py"
echo ""
echo "4. Or use Docker:"
echo "   docker-compose up --build"
echo ""
echo "Access the application:"
echo "  - API: http://localhost:8000"
echo "  - UI: http://localhost:8501"
echo "  - API Docs: http://localhost:8000/docs"
echo ""
echo "================================================"