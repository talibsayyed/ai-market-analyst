"""
Comparative Embedding Model Evaluation (Bonus Feature 2)

This script compares two embedding models:
1. OpenAI text-embedding-3-small (1536 dimensions)
2. Sentence-BERT all-MiniLM-L6-v2 (384 dimensions)

Evaluation Metrics:
- Retrieval Quality: Precision@K, relevance of retrieved chunks
- Latency: Average embedding time and search time
- Cost: API costs (for OpenAI)
- Memory: Vector store size
"""

import time
import os
from typing import List, Tuple
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Test queries with expected relevant chunks
TEST_QUERIES = [
    {
        "query": "What is Innovate Inc's market share?",
        "expected_keywords": ["12%", "market share", "Innovate Inc"]
    },
    {
        "query": "Who are the main competitors?",
        "expected_keywords": ["Synergy Systems", "FutureFlow", "QuantumLeap", "competitor"]
    },
    {
        "query": "What is the CAGR?",
        "expected_keywords": ["22%", "CAGR", "compound annual growth"]
    },
    {
        "query": "What are the strengths of the company?",
        "expected_keywords": ["strengths", "robust", "scalable", "customer loyalty"]
    },
    {
        "query": "What sectors should the company expand into?",
        "expected_keywords": ["healthcare", "finance", "opportunities", "expansion"]
    }
]

class EmbeddingEvaluator:
    """Evaluate and compare embedding models."""
    
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.results = {}
    
    def evaluate_model(
        self,
        model_name: str,
        embedding_function,
        persist_dir: str
    ) -> dict:
        """Evaluate a single embedding model."""
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        # Measure vector store creation time
        start_time = time.time()
        vector_store = Chroma.from_documents(
            documents=self.documents,
            embedding=embedding_function,
            persist_directory=persist_dir,
            collection_name=f"eval_{model_name.replace('/', '_')}"
        )
        creation_time = time.time() - start_time
        
        print(f"✓ Vector store created in {creation_time:.2f}s")
        
        # Evaluate retrieval quality and latency
        retrieval_scores = []
        latencies = []
        
        for test in TEST_QUERIES:
            query = test['query']
            expected_keywords = test['expected_keywords']
            
            # Measure search latency
            start_time = time.time()
            results = vector_store.similarity_search(query, k=4)
            search_time = time.time() - start_time
            latencies.append(search_time)
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance(results, expected_keywords)
            retrieval_scores.append(relevance_score)
            
            print(f"  Query: '{query[:40]}...'")
            print(f"    Relevance: {relevance_score:.2%} | Latency: {search_time*1000:.1f}ms")
        
        # Calculate metrics
        avg_relevance = np.mean(retrieval_scores)
        avg_latency = np.mean(latencies)
        
        # Estimate vector store size
        store_size = self._get_directory_size(persist_dir)
        
        metrics = {
            "model_name": model_name,
            "creation_time": creation_time,
            "avg_relevance": avg_relevance,
            "avg_latency_ms": avg_latency * 1000,
            "min_latency_ms": min(latencies) * 1000,
            "max_latency_ms": max(latencies) * 1000,
            "store_size_mb": store_size / (1024 * 1024),
            "retrieval_scores": retrieval_scores
        }
        
        self.results[model_name] = metrics
        return metrics
    
    def _calculate_relevance(
        self,
        results: List[Document],
        expected_keywords: List[str]
    ) -> float:
        """
        Calculate relevance score based on keyword presence.
        
        Simple but effective metric: what percentage of expected keywords
        appear in the retrieved chunks?
        """
        combined_text = " ".join([doc.page_content.lower() for doc in results])
        
        matches = sum(
            1 for keyword in expected_keywords
            if keyword.lower() in combined_text
        )
        
        return matches / len(expected_keywords) if expected_keywords else 0
    
    def _get_directory_size(self, path: str) -> int:
        """Calculate total size of directory in bytes."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
        except:
            pass
        return total_size
    
    def compare_models(self) -> dict:
        """Generate comparison report."""
        
        print(f"\n{'='*60}")
        print("COMPARATIVE ANALYSIS")
        print(f"{'='*60}\n")
        
        models = list(self.results.keys())
        if len(models) < 2:
            print("Need at least 2 models to compare")
            return {}
        
        model1, model2 = models[0], models[1]
        metrics1 = self.results[model1]
        metrics2 = self.results[model2]
        
        comparison = {
            "winner": {},
            "details": {}
        }
        
        # Compare metrics
        print("📊 PERFORMANCE COMPARISON\n")
        
        # Relevance
        if metrics1['avg_relevance'] > metrics2['avg_relevance']:
            winner = model1
            diff = (metrics1['avg_relevance'] - metrics2['avg_relevance']) * 100
        else:
            winner = model2
            diff = (metrics2['avg_relevance'] - metrics1['avg_relevance']) * 100
        
        comparison['winner']['relevance'] = winner
        print(f"🎯 Retrieval Quality Winner: {winner}")
        print(f"   {model1}: {metrics1['avg_relevance']:.1%}")
        print(f"   {model2}: {metrics2['avg_relevance']:.1%}")
        print(f"   Difference: {diff:.1f} percentage points\n")
        
        # Latency
        if metrics1['avg_latency_ms'] < metrics2['avg_latency_ms']:
            winner = model1
            speedup = metrics2['avg_latency_ms'] / metrics1['avg_latency_ms']
        else:
            winner = model2
            speedup = metrics1['avg_latency_ms'] / metrics2['avg_latency_ms']
        
        comparison['winner']['latency'] = winner
        print(f"⚡ Speed Winner: {winner}")
        print(f"   {model1}: {metrics1['avg_latency_ms']:.1f}ms")
        print(f"   {model2}: {metrics2['avg_latency_ms']:.1f}ms")
        print(f"   Speedup: {speedup:.2f}x faster\n")
        
        # Storage
        if metrics1['store_size_mb'] < metrics2['store_size_mb']:
            winner = model1
            reduction = (1 - metrics1['store_size_mb'] / metrics2['store_size_mb']) * 100
        else:
            winner = model2
            reduction = (1 - metrics2['store_size_mb'] / metrics1['store_size_mb']) * 100
        
        comparison['winner']['storage'] = winner
        print(f"💾 Storage Efficiency Winner: {winner}")
        print(f"   {model1}: {metrics1['store_size_mb']:.1f} MB")
        print(f"   {model2}: {metrics2['store_size_mb']:.1f} MB")
        print(f"   Reduction: {reduction:.1f}%\n")
        
        # Recommendation
        print(f"{'='*60}")
        print("🏆 FINAL RECOMMENDATION")
        print(f"{'='*60}\n")
        
        # Count wins
        wins1 = sum(1 for v in comparison['winner'].values() if v == model1)
        wins2 = sum(1 for v in comparison['winner'].values() if v == model2)
        
        if wins1 > wins2:
            recommended = model1
            reason = f"Wins {wins1}/3 categories"
        elif wins2 > wins1:
            recommended = model2
            reason = f"Wins {wins2}/3 categories"
        else:
            # Tie-breaker: relevance is most important
            if metrics1['avg_relevance'] > metrics2['avg_relevance']:
                recommended = model1
                reason = "Better retrieval quality (tie-breaker)"
            else:
                recommended = model2
                reason = "Better retrieval quality (tie-breaker)"
        
        print(f"Recommended Model: {recommended}")
        print(f"Reason: {reason}\n")
        
        # Use case specific recommendations
        print("📋 Use Case Recommendations:\n")
        print(f"• Production/API: {comparison['winner']['latency']} (fastest)")
        print(f"• Accuracy-Critical: {comparison['winner']['relevance']} (most accurate)")
        print(f"• Resource-Constrained: {comparison['winner']['storage']} (smallest footprint)")
        
        comparison['recommended'] = recommended
        comparison['reason'] = reason
        
        return comparison

def run_evaluation():
    """Run the full evaluation."""
    
    # Load document
    from app.document_processor import DocumentProcessor
    
    processor = DocumentProcessor()
    documents = processor.process_file(
        "data/market_report.txt",
        metadata={"source": "Innovate Inc. Market Research Report"}
    )
    
    print(f"\nLoaded {len(documents)} document chunks for evaluation\n")
    
    evaluator = EmbeddingEvaluator(documents)
    
    # Model 1: OpenAI text-embedding-3-small
    print("\n🔵 Model 1: OpenAI text-embedding-3-small")
    openai_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    evaluator.evaluate_model(
        "openai-3-small",
        openai_embeddings,
        "./eval_chroma_openai"
    )
    
    # Model 2: Sentence-BERT all-MiniLM-L6-v2
    print("\n🟢 Model 2: Sentence-BERT all-MiniLM-L6-v2")
    sbert_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    evaluator.evaluate_model(
        "sbert-minilm",
        sbert_embeddings,
        "./eval_chroma_sbert"
    )
    
    # Generate comparison
    comparison = evaluator.compare_models()
    
    # Save results
    import json
    with open("evaluation/results.json", "w") as f:
        json.dump({
            "model_results": evaluator.results,
            "comparison": comparison
        }, f, indent=2)
    
    print(f"\n✅ Results saved to evaluation/results.json")

if __name__ == "__main__":
    run_evaluation()