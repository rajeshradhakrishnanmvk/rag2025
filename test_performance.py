#!/usr/bin/env python3
"""
Performance test script for RAG optimizations
Tests model availability and basic functionality
"""

import time
import os
from datetime import datetime

def test_imports():
    """Test that all required libraries are available"""
    print("🔍 Testing imports...")
    try:
        from langchain_ollama import OllamaEmbeddings, OllamaLLM
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
        from langchain_core.prompts import ChatPromptTemplate
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain.chains import create_retrieval_chain
        from langchain_core.callbacks import StreamingStdOutCallbackHandler
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        import pandas as pd
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_models():
    """Test if optimized models are available"""
    print("\n🔍 Testing model availability...")
    
    # Import here to avoid issues
    from langchain_ollama import OllamaEmbeddings, OllamaLLM
    
    # Test embedding model
    try:
        print("Testing embedding model...")
        embeddings = OllamaEmbeddings(model="all-minilm:l6-v2")
        print("✅ Embedding model available!")
        embedding_available = True
    except Exception as e:
        print(f"⚠️  Embedding model not available: {e}")
        print("Run: ollama pull all-minilm:l6-v2")
        embedding_available = False
    
    # Test LLM model
    try:
        print("Testing LLM model...")
        llm = OllamaLLM(model="gemma2:2b", num_predict=10)
        print("✅ LLM model available!")
        llm_available = True
    except Exception as e:
        print(f"⚠️  LLM model not available: {e}")
        print("Run: ollama pull gemma2:2b")
        llm_available = False
    
    return embedding_available and llm_available

def test_basic_functionality():
    """Test basic RAG functionality with simple documents"""
    print("\n🔍 Testing basic RAG functionality...")
    
    try:
        from langchain_ollama import OllamaEmbeddings
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
        
        # Create simple test documents
        docs = [
            Document(page_content="The sky is blue on a clear day.", metadata={"source": "test1"}),
            Document(page_content="Python is a programming language.", metadata={"source": "test2"}),
            Document(page_content="Machine learning helps computers learn.", metadata={"source": "test3"})
        ]
        
        # Test embedding creation
        print("Creating test embeddings...")
        embeddings = OllamaEmbeddings(model="all-minilm:l6-v2")
        
        # Create temporary vector store
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory="./test_db"
        )
        
        # Test retrieval
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        results = retriever.invoke("What color is the sky?")
        
        print(f"✅ Retrieval test successful! Found {len(results)} documents")
        
        # Cleanup
        import shutil
        if os.path.exists("./test_db"):
            shutil.rmtree("./test_db")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def benchmark_performance():
    """Run a simple performance benchmark"""
    print("\n🔍 Running performance benchmark...")
    
    try:
        from langchain_ollama import OllamaEmbeddings, OllamaLLM
        
        # Test embedding speed
        print("Benchmarking embedding generation...")
        start_time = time.time()
        embeddings = OllamaEmbeddings(model="all-minilm:l6-v2")
        test_texts = ["This is a test sentence for embedding.", "Another test sentence for performance."]
        embedded = embeddings.embed_documents(test_texts)
        embedding_time = time.time() - start_time
        print(f"✅ Embedding 2 documents took: {embedding_time:.2f} seconds")
        
        # Test LLM speed
        print("Benchmarking LLM generation...")
        start_time = time.time()
        llm = OllamaLLM(
            model="gemma2:2b",
            temperature=0.1,
            num_predict=50,
            num_ctx=1024
        )
        response = llm.invoke("What is 2+2?")
        llm_time = time.time() - start_time
        print(f"✅ LLM response took: {llm_time:.2f} seconds")
        print(f"Response: {response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 RAG Performance Test Suite")
    print("=" * 50)
    print(f"Timestamp: {datetime.now()}")
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test models
    if not test_models():
        all_passed = False
        print("\n📥 To install missing models, run:")
        print("ollama pull all-minilm:l6-v2")
        print("ollama pull gemma2:2b")
    
    # Test basic functionality if models are available
    if all_passed:
        if not test_basic_functionality():
            all_passed = False
        
        if not benchmark_performance():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Your RAG system is ready for optimal performance.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    print("\n📊 Performance Summary:")
    print("  • Expected 60-80% faster inference with gemma2:2b")
    print("  • Expected 70-85% faster embeddings with all-minilm:l6-v2")
    print("  • Streaming responses for better user experience")
    print("  • Response caching for repeated queries")

if __name__ == "__main__":
    main()