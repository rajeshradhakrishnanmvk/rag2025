import os
import time
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Performance optimizations for better latency
FAST_LLM_MODEL = "gemma2:2b"  # Smaller, faster model than gemma3n:e4b
FAST_EMBEDDING_MODEL = "all-minilm:l6-v2"  # Faster embedding model

# LLM configuration optimized for speed
LLM_CONFIG = {
    "temperature": 0.1,  # Lower temperature for faster inference
    "num_predict": 256,  # Limit response length for speed
    "top_p": 0.9,
    "top_k": 20,
    "num_ctx": 2048,  # Reduced context window for speed
}

# Text splitting configuration for better performance
TEXT_SPLITTER_CONFIG = {
    "chunk_size": 500,  # Smaller chunks for faster processing
    "chunk_overlap": 50,
    "length_function": len,
}

class CachedRAGChain:
    """Simple caching wrapper for RAG responses"""
    def __init__(self, chain):
        self.chain = chain
        self.cache = {}
    
    def invoke(self, query_dict):
        query = query_dict["input"].strip().lower()
        if query in self.cache:
            print("🚀 Using cached response...")
            return self.cache[query]
        
        start_time = time.time()
        result = self.chain.invoke(query_dict)
        end_time = time.time()
        
        print(f"⏱️  Response time: {end_time - start_time:.2f} seconds")
        self.cache[query] = result
        return result

def find_pdf_files(folder):
    pdf_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def load_documents_from_pdfs(pdf_paths, text_splitter):
    """Load and split documents from PDFs with optimized chunking"""
    documents = []
    for i, path in enumerate(pdf_paths):
        print(f"📄 Processing PDF {i+1}/{len(pdf_paths)}: {os.path.basename(path)}")
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
            
            # Split documents into smaller chunks for better performance
            split_docs = text_splitter.split_documents(docs)
            
            for doc in split_docs:
                if doc.page_content and doc.page_content.strip():
                    doc.metadata["source"] = path
                    documents.append(doc)
        except Exception as e:
            print(f"⚠️  Error processing {path}: {e}")
            continue
    return documents

if __name__ == "__main__":
    print("🚀 Starting optimized PDF RAG with performance improvements...")
    
    pdf_folder = "<your pdf files path>"  # Change this to your folder path
    db_location = "./chroma_pdf_db"

    # Use faster embedding model instead of embeddinggemma:300m
    embeddings = OllamaEmbeddings(model=FAST_EMBEDDING_MODEL)
    
    # Initialize text splitter for optimal chunk sizes
    text_splitter = RecursiveCharacterTextSplitter(**TEXT_SPLITTER_CONFIG)

    # If DB exists, skip PDF processing and use existing embeddings
    if os.path.exists(db_location) and any(
        fname.endswith(".bin") or fname.endswith(".pkl") for fname in os.listdir(db_location)
    ):
        print("📚 Chroma DB exists. Loading existing embeddings...")
        vector_store = Chroma(
            collection_name="local_pdf_data",
            persist_directory=db_location,
            embedding_function=embeddings
        )
    else:
        pdf_files = find_pdf_files(pdf_folder)
        if not pdf_files:
            print("❌ No PDF files found.")
            exit(1)

        print(f"📁 Found {len(pdf_files)} PDF files. Processing with optimized chunking...")

        # Load and split documents
        documents = load_documents_from_pdfs(pdf_files, text_splitter)
        
        # Filter out empty documents
        documents = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
        if not documents:
            print("❌ No non-empty PDF documents found.")
            exit(1)

        print(f"📝 Created {len(documents)} text chunks for embedding...")

        # Always rebuild the vector store for fresh data
        if os.path.exists(db_location):
            import shutil
            shutil.rmtree(db_location)

        vector_store = Chroma(
            collection_name="local_pdf_data",
            persist_directory=db_location,
            embedding_function=embeddings
        )

        # Add documents in batches for better performance
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            print(f"🔄 Embedding batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
            vector_store.add_documents(documents=batch)

    # Optimized retriever - reduced from k=5 to k=3 for faster retrieval
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Use smaller, faster LLM with streaming and performance optimizations
    streaming_handler = StreamingStdOutCallbackHandler()
    llm = OllamaLLM(
        model=FAST_LLM_MODEL,  # Much faster than gemma3n:e4b
        callbacks=[streaming_handler],
        **LLM_CONFIG
    )

    # Optimized prompt template for concise responses
    prompt = ChatPromptTemplate.from_template(
        """Based on the provided PDF context, give a concise and accurate answer to the question.
        
        Context: {context}
        Question: {input}
        
        Answer:"""
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    # Add caching wrapper for repeated queries
    cached_chain = CachedRAGChain(rag_chain)
    
    print("\n🎯 PDF RAG system ready! Performance optimizations:")
    print(f"  • Using faster LLM: {FAST_LLM_MODEL}")
    print(f"  • Using faster embeddings: {FAST_EMBEDDING_MODEL}")
    print(f"  • Retrieving 3 documents (reduced for speed)")
    print(f"  • Streaming responses enabled")
    print(f"  • Response caching enabled")
    print(f"  • Optimized text chunks: {TEXT_SPLITTER_CONFIG['chunk_size']} chars")
    print("=" * 50)

    while True:
        question = input("\n💬 Enter your question (or type 'quit' to exit): ")
        if question.strip().lower() in ['quit', 'exit']:
            print("👋 Exiting.")
            break
        print("\n🤖 Generating response...\n")
        response = cached_chain.invoke({"input": question})
        print(f"\n\n📝 Full answer: {response['answer']}")