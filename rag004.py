from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.callbacks import StreamingStdOutCallbackHandler
import pandas as pd
import os
import time

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

print("🚀 Starting optimized CSV RAG with performance improvements...")

df = pd.read_csv("local_data.csv")

# Use faster embedding model instead of embeddinggemma:300m
embeddings = OllamaEmbeddings(model=FAST_EMBEDDING_MODEL)

db_location = "./chroma_langchain_db"

# Check if Chroma DB exists (by checking for index files)
db_exists = os.path.exists(db_location) and any(
    fname.endswith(".bin") or fname.endswith(".pkl") for fname in os.listdir(db_location)
)

vector_store = Chroma(
    collection_name="local_data",
    persist_directory=db_location,
    embedding_function=embeddings
)

if not db_exists:
    documents = []
    ids = []
    for i, row in df.iterrows():
        document = Document(
            page_content=row["Type"] + " " + str(row["Assigned To"]) + " " + row["Title"],
            metadata={"assigned_to": row["Assigned To"], "type": row["Type"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)
    vector_store.add_documents(documents=documents, ids=ids)

# Optimized retriever - reduced from k=5 to k=3 for faster retrieval
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# Use smaller, faster LLM with streaming and performance optimizations
streaming_handler = StreamingStdOutCallbackHandler()
llm = OllamaLLM(
    model=FAST_LLM_MODEL,  # Much faster than gemma3n:e4b
    callbacks=[streaming_handler],
    **LLM_CONFIG
)

# Optimized prompt template for concise responses
prompt = ChatPromptTemplate.from_template(
    """Based on the context, provide a concise answer to the question.
    
    Context: {context}
    Question: {input}
    
    Answer:""")

# Create a chain to combine the retrieved documents and format the prompt
document_chain = create_stuff_documents_chain(llm, prompt)
# Create the full retrieval chain
rag_chain = create_retrieval_chain(retriever, document_chain)

# Add caching wrapper for repeated queries
cached_chain = CachedRAGChain(rag_chain)

print("\n🎯 RAG system ready! Performance optimizations:")
print(f"  • Using faster LLM: {FAST_LLM_MODEL}")
print(f"  • Using faster embeddings: {FAST_EMBEDDING_MODEL}")
print(f"  • Retrieving 3 documents (reduced for speed)")
print(f"  • Streaming responses enabled")
print(f"  • Response caching enabled")
print("=" * 50)

while True:
    question = input("\n💬 Enter your question (or type 'quit' to exit): ")
    if question.strip().lower() in ['quit', 'exit']:
        print("👋 Exiting.")
        break
    print("\n🤖 Generating response...\n")
    response = cached_chain.invoke({"input": question})
    print(f"\n\n📝 Full answer: {response['answer']}")