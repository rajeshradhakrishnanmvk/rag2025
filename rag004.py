from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import pandas as pd
import os

df = pd.read_csv("local_data.csv")

embeddings = OllamaEmbeddings(model="embeddinggemma:300m")

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

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

llm = OllamaLLM(model="gemma3n:e4b")

prompt = ChatPromptTemplate.from_template(
    """Answer the user's question based on the provided context.
    Context: {context}
    Question: {input}
    """)

# Create a chain to combine the retrieved documents and format the prompt
document_chain = create_stuff_documents_chain(llm, prompt)
# Create the full retrieval chain
rag_chain = create_retrieval_chain(retriever, document_chain)
while True:
    question = input("Enter your question (or type 'quit' to exit): ")
    if question.strip().lower() in ['quit', 'exit']:
        print("Exiting.")
        break
    response = rag_chain.invoke({"input": question})
    print(response["answer"])