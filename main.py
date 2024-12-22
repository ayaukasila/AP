import streamlit as st
import logging
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import chromadb
import os
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Configure ChromaDB
chroma_client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chroma_db"))

# Define the LLM model to be used
llm_model = "llama3.1:8b"

# Define a custom embedding function for ChromaDB using Ollama
class ChromaDBEmbeddingFunction:
    def __init__(self, langchain_embeddings):
        self.langchain_embeddings = langchain_embeddings

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return self.langchain_embeddings.embed_documents(input)

# Initialize the embedding function with Ollama embeddings
embedding = ChromaDBEmbeddingFunction(
    OllamaEmbeddings(
        model=llm_model,
        base_url="http://localhost:11434"  # Adjust the base URL as per your Ollama server configuration
    )
)

# Define a collection for the RAG workflow
collection_name = "rag_collection_demo_1"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    metadata={"description": "A collection for RAG with Ollama - Demo1"},
    embedding_function=embedding  # Use the custom embedding function
)

# Function to add documents to the ChromaDB collection
def add_documents_to_collection(documents, ids):
    collection.add(
        documents=documents,
        ids=ids
    )

# Function to query the ChromaDB collection
def query_chromadb(query_text, n_results=1):
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results["documents"], results["metadatas"]

# Function to interact with the Ollama LLM
def query_ollama(prompt):
    llm = OllamaLLM(model=llm_model)
    return llm.invoke(prompt)

# RAG pipeline: Combine ChromaDB and Ollama for Retrieval-Augmented Generation
def rag_pipeline(query_text):
    # Step 1: Retrieve relevant documents from ChromaDB
    retrieved_docs, metadata = query_chromadb(query_text)
    context = " ".join(retrieved_docs[0]) if retrieved_docs else "No relevant documents found."

    # Step 2: Send the query along with the context to Ollama
    augmented_prompt = f"Context: {context}\n\nQuestion: {query_text}\nAnswer:"
    response = query_ollama(augmented_prompt)
    return response

# Streamlit Web Interface
st.title("Chat with Ollama Model")

# Select a model from the sidebar
model = st.sidebar.selectbox("Choose a model", ["llama3.1:8b", "phi3", "mistral"])

if not model:
    st.warning("Please select a model.")

# Initialize Ollama model
def stream_chat(model, messages):
    try:
        llm = Ollama(model=model, request_timeout=120.0)
        resp = llm.stream_chat(messages)
        response = ""
        response_placeholder = st.empty()
        for r in resp:
            response += r.delta
            response_placeholder.write(response)
        logging.info(f"Response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error during streaming: {str(e)}")
        raise e

# Check for messages in session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Take user input for a new message
if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        st.spinner("Thinking...")

        # Convert the messages list into a list of ChatMessage objects
        messages = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in st.session_state.messages]
        
        # Get the response from Ollama
        response = stream_chat(model, messages)

        if response:
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Example: Add sample documents to the collection
documents = [
    "Artificial intelligence is the simulation of human intelligence processes by machines.",
    "Python is a programming language that lets you work quickly and integrate systems more effectively.",
    "ChromaDB is a vector database designed for AI applications."
]
doc_ids = ["doc1", "doc2", "doc3"]
add_documents_to_collection(documents, doc_ids)

# Query RAG pipeline
query = st.text_input("Ask a question with context:")
if query:
    response = rag_pipeline(query)
    st.write("Response:", response)
