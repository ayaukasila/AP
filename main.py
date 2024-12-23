__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import logging
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import chromadb
import os
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage


logging.basicConfig(level=logging.INFO)

chroma_client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chroma_db"))
llm_model = "llama3.1:8b"

class ChromaDBEmbeddingFunction:
    def __init__(self, langchain_embeddings):
        self.langchain_embeddings = langchain_embeddings

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return self.langchain_embeddings.embed_documents(input)

embedding = ChromaDBEmbeddingFunction(
    OllamaEmbeddings(
        model=llm_model,
        base_url="http://localhost:11434"
    )
)

collection_name = "rag_collection_demo_1"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    metadata={"description": "A collection for RAG with Ollama - Demo1"},
    embedding_function=embedding
)

def add_documents_to_collection(documents, ids):
    collection.add(
        documents=documents,
        ids=ids
    )

def query_chromadb(query_text, n_results=1):
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results["documents"], results["metadatas"]

def query_ollama(prompt):
    llm = OllamaLLM(model=llm_model)
    return llm.invoke(prompt)

def rag_pipeline(query_text):
    retrieved_docs, metadata = query_chromadb(query_text)
    context = " ".join(retrieved_docs[0]) if retrieved_docs else "No relevant documents found."

    augmented_prompt = f"Context: {context}\n\nQuestion: {query_text}\nAnswer:"
    response = query_ollama(augmented_prompt)
    return response

st.title("Chat with Ollama")

model = st.sidebar.selectbox("Choose a model", ["llama3.1:8b", "phi3", "mistral"])

if not model:
    st.warning("Please select a model.")

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

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        st.spinner("Thinking...")

        messages = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in st.session_state.messages]
        
        response = stream_chat(model, messages)

        if response:
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

documents = [
    "Artificial intelligence is the simulation of human intelligence processes by machines.",
    "Python is a programming language that lets you work quickly and integrate systems more effectively.",
    "ChromaDB is a vector database designed for AI applications."
]
doc_ids = ["doc1", "doc2", "doc3"]
add_documents_to_collection(documents, doc_ids)


query = st.text_input("Ask a question with context:")
if query:
    response = rag_pipeline(query)
    st.write("Response:", response)
