import os
import streamlit as st
import chromadb
from dotenv import load_dotenv
from langchain.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from pypdf import PdfReader
from langchain.schema import Document
from chromadb.api.client import SharedSystemClient

# Load environment variables
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def init_app():
    """Initialize the Streamlit app with configurations."""
    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
    st.title("🤖 Document Q&A Chatbot")
    st.markdown(
        """
        Welcome to the **Document Q&A Chatbot**!  
        Upload a PDF document and interact with it using natural language queries.
        """
    )


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload a document and ask me questions about it!"}
    ]


def process_uploaded_file(uploaded_file):
    """Processes an uploaded PDF file into smaller chunks of text."""
    reader = PdfReader(uploaded_file)
    documents = []
    for page_number, page in enumerate(reader.pages):
        page_text = page.extract_text()
        document = Document(page_content=page_text, metadata={"page": page_number + 1})
        documents.append(document)

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    return text_splitter.split_documents(documents)


def initialize_vectorstore(doc_splits, collection_name):
    """Initializes a vector store for document retrieval."""
    return Chroma.from_documents(
        documents=doc_splits,
        collection_name=str(collection_name),
        embedding=OpenAIEmbeddings(api_key=OPENAI_API_KEY),
    )


def initialize_llm_and_retriever(vectorstore):
    """Initializes the language model and retriever."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    PROMPT_TEMPLATE = """
        You are an AI assistant. Using the following retrieved documents, answer the user's query accurately and concisely. While answering keep the following points in mind:
        -- Try to understand the user's query and what they are looking for like if they want short or detailed answers.
        -- Provide accurate and relevant information from the documents.
        -- Don't answer questions that are not related to the documents.
        -- Do not hallucinate any answer on your own.
        -- stay consistent with your responses.

        Retrieved Documents:
        {results}

        User Query:
        {user_query}

        Your Response:
    """
    prompt_template = PromptTemplate(
        input_variables=["results", "user_query"],
        template=PROMPT_TEMPLATE,
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    retriever = vectorstore.as_retriever(k=3)
    return llm_chain, retriever


def display_chat_interface(llm_chain, retriever):
    """Displays the chat interface using Streamlit's chat components."""
    st.subheader("💬 Chat with the Document")

    
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])

    
    if user_input := st.chat_input("Ask your question about the document:"):
        
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        
        with st.spinner("Thinking..."):
            retrieved_docs = retriever.get_relevant_documents(user_input)
            retrieved_docs_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

            response = llm_chain.run(results=retrieved_docs_text, user_query=user_input)

            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)



init_app()

collection_name = "RAG_DB"


client = chromadb.Client()
collection = client.get_or_create_collection(name=collection_name)


uploaded_file = st.file_uploader("Upload a PDF file:", type=["pdf"])

if uploaded_file:
    st.success(f"File uploaded: {uploaded_file.name}")
    doc_splits = process_uploaded_file(uploaded_file)
    vectorstore = initialize_vectorstore(doc_splits, collection_name)
    llm_chain, retriever = initialize_llm_and_retriever(vectorstore)

    
    display_chat_interface(llm_chain, retriever)




