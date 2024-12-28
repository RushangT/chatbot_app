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



load_dotenv(override=True)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


def init_app():
    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
    st.title("📄Document Q&A Chatbot")
    st.markdown("""
        Welcome to the **Document Q&A Chatbot**!  
        Upload a PDF document and interact with it using natural language queries.  
    """)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 


def process_uploaded_file(uploaded_file):
    reader = PdfReader(uploaded_file)
    documents = []
    for page_number, page in enumerate(reader.pages):
        page_text = page.extract_text()
        document = Document(page_content=page_text, metadata={'page': page_number + 1})
        documents.append(document)

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    return text_splitter.split_documents(documents)


def initialize_vectorstore(doc_splits, collection_name):
    return Chroma.from_documents(
        documents=doc_splits,
        collection_name=str(collection_name),
        embedding=OpenAIEmbeddings(api_key=OPENAI_API_KEY),
    )


def initialize_llm_and_retriever(vectorstore):
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
    st.subheader("💬 Chat with the Document")
    st.markdown("Ask a question about the uploaded document, and get AI-generated responses.")

    query = st.chat_input("🔍 Your Question:")
    if query:
        with st.spinner("Thinking..."):
            retrieved_docs = retriever.get_relevant_documents(query)
            retrieved_docs_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

            response = llm_chain.run(
                results=retrieved_docs_text,
                user_query=query
            )
            st.write("### 🤖 AI Response:")
            st.success(response)
            st.session_state.chat_history.append({"user": query, "AI": response})


def show_chat_history():
    st.subheader("Chat History")
    for chat in st.session_state.chat_history:
        st.write(f"**User:** {chat['user']}")
        st.write(f"**AI:** {chat['AI']}")
        st.write("----")


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
    if st.button("Show Chat History"):
        show_chat_history()



