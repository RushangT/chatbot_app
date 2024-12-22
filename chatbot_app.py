
from dotenv import load_dotenv
import os
import streamlit as st
import json
from langchain.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from pypdf import PdfReader
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
import chromadb
from chromadb.api.client import SharedSystemClient

# Load environment variables
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Streamlit app title and header
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("📄 AI-Powered Document Q&A Chatbot")
st.markdown("""
Welcome to the **AI-Powered Document Q&A Chatbot**!  
Upload a PDF document in the sidebar and interact with it using natural language queries.  
""")

# Initialize or load chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

# Function to save chat history to a file
def save_chat_history():
    directory = "d:/chat history"
    file_path = os.path.join(directory, "chat_history.txt")
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_path, "w") as file:
        json.dump(st.session_state.chat_history, file, indent=4)

# Sidebar for file upload and processing

with st.sidebar:
    SharedSystemClient.clear_system_cache()
    collection_name = st.text_input("Enter a collection name (or leave blank for default):", "default_collection")

# Connect to ChromaDB
    client = chromadb.Client()

# Check if collection exists, create if not
    
    collection = client.get_or_create_collection(name=collection_name)
    st.success(f"Using collection: {collection_name}")
    
    
    uploaded_file = st.file_uploader("Upload a PDF file:", type=["pdf"])
    
    if uploaded_file:
        st.success(f"File uploaded: {uploaded_file.name}")
        
        reader = PdfReader(uploaded_file)
        documents = []
        for page_number, page in enumerate(reader.pages):
            page_text = page.extract_text()
            document = Document(page_content=page_text, metadata={'filetype': 'resume', 'page': page_number + 1})
            documents.append(document)

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=200
        )
        doc_splits = text_splitter.split_documents(documents)



        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name=str(collection_name),
            embedding=OpenAIEmbeddings(api_key=OPENAI_API_KEY),
        )

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        PROMPT_TEMPLATE = """
        You are an AI assistant. Using the following retrieved documents, answer the user's query accurately and concisely.You can give detailed responses utilizing the information in the documents.
        Don't answer questions that are not related to the documents or hallucinate any answer on your
        own.

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

# Main content area for chat interface
st.subheader("💬 Chat with the Document")
st.markdown("Ask a question about the uploaded document, and get AI-generated responses.")

query = st.text_input("🔍 Your Question:")
if query:
    retrieved_docs = retriever.get_relevant_documents(query)
    retrieved_docs_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    response = llm_chain.run(
        history=st.session_state.memory.buffer,
        results=retrieved_docs_text,
        user_query=query
    )
    st.write("### 🤖 AI Response:")
    st.success(response)

    SharedSystemClient.clear_system_cache()
    st.session_state.chat_history.append({"user": query, "AI": response})

    save_chat_history()

# Button to display chat history
if st.button("📜 Show Chat History"):
    if st.session_state.chat_history:
        st.subheader("🗂️ Chat History")
        st.markdown("Here's the history of your interactions with the document:")
        for entry in st.session_state.chat_history:
            st.markdown(f"**Q:** {entry['user']}")
            st.markdown(f"**A:** {entry['AI']}")
    else:
        st.warning("No chat history available.")



