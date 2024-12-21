# %%
from dotenv import load_dotenv
import os



load_dotenv(override = True)
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
TAVILY_API_KEY=os.getenv('TAVILY_API_KEY')


import streamlit as st
from dotenv import load_dotenv
import os
from langchain.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from pypdf import PdfReader
from langchain.schema import Document
import chromadb
import json
import chromadb.api
from chromadb.api.client import SharedSystemClient

from chromadb.api.client import SharedSystemClient
import streamlit as st
from langchain.memory import ConversationBufferMemory





# Streamlit app title and description
st.title("RAG chatbot")
st.markdown("""
Upload a PDF document, and ask questions about its content using an AI-powered chat interface.
""")

# Initialize or load chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # To store chat history as a list of dictionaries

# Function to save chat history to a file
def save_chat_history():
    # Define directory and file path
    directory = "d:/chat history"
    file_path = os.path.join(directory, "chat_history.txt")
    
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_path, "w") as file:
        json.dump(st.session_state.chat_history, file, indent=4)

# Upload a PDF file
with st.sidebar:
    SharedSystemClient.clear_system_cache()
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file:
        st.write(f"Processing file: {uploaded_file.name}")

    # Extract text from PDF
        reader = PdfReader(uploaded_file)
        documents = []
        for page_number, page in enumerate(reader.pages):
            page_text = page.extract_text()
            document = Document(page_content=page_text, metadata={'filetype': 'resume', 'page': page_number + 1})
            documents.append(document)

    # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=200
        )
        doc_splits = text_splitter.split_documents(documents)
    
        client = chromadb.Client()

        collection = client.get_or_create_collection(name="rushang3")

    # Create Chroma VectorStore
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rushang3",
            embedding=OpenAIEmbeddings(api_key=OPENAI_API_KEY),
        )

    # Define LLM and prompt template
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        PROMPT_TEMPLATE = """
        You are an AI assistant. Using the following retrieved documents, answer the user's query accurately and concisely.

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

    # Chat interface
st.subheader("Chat with the document")
query = st.text_input("Ask a question about the document:")
if query:
        # Retrieve similar documents
    retrieved_docs = retriever.get_relevant_documents(query)
    retrieved_docs_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Run the LLM chain
    response = llm_chain.run(
        results=retrieved_docs_text,
        user_query=query
    )
    st.write("### AI Response:")
    st.write(response)
    SharedSystemClient.clear_system_cache()
    st.session_state.chat_history.append({"user": query, "AI": response})
        
        
        # Save to file (optional step to persist)
    save_chat_history()
        
if st.button("Show Chat History"):
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for entry in st.session_state.chat_history:
            st.write(f"**Q:** {entry['user']}")
            st.write(f"**A:** {entry['AI']}")
    else:
        st.write("No chat history available.")




