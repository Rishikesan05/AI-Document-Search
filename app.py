import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

# Load environment variables (GOOGLE_API_KEY)
load_dotenv()

# Streamlit App Configuration
st.set_page_config(page_title="AI Document Search", page_icon="📄")
st.title("📄 AI Document Search (RAG Chatbot)")
st.write("Upload a PDF document and ask questions about its content.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for PDF Upload
with st.sidebar:
    st.header("Document Upload")
    pdf_file = st.file_uploader("Upload your PDF file", type=["pdf"])
    
    if pdf_file is not None:
        st.success("PDF Uploaded Successfully!")

@st.cache_resource(show_spinner=False)
def process_pdf(file):
    """Reads PDF, extracts text, chunks it, and creates FAISS Vector Store."""
    # 1. Read Text
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text()
            
    # 2. Split Text into Chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # 3. Create FAISS Vector Store using Google Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

# Main Logic
if pdf_file:
    with st.spinner("Processing document (Reading, Chunking, Embedding)..."):
        vector_store = process_pdf(pdf_file)
        
    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    user_query = st.chat_input("Ask a question about the document:")
    
    if user_query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # 4. Retrieval & Generation (RAG)
        with st.chat_message("assistant"):
            with st.spinner("Searching document for answers..."):
                # Retrieve relevant chunks from FAISS
                docs = vector_store.similarity_search(user_query, k=3)
                
                # Load QA Chain with Gemini model
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
                chain = load_qa_chain(llm, chain_type="stuff")
                
                # Generate Answer
                response = chain.run(input_documents=docs, question=user_query)
                
                st.markdown(response)
                
        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("👈 Please upload a PDF document in the sidebar to get started.")
