import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(
    page_title="AI Document Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS - Premium Dark Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ── */
    * { font-family: 'Inter', sans-serif !important; }

    .stApp {
        background: linear-gradient(135deg, #0F0F1A 0%, #1A1A2E 50%, #16213E 100%);
    }

    /* ── Hide Streamlit Defaults ── */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A1A2E 0%, #16213E 100%) !important;
        border-right: 1px solid rgba(124, 58, 237, 0.2);
    }

    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        background: linear-gradient(135deg, #7C3AED, #A855F7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }

    /* ── File Uploader ── */
    section[data-testid="stFileUploader"] {
        border: 2px dashed rgba(124, 58, 237, 0.4) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        background: rgba(124, 58, 237, 0.05) !important;
        transition: all 0.3s ease;
    }

    section[data-testid="stFileUploader"]:hover {
        border-color: rgba(124, 58, 237, 0.8) !important;
        background: rgba(124, 58, 237, 0.1) !important;
    }

    /* ── Chat Messages ── */
    .stChatMessage[data-testid="stChatMessage"] {
        border-radius: 16px !important;
        padding: 16px 20px !important;
        margin-bottom: 12px !important;
        border: 1px solid rgba(124, 58, 237, 0.1) !important;
        backdrop-filter: blur(10px);
    }

    /* User message */
    .stChatMessage[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.15), rgba(168, 85, 247, 0.08)) !important;
        border: 1px solid rgba(124, 58, 237, 0.25) !important;
    }

    /* Assistant message */
    .stChatMessage[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: rgba(30, 30, 60, 0.6) !important;
        border: 1px solid rgba(100, 100, 180, 0.15) !important;
    }

    /* ── Chat Input ── */
    .stChatInput > div {
        border-radius: 16px !important;
        border: 1px solid rgba(124, 58, 237, 0.3) !important;
        background: rgba(26, 26, 46, 0.8) !important;
        transition: all 0.3s ease;
    }

    .stChatInput > div:focus-within {
        border-color: #7C3AED !important;
        box-shadow: 0 0 20px rgba(124, 58, 237, 0.2) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #7C3AED, #A855F7) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.4) !important;
    }

    /* ── Success / Info Alerts ── */
    .stAlert {
        border-radius: 12px !important;
        border: none !important;
    }

    div[data-testid="stAlert"][data-baseweb="notification"] {
        border-radius: 12px !important;
    }

    /* ── Spinner ── */
    .stSpinner > div > div {
        border-top-color: #7C3AED !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0F0F1A; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #7C3AED, #A855F7);
        border-radius: 10px;
    }

    /* ── Hero Section ── */
    .hero-container {
        text-align: center;
        padding: 60px 20px 40px 20px;
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #7C3AED, #A855F7, #C084FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 12px;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #94A3B8;
        font-weight: 300;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }

    .hero-divider {
        width: 80px;
        height: 3px;
        background: linear-gradient(90deg, #7C3AED, #A855F7);
        margin: 24px auto;
        border-radius: 2px;
    }

    /* ── Feature Cards ── */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        padding: 0 20px;
        max-width: 800px;
        margin: 0 auto 40px auto;
    }

    .feature-card {
        background: rgba(26, 26, 46, 0.6);
        border: 1px solid rgba(124, 58, 237, 0.15);
        border-radius: 16px;
        padding: 24px 16px;
        text-align: center;
        transition: all 0.3s ease;
    }

    .feature-card:hover {
        border-color: rgba(124, 58, 237, 0.4);
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(124, 58, 237, 0.15);
    }

    .feature-icon {
        font-size: 2rem;
        margin-bottom: 10px;
    }

    .feature-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #C084FC;
        margin-bottom: 6px;
    }

    .feature-desc {
        font-size: 0.75rem;
        color: #64748B;
        line-height: 1.4;
    }

    /* ── Sidebar Info Card ── */
    .sidebar-info {
        background: rgba(124, 58, 237, 0.08);
        border: 1px solid rgba(124, 58, 237, 0.2);
        border-radius: 12px;
        padding: 16px;
        margin-top: 20px;
    }

    .sidebar-info-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #A855F7;
        margin-bottom: 8px;
    }

    .sidebar-info-text {
        font-size: 0.75rem;
        color: #94A3B8;
        line-height: 1.5;
    }

    /* ── Footer ── */
    .app-footer {
        text-align: center;
        padding: 30px 0;
        color: #475569;
        font-size: 0.8rem;
    }

    .app-footer a {
        color: #7C3AED;
        text-decoration: none;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Upload Document")
    pdf_file = st.file_uploader(
        "Drag & drop your PDF here",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if pdf_file is not None:
        st.success(f"✅ **{pdf_file.name}** uploaded!")
        file_size = round(pdf_file.size / 1024, 1)
        st.caption(f"📄 Size: {file_size} KB")

    st.markdown("""
    <div class="sidebar-info">
        <div class="sidebar-info-title">💡 How it works</div>
        <div class="sidebar-info-text">
            1. Upload any PDF document<br>
            2. AI processes & indexes the content<br>
            3. Ask questions in natural language<br>
            4. Get accurate, context-aware answers
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding: 8px 0;">
        <span style="color: #64748B; font-size: 0.75rem;">
            Built with Streamlit • LangChain • Gemini
        </span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PDF Processing
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def process_pdf(file):
    """Reads PDF, extracts text, chunks it, and creates FAISS Vector Store."""
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store


# ─────────────────────────────────────────────
# Initialize Chat History
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ─────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────
if pdf_file:
    with st.spinner("🔄 Processing document — extracting, chunking & embedding..."):
        vector_store = process_pdf(pdf_file)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_query = st.chat_input("💬 Ask anything about your document...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching document..."):
                docs = vector_store.similarity_search(user_query, k=3)
                llm = ChatGoogleGenerativeAI(model="models/gemini-3.5-flash", temperature=0.3)
                chain = load_qa_chain(llm, chain_type="stuff")
                response = chain.run(input_documents=docs, question=user_query)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    # ── Landing Page ──
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">AI Document Search</div>
        <div class="hero-divider"></div>
        <div class="hero-subtitle">
            Upload any PDF and have intelligent conversations with your documents.
            Powered by RAG (Retrieval-Augmented Generation) and Google Gemini AI.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="features-grid">
        <div class="feature-card">
            <div class="feature-icon">📄</div>
            <div class="feature-title">Smart Upload</div>
            <div class="feature-desc">Drop any PDF and it's instantly processed</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">AI-Powered</div>
            <div class="feature-desc">Semantic search with vector embeddings</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">💬</div>
            <div class="feature-title">Chat Interface</div>
            <div class="feature-desc">Ask questions in natural language</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="app-footer">
        Built by <a href="https://github.com/Rishikesan05" target="_blank">Rishikesan</a> •
        Powered by Google Gemini & LangChain
    </div>
    """, unsafe_allow_html=True)
