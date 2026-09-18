import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AI Document Search",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS Design System (rishiware.com style) ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;500;600;700&display=swap');

    :root {
        --bg: #ffffff;
        --bg2: #f8f8f8;
        --card: #ffffff;
        --text: #1a1a1a;
        --text2: #555555;
        --text3: #888888;
        --border: #e0e0e0;
        --accent: #1a1a1a;
        --accent-soft: rgba(0, 0, 0, 0.06);
        --radius: 16px;
        --radius-lg: 24px;
    }

    * {
        font-family: 'Roboto Mono', monospace !important;
    }

    html, body, .stApp {
        background-color: var(--bg) !important;
        color: var(--text) !important;
        overflow-x: hidden !important;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: var(--bg2) !important;
        border-right: 0.8px solid var(--border) !important;
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--text2) !important;
        font-size: 13px;
    }

    /* File Uploader */
    section[data-testid="stFileUploader"] {
        border: 1px dashed var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 24px 16px !important;
        background: var(--bg) !important;
        transition: border-color 0.2s ease;
    }

    section[data-testid="stFileUploader"]:hover {
        border-color: var(--accent) !important;
    }

    /* Chat Messages */
    .stChatMessage[data-testid="stChatMessage"] {
        border-radius: var(--radius) !important;
        padding: 20px 24px !important;
        margin-bottom: 12px !important;
        font-size: 13px;
        line-height: 1.6;
    }

    .stChatMessage[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: var(--bg2) !important;
        border: 0.8px solid var(--border) !important;
    }

    .stChatMessage[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: var(--bg) !important;
        border: 0.8px solid var(--border) !important;
    }

    /* Chat Input */
    .stChatInput > div {
        border-radius: var(--radius) !important;
        border: 0.8px solid var(--border) !important;
        background: var(--bg) !important;
        transition: border-color 0.2s ease;
    }

    .stChatInput > div:focus-within {
        border-color: var(--accent) !important;
        box-shadow: none !important;
    }

    .stChatInput textarea {
        color: var(--text) !important;
        font-size: 13px !important;
    }

    .stChatInput textarea::placeholder {
        color: var(--text3) !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: var(--accent) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 10px 24px !important;
        font-weight: 500 !important;
        font-size: 13px !important;
        transition: opacity 0.15s ease;
        cursor: pointer;
    }

    .stButton > button:hover {
        opacity: 0.85 !important;
    }

    /* Alerts */
    .stAlert > div {
        border-radius: var(--radius) !important;
        font-size: 13px;
    }

    /* Spinner */
    .stSpinner > div > div {
        border-top-color: var(--accent) !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

    /* Hero Section */
    .hero {
        text-align: center;
        padding: 96px 24px 48px 24px;
        max-width: 640px;
        margin: 0 auto;
    }

    .hero-kicker {
        font-size: 12px;
        font-weight: 500;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--text2);
        margin: 0 0 16px 0;
    }

    .hero-title {
        font-size: 36px;
        font-weight: 700;
        line-height: 1.2;
        color: var(--text);
        margin: 0 0 20px 0;
        letter-spacing: -0.5px;
    }

    .hero-desc {
        font-size: 14px;
        font-weight: 400;
        line-height: 1.7;
        color: var(--text2);
        margin: 0;
    }

    /* Feature Cards Grid */
    .features {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        max-width: 720px;
        margin: 48px auto 0 auto;
        padding: 0 24px;
    }

    .feat {
        background: var(--card);
        border: 0.8px solid var(--border);
        border-radius: var(--radius);
        padding: 24px 20px;
    }

    .feat-icon {
        display: inline-block;
        background: var(--accent-soft);
        border-radius: 50px;
        padding: 6px 12px;
        font-size: 14px;
        margin-bottom: 14px;
    }

    .feat-title {
        font-size: 14px;
        font-weight: 600;
        color: var(--text);
        margin: 0 0 6px 0;
    }

    .feat-desc {
        font-size: 12px;
        font-weight: 400;
        color: var(--text2);
        line-height: 1.6;
        margin: 0;
    }

    /* Tech Stack Pills */
    .stack {
        display: flex;
        justify-content: center;
        gap: 8px;
        flex-wrap: wrap;
        max-width: 640px;
        margin: 40px auto 0 auto;
        padding: 0 24px;
    }

    .pill {
        display: inline-block;
        background: var(--accent-soft);
        color: var(--text);
        font-size: 11px;
        font-weight: 500;
        padding: 5px 14px;
        border-radius: 50px;
    }

    /* Sidebar Sections */
    .sb-brand {
        font-size: 16px;
        font-weight: 700;
        color: var(--text);
        margin: 0 0 4px 0;
    }

    .sb-sub {
        font-size: 11px;
        font-weight: 400;
        color: var(--text3);
        margin: 0 0 24px 0;
    }

    .sb-section {
        background: var(--bg);
        border: 0.8px solid var(--border);
        border-radius: var(--radius);
        padding: 20px;
        margin-top: 20px;
    }

    .sb-section-title {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: var(--text2);
        margin: 0 0 12px 0;
    }

    .sb-step {
        font-size: 12px;
        color: var(--text2);
        line-height: 2;
        margin: 0;
    }

    .sb-step strong {
        color: var(--text);
    }

    /* Footer */
    .ft {
        text-align: center;
        padding: 48px 0 24px 0;
        font-size: 12px;
        color: var(--text3);
        display: flex;
        flex-direction: column;
        gap: 8px;
    }

    .ft-links {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-top: 10px;
    }

    .ft a {
        color: var(--text3);
        text-decoration: none;
        transition: color 0.2s ease;
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }

    .ft a:hover { 
        color: var(--text);
    }
    
    .ft svg {
        width: 18px;
        height: 18px;
    }

    /* Sidebar footer */
    .sb-footer {
        display: flex;
        justify-content: center;
        gap: 6px;
        flex-wrap: wrap;
        margin-top: 16px;
    }

    .sb-footer-pill {
        font-size: 10px;
        font-weight: 500;
        color: var(--text3);
        background: var(--accent-soft);
        padding: 3px 10px;
        border-radius: 50px;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──
with st.sidebar:
    st.markdown("""
    <p class="sb-brand">📄 Document Search</p>
    <p class="sb-sub">AI-powered PDF analysis tool</p>
    """, unsafe_allow_html=True)

    pdf_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if pdf_file is not None:
        file_size = round(pdf_file.size / 1024, 1)
        st.success(f"**{pdf_file.name}** ready ({file_size} KB)")

    st.markdown("""
    <div class="sb-section">
        <p class="sb-section-title">How it works</p>
        <p class="sb-step">
            <strong>1.</strong> Upload any PDF document<br>
            <strong>2.</strong> AI reads and indexes every page<br>
            <strong>3.</strong> Ask questions in plain English<br>
            <strong>4.</strong> Get answers from your document
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="sb-footer">
        <span class="sb-footer-pill">Streamlit</span>
        <span class="sb-footer-pill">LangChain</span>
        <span class="sb-footer-pill">Gemini</span>
        <span class="sb-footer-pill">FAISS</span>
    </div>
    """, unsafe_allow_html=True)


# ── PDF Processing ──
@st.cache_resource(show_spinner=False)
def process_pdf(file):
    """Read PDF, chunk text, build FAISS vector store."""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(text)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    return FAISS.from_texts(chunks, embeddings)


if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Main ──
if pdf_file:
    with st.spinner("Processing document..."):
        vector_store = process_pdf(pdf_file)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input("Ask a question about your document...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                docs = vector_store.similarity_search(user_query, k=3)
                llm = ChatGoogleGenerativeAI(
                    model="models/gemini-3.5-flash",
                    temperature=0.3
                )
                chain = load_qa_chain(llm, chain_type="stuff")
                response = chain.invoke(
                    {"input_documents": docs, "question": user_query}
                )
                st.markdown(response["output_text"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": response["output_text"]
        })

else:
    st.markdown("""
    <div class="hero">
        <p class="hero-kicker">AI Powered</p>
        <h1 class="hero-title">Search your documents<br>with intelligence</h1>
        <p class="hero-desc">
            Upload a PDF and ask questions in plain language.
            Powered by retrieval-augmented generation.
            Answers are grounded entirely in your document.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="features">
        <div class="feat">
            <span class="feat-icon">📄</span>
            <p class="feat-title">Upload</p>
            <p class="feat-desc">Drop any PDF into the sidebar. Text is extracted and chunked automatically.</p>
        </div>
        <div class="feat">
            <span class="feat-icon">🔍</span>
            <p class="feat-title">Search</p>
            <p class="feat-desc">FAISS builds a local vector index. No data leaves your machine.</p>
        </div>
        <div class="feat">
            <span class="feat-icon">💬</span>
            <p class="feat-title">Converse</p>
            <p class="feat-desc">Ask follow-up questions naturally. Gemini answers from retrieved context.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stack">
        <span class="pill">Python</span>
        <span class="pill">Streamlit</span>
        <span class="pill">LangChain</span>
        <span class="pill">FAISS</span>
        <span class="pill">Google Gemini</span>
        <span class="pill">PyPDF2</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="ft">
        <span>Built by <strong>Rishikesan</strong></span>
        <div class="ft-links">
            <a href="https://rishiware.com/" target="_blank" title="Portfolio">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path></svg>
            </a>
            <a href="https://www.linkedin.com/in/rishikesan05/" target="_blank" title="LinkedIn">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"></path><rect x="2" y="9" width="4" height="12"></rect><circle cx="4" cy="4" r="2"></circle></svg>
            </a>
            <a href="https://github.com/Rishikesan05" target="_blank" title="GitHub">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path></svg>
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)
