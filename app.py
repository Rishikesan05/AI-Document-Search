import streamlit as st
import time
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
    layout="centered",
    initial_sidebar_state="collapsed"
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

    html, body, .stApp, [data-testid="stAppViewContainer"] {
        background-color: var(--bg) !important;
        color: var(--text) !important;
        overflow-x: hidden !important;
    }
    
    /* Reduce default Streamlit padding to prevent unwanted vertical scrolling and massive side margins on mobile */
    [data-testid="stMainBlockContainer"] {
        padding: 1rem 1rem 2rem 1rem !important;
        max-width: 100% !important;
    }

    #MainMenu, [data-testid="stHeader"], [data-testid="stFooter"] { 
        display: none !important; 
    }
    
    /* Hide Streamlit Cloud Deploy, Viewer, and Creator Badges */
    .stDeployButton,
    [data-testid="viewerBadge"],
    [data-testid="creatorBadge"],
    [data-testid="stDecoration"],
    [data-testid="stToolbar"],
    div[class^="viewerBadge_container"],
    div[class*="viewerBadge_container"],
    div[class^="creatorBadge"],
    div[class*="creatorBadge"],
    a[href*="streamlit.io/cloud"] {
        display: none !important;
    }
    
    /* Hide default Streamlit header anchor links (the 🔗 icon) */
    a.header-anchor,
    [data-testid="stMarkdownContainer"] h1 a,
    [data-testid="stMarkdownContainer"] h2 a,
    [data-testid="stMarkdownContainer"] h3 a,
    [data-testid="stMarkdownContainer"] h4 a,
    [data-testid="stMarkdownContainer"] h5 a,
    [data-testid="stMarkdownContainer"] h6 a {
        display: none !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: var(--bg2) !important;
        border-right: 0.8px solid var(--border) !important;
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--text2) !important;
        font-size: 13px;
    }

    /* File Uploader styling removed to prevent conflicts with Streamlit 1.38 native layout */

    /* Chat Messages */
    .stChatMessage[data-testid="stChatMessage"] {
        border-radius: var(--radius) !important;
        padding: 20px 24px !important;
        margin-bottom: 12px !important;
        font-size: 13px;
        line-height: 1.6;
    }

    .stChatMessage[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]),
    .stChatMessage[data-testid="stChatMessage"]:has([data-testid*="user"]) {
        background: var(--bg2) !important;
        border: 0.8px solid var(--border) !important;
    }

    .stChatMessage[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]),
    .stChatMessage[data-testid="stChatMessage"]:has([data-testid*="assistant"]) {
        background: var(--bg) !important;
        border: 0.8px solid var(--border) !important;
    }

    /* Minimalist Monochrome Chat Avatars (Replaces bright red/orange/green colors) */
    [data-testid="stChatMessageAvatarContainer"],
    [data-testid="stChatMessageAvatarCustom"],
    [data-testid="chatAvatarIcon-user"],
    [data-testid="chatAvatarIcon-assistant"] {
        background-color: #f0f0f0 !important;
        color: #1a1a1a !important;
        border: 0.8px solid var(--border) !important;
        border-radius: 8px !important;
    }

    [data-testid="chatAvatarIcon-user"] svg,
    [data-testid="chatAvatarIcon-assistant"] svg,
    [data-testid="stChatMessageAvatarContainer"] svg {
        fill: #1a1a1a !important;
        color: #1a1a1a !important;
    }

    /* Chat Input submit button & icons */
    [data-testid="stChatInputSubmitButton"] {
        color: #1a1a1a !important;
    }
    
    [data-testid="stChatInputSubmitButton"] svg {
        fill: #1a1a1a !important;
        color: #1a1a1a !important;
    }

    .stChatInput [data-testid="stIconMaterial"],
    .stChatInput [data-testid="stChatMessageAvatarContainer"] {
        background-color: #f0f0f0 !important;
        color: #1a1a1a !important;
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
    .stButton > button,
    [data-testid="stDownloadButton"] > button {
        border-radius: 50px !important;
        padding: 8px 16px !important;
        font-weight: 500 !important;
        font-size: 12px !important;
        transition: all 0.2s ease;
        cursor: pointer;
        min-height: 0 !important;
        line-height: 1.4 !important;
        white-space: nowrap !important;
    }

    /* Secondary Buttons (Chips / Pills) */
    .stButton > button[data-testid="baseButton-secondary"],
    [data-testid="stDownloadButton"] > button {
        background-color: var(--bg) !important;
        color: var(--text) !important;
        border: 0.8px solid var(--border) !important;
    }

    .stButton > button[data-testid="baseButton-secondary"]:hover,
    [data-testid="stDownloadButton"] > button:hover {
        border-color: var(--text) !important;
        background-color: var(--bg2) !important;
    }

    /* Primary Buttons */
    .stButton > button[data-testid="baseButton-primary"] {
        background-color: var(--accent) !important;
        color: #ffffff !important;
        border: none !important;
    }

    .stButton > button[data-testid="baseButton-primary"]:hover {
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
        padding: 32px 24px 48px 24px;
        max-width: 640px;
        margin: 0 auto;
    }

    .hero-kicker {
        font-size: 12px;
        font-weight: 500;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--text2);
        margin: 0 0 16px 0 !important;
    }

    .hero-title {
        font-size: clamp(16px, 10vw, 36px);
        font-weight: 700;
        line-height: 1.3;
        color: var(--text);
        margin: 0 0 20px 0;
        letter-spacing: -0.5px;
        word-break: keep-all !important;
        overflow-wrap: normal !important;
        white-space: pre-wrap !important;
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
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: var(--accent-soft);
        border-radius: 50px;
        padding: 10px;
        margin-bottom: 14px;
        color: var(--text);
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
    
    /* Responsive Design for Mobile */
    @media (max-width: 768px) {
        .features {
            grid-template-columns: 1fr;
            gap: 12px;
            padding: 0 16px;
        }
        
        .hero-title {
            font-size: 28px;
        }
        
        .hero {
            padding: 16px 16px 24px 16px;
        }
    }
    
    /* Extreme narrow screens (e.g., resized desktop windows < 400px) */
    @media (max-width: 400px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1rem 0.5rem !important;
        }
        
        .hero {
            padding: 24px 4px 12px 4px;
        }
        
        .hero-title {
            font-size: clamp(14px, 12vw, 24px);
        }
    }
</style>
""", unsafe_allow_html=True)


# ── PDF Processing ──
@st.cache_resource(show_spinner=False)
def process_pdfs(files):
    """Read multiple PDFs, chunk text, build FAISS vector store."""
    text = ""
    for file in files:
        reader = PdfReader(file)
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
if not st.session_state.messages:
    st.markdown("""
    <div class="hero">
        <p class="hero-kicker">AI Powered</p>
        <div class="hero-title">Search your documents<br>with intelligence</div>
        <p class="hero-desc">
            Upload PDFs and ask questions in plain language.
            Powered by retrieval-augmented generation.
            Answers are grounded entirely in your documents.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
pdf_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

if pdf_files:
    total_size = sum([f.size for f in pdf_files])
    file_size = round(total_size / 1024, 1)
    
    with st.spinner("Processing documents..."):
        vector_store = process_pdfs(pdf_files)

    # ── Quick Prompts ──
    if not st.session_state.messages:
        cols = st.columns(3)
        with cols[0]:
            if st.button("⊡  Summarize Key Points", use_container_width=True):
                st.session_state.quick_query = "Summarize the key points of the uploaded documents."
        with cols[1]:
            if st.button("✓  Extract Action Items", use_container_width=True):
                st.session_state.quick_query = "Extract the main action items or tasks mentioned in the documents."
        with cols[2]:
            if st.button("◷  Find Important Dates", use_container_width=True):
                st.session_state.quick_query = "List any important dates, deadlines, or schedules mentioned."

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if "failed_query" in st.session_state:
        # Show the user's message that failed so it doesn't disappear
        with st.chat_message("user"):
            st.markdown(st.session_state.failed_query)
            
        error_msg = st.session_state.get("last_error", "")
        if "429" in error_msg or "ResourceExhausted" in error_msg:
            st.error("**Rate Limit Exceeded:** Please wait 30 seconds and try again.")
        else:
            st.error(f"**Error generating response:** {error_msg}")

    # ── Action Controls (Bottom of chat) ──
    if st.session_state.messages or "failed_query" in st.session_state:
        st.markdown("<br>", unsafe_allow_html=True)
        has_error = "failed_query" in st.session_state
        if has_error:
            ctrl_cols = st.columns([2.5, 2.5, 2.5, 2.5])
        else:
            ctrl_cols = st.columns([2.5, 2.5, 5])
            
        col_idx = 0
        if has_error:
            with ctrl_cols[col_idx]:
                if st.button("↻ Retry", key="retry_btn", use_container_width=True):
                    st.session_state.retry_query = st.session_state.failed_query
                    del st.session_state.failed_query
                    st.rerun()
            col_idx += 1
            
        with ctrl_cols[col_idx]:
            chat_export = ""
            for m in st.session_state.messages:
                role = "User" if m["role"] == "user" else "AI"
                chat_export += f"{role}: {m['content']}\n\n"
            st.download_button(
                label="⤓ Export Chat",
                data=chat_export if chat_export else "No messages.",
                file_name="chat_history.txt",
                mime="text/plain",
                use_container_width=True
            )
        col_idx += 1
            
        with ctrl_cols[col_idx]:
            if st.button("🗑 Clear Chat", use_container_width=True):
                st.session_state.messages = []
                if "failed_query" in st.session_state:
                    del st.session_state.failed_query
                st.rerun()

    user_query = st.chat_input("Ask a question about your documents...")
    
    if "quick_query" in st.session_state:
        user_query = st.session_state.quick_query
        del st.session_state.quick_query
        
    if "retry_query" in st.session_state:
        user_query = st.session_state.retry_query
        del st.session_state.retry_query

    if user_query:
        if "failed_query" in st.session_state:
            del st.session_state.failed_query
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            start_time = time.time()
            with st.spinner("Searching..."):
                docs = vector_store.similarity_search(user_query, k=3)
                context = "\n\n".join([doc.page_content for doc in docs])
                prompt = f"Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\nContext:\n{context}\n\nQuestion: {user_query}\n\nHelpful Answer:"
                
                llm = ChatGoogleGenerativeAI(
                    model="models/gemini-1.5-flash",
                    temperature=0.3
                )
                
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                for chunk in llm.stream(prompt):
                    full_response += chunk.content
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                
                latency = time.time() - start_time
                
                st.markdown(f"<p style='font-size: 11px; color: var(--text3); margin-top: 8px;'>⚡ Retrieved in {latency:.2f}s · {len(docs)} chunks analyzed</p>", unsafe_allow_html=True)
                
                with st.expander("View Retrieved Context"):
                    for i, doc in enumerate(docs):
                        st.markdown(f"**Chunk {i+1}**")
                        st.info(doc.page_content)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response
                })
            except Exception as e:
                failed_msg = st.session_state.messages.pop()
                st.session_state.failed_query = failed_msg["content"]
                st.session_state.last_error = str(e)
                st.rerun()

else:
    st.markdown("""
    <div class="features">
        <div class="feat">
            <span class="feat-icon">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
            </span>
            <p class="feat-title">Upload</p>
            <p class="feat-desc">Drop any PDF into the uploader. Text is extracted and chunked automatically.</p>
        </div>
        <div class="feat">
            <span class="feat-icon">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
            </span>
            <p class="feat-title">Search</p>
            <p class="feat-desc">FAISS builds a local vector index. No data leaves your machine.</p>
        </div>
        <div class="feat">
            <span class="feat-icon">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
            </span>
            <p class="feat-title">Converse</p>
            <p class="feat-desc">Ask follow-up questions naturally. Gemini answers from retrieved context.</p>
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
