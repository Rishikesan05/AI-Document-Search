# AI Document Search — Conversational PDF RAG Assistant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://github.com/Rishikesan05/AI-Document-Search)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/Orchestration-LangChain-green)](https://python.langchain.com/)
[![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS%20Index-orange)](https://github.com/facebookresearch/faiss)
[![Gemini](https://img.shields.io/badge/LLM-Google%20Gemini%203.5%20Flash-blue)](https://ai.google.dev/)
[![Design](https://img.shields.io/badge/UI-Studio%20Minimalist%20%7C%20Roboto%20Mono-black)](https://rishiware.com/)

A production-grade, privacy-first **Document Intelligence & Conversational RAG (Retrieval-Augmented Generation)** assistant. While generic conversational AI models hallucinate or lack access to private documentation, **AI Document Search** ingests, chunks, and semantically indexes multiple PDF documents completely in-process using Meta's **FAISS** vector store and **gemini-embedding-2**.

On every query, the agent performs sub-second vector similarity retrieval to supply grounded source chunks directly into the prompt context. It generates factual, hallucination-free answers via real-time token streaming with **Google Gemini 3.5 Flash**, accompanied by full source verification, latency performance metrics, 1-click quick analysis prompts, and a bespoke Studio Minimalist design system inspired by [rishiware.com](https://rishiware.com/).

---

## 🏗️ Architecture Overview

```text
               Multiple PDF Documents
                         │
                         ▼
             ┌─────────────────────────┐
             │ PyPDF2 Document Loader  │ ◄── Aggregates multi-page text
             └───────────┬─────────────┘
                         │
                         ▼
             ┌─────────────────────────┐
             │ Recursive Text Splitter │ ◄── Chunk Size: 1000 | Overlap: 200
             └───────────┬─────────────┘
                         │
                         ▼
             ┌─────────────────────────┐
             │   gemini-embedding-2    │ ◄── Dense Vector Transformation
             └───────────┬─────────────┘
                         │
                         ▼
             ┌─────────────────────────┐
             │     FAISS Vector DB     │ ◄── In-Memory Dense Semantic Index
             └───────────┬─────────────┘
                         │
    User Query / Quick Prompts (Summary, Actions, Dates, Humanize)
                         │
                         ▼
             ┌─────────────────────────┐
             │ Top-K Similarity Search │ ◄── Retrieves k=3 most relevant chunks
             └───────────┬─────────────┘
                         │
                         ▼
             ┌─────────────────────────┐
             │ Augmented Prompt Context│ ◄── Grounded Prompt: "Use context... don't make up"
             └───────────┬─────────────┘
                         │
                         ▼
             ┌─────────────────────────┐
             │ Gemini 3.5 Flash Stream │ ◄── Real-Time Token Streaming (temp: 0.3)
             └───────────┬─────────────┘
                         │
           ┌─────────────┴─────────────┐
           ▼                           ▼
    Streaming Response         Source Verification Drawer
(Latency & Chunk Count)       (Inspect Raw Text Chunks)
```

---

## ⚡ Technical Specifications

| Component | Technology | Role & Purpose |
|---|---|---|
| **Primary LLM** | **Google Gemini 3.5 Flash** | Low-latency, instruction-following generation (`temperature=0.3`) |
| **Embedding Engine** | **gemini-embedding-2** | Dense vector representations capturing deep semantic relationships |
| **Vector Database** | **FAISS (Meta)** | Local, in-memory vector index delivering sub-second similarity searches |
| **Document Ingestion** | **PyPDF2** | Multi-file text extraction across all PDF pages |
| **Chunking Strategy** | **RecursiveCharacterTextSplitter** | `chunk_size=1000`, `chunk_overlap=200`, preserving semantic boundaries |
| **Orchestration** | **LangChain** | Document loaders, text splitters, vector stores, and prompt pipelines |
| **Design System** | **Streamlit + Custom CSS** | Studio Minimalist monochrome theme, `Roboto Mono` typography, mobile responsive down to 390px |
| **Error Resilience** | **Session-State Retry Handler** | Graceful handling for Gemini `429 Quota Exceeded` without chat loss |

---

## 🌟 Core Features

### 1. Multi-Document Aggregation & Instant Processing
- **Simultaneous Uploads:** Drag and drop multiple PDF files at once. The processing pipeline dynamically aggregates text across all pages and documents into a unified semantic index.
- **Resource Caching:** Processed vector stores are cached via `@st.cache_resource` to prevent unnecessary re-embedding across re-runs.

### 2. High-Precision Local Semantic Search (FAISS)
- **Sub-Second Retrieval:** Meta's FAISS library executes vector similarity searches over dense embeddings in milliseconds.
- **True Semantic Understanding:** Matches queries by conceptual meaning rather than exact keywords (e.g., querying *"financial penalties"* retrieves sections discussing *"indemnity liabilities and fines"*).

### 3. Real-Time Token Streaming
- **Zero Latency Perception:** Tokens stream progressively onto the screen with an interactive typing cursor (`▌`), eliminating wait-time friction.
- **Conservative Generation (`temperature=0.3`):** Configured with low temperature to guarantee strict factual alignment with the uploaded documents and eliminate hallucinations.

### 4. Source Context Verification Drawer
- **Explainable AI:** Every response includes an expandable **"View Retrieved Context"** drawer.
- **Chunk Inspection:** Users can inspect the exact raw text chunks retrieved from the document that informed the AI's answer, ensuring complete transparency and compliance.

### 5. Performance & Retrieval Latency Tracker
- **Real-Time Metrics:** Each response displays an inline performance badge:
  $$\text{⚡ Retrieved in } X.XX\text{s } \cdot \ K \text{ chunks analyzed}$$
- Provides transparent feedback on retrieval and inference speed.

### 6. 1-Click Quick Prompts
Get instant insights without typing a single word:
- **✧ Summary:** Generates a high-level executive summary of key points.
- **✓ Actions:** Extracts critical action items, tasks, and deliverables.
- **◷ Dates:** Highlights important dates, milestones, deadlines, and schedules.
- **✎ Humanize:** Rewrites complex technical or legal text in clear, accessible language.

### 7. Graceful Rate-Limit & Error Handling
- **429 / ResourceExhausted Resilience:** Catches API rate limits cleanly without crashing or displaying Python tracebacks.
- **State Preservation & 1-Click Retry:** If an error occurs, the user's failed query is preserved in the interface alongside a dedicated **↻ Retry** button.

### 8. Conversation Management & Clean Export
- **Export Chat:** Download the full conversation transcript as a clean `.txt` file. Markdown syntax (`*`, `_`, `#`, `` ` ``) is stripped via regex for clean plain-text reading.
- **Clear Chat & PDF Reset:** 1-click **🗑 Clear Chat** resets the conversation history and increments the uploader key to wipe cached files cleanly.

### 9. Studio Minimalist Design System
- **Custom Aesthetic:** Inspired by [rishiware.com](https://rishiware.com/), featuring custom CSS custom properties, monochrome avatars, and a clean card grid.
- **Typography:** Fully integrated with Google's **Roboto Mono** monospace typography.
- **Certified Mobile Responsive:** Custom media queries ensure responsive layouts across desktop, tablet, and mobile screens down to **390px** viewport width.

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10+**
- A free **Google Gemini API Key** ([Get your API key](https://aistudio.google.com/apikey))

### 1. Clone the Repository
```bash
git clone https://github.com/Rishikesan05/AI-Document-Search.git
cd AI-Document-Search
```

### 2. Set Up a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory:
```bash
cp .env.example .env
```
Inside `.env`:
```env
GOOGLE_API_KEY="your_actual_gemini_api_key_here"
```

### 5. Run the Application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## ☁️ Streamlit Cloud Deployment

This repository is pre-configured for instant deployment on **Streamlit Community Cloud**:

1. Push your repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/).
3. Click **New app**, select your repository (`AI-Document-Search`), branch (`main`), and main file path (`app.py`).
4. In **Advanced settings &rarr; Secrets**, add your API key:
   ```toml
   GOOGLE_API_KEY = "your_actual_api_key_here"
   ```
5. Click **Deploy!** Your RAG Document Assistant will be live and accessible from anywhere.

---

## 📁 Project Structure

```text
AI-Document-Search/
├── app.py                     # Main application (UI, PDF processing, RAG pipeline, CSS)
├── requirements.txt           # Python dependencies (Streamlit, LangChain, FAISS, PyPDF2)
├── .env.example               # Environment variable template
├── .streamlit/
│   └── config.toml            # Server settings and Streamlit theme configuration
└── README.md                  # Comprehensive project documentation
```

---

## 👨‍💻 Author

Developed by **Rishikesan**
- **Portfolio:** [rishiware.com](https://rishiware.com)
- **LinkedIn:** [linkedin.com/in/rishikesan05](https://linkedin.com/in/rishikesan05)
- **GitHub:** [github.com/Rishikesan05](https://github.com/Rishikesan05)
