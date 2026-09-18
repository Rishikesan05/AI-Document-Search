# AI Document Search - RAG Chatbot

**AI Document Search** is an intelligent, minimalist conversational assistant designed to help you interact with your PDF documents seamlessly. Built purely in Python, the platform extracts context-aware answers directly from your files using semantic search and Large Language Models, all wrapped in a clean, mobile-responsive UI inspired by [Rishiware](https://rishiware.com/).

## 🚀 Key Features
- **Multiple Document Upload**: Drag and drop multiple PDF documents simultaneously.
- **Real-Time Streaming**: Watch answers generate in real-time with zero lag using Gemini's streaming API.
- **Source Context & Citations**: Verify the AI's answers instantly by expanding the "View Retrieved Context" section to see exact document chunks.
- **Performance Tracking**: Built-in latency tracker displays retrieval time and the number of chunks analyzed for every query.
- **Automated Processing**: Instant text extraction, intelligent chunking, and local vector embedding creation (FAISS).
- **Graceful Error Handling**: Automatically catches and handles Gemini API rate limits (429 Quota Exceeded) without crashing.
- **Chat Management**: Easily clear conversation history or export your chat logs with single-click UI chips.

## 🛠️ Technical Stack
- **Frontend/UI**: Streamlit with Custom CSS (Mobile-responsive, Minimalist Design)
- **Document Processing**: PyPDF2 & LangChain Text Splitters
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **AI Models**: Google Gemini (`gemini-3.5-flash` for generation, `gemini-embedding-2` for embeddings)

## 💻 Getting Started

1. **Clone the repository:**
```bash
git clone https://github.com/Rishikesan05/AI-Document-Search.git
cd AI-Document-Search
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Setup Environment:**
Rename `.env.example` to `.env` (or create a new `.env` file) and add your Google API Key:
```env
GOOGLE_API_KEY=your_api_key_here
```

4. **Run the app:**
```bash
streamlit run app.py
```
*(The app will launch in your browser at `http://localhost:8501`)*

---
*Created by [Rishikesan](https://rishiware.com)*
