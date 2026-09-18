# AI Document Search - RAG Chatbot

**AI Document Search** is an intelligent conversational assistant designed to help you interact with your PDF documents seamlessly. Built purely in Python, the platform focuses on extracting context-aware answers directly from your files using semantic search and Large Language Models.

## 🚀 Key Features
- **Document Upload**: Easily upload any PDF document via a clean sidebar interface.
- **Automated Processing**: Instant text extraction, intelligent chunking, and vector embedding creation.
- **Local Vector Database**: Fast, offline semantic search powered by FAISS.
- **Conversational Chat**: Ask questions naturally and receive accurate answers derived *only* from the uploaded document's content.

## 🛠️ Technical Stack
- **Framework**: Streamlit (Python Web UI)
- **Document Processing**: PyPDF2 & LangChain Text Splitters
- **Database**: FAISS (Facebook AI Similarity Search)
- **AI Models**: Google Gemini (`gemini-3.5-flash` for generation, `gemini-embedding-2` for embeddings)

## 💻 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/Rishikesan05/AI-Document-Search.git
cd AI-Document-Search
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Setup Environment:
Rename `.env.example` to `.env` and add your Google API Key:
```env
GOOGLE_API_KEY=your_api_key_here
```

4. Run the app:
```bash
streamlit run app.py
```
*(The app will launch in your browser at `http://localhost:8501`)*
