# AI Document Search (RAG Chatbot)

An intelligent conversational assistant that allows users to chat with PDF documents. It uses Large Language Models (LLMs) and semantic search (RAG) to extract context-aware answers directly from uploaded files.

## Architecture

This project is built using a simple, unified Python stack:
- **Frontend / UI**: [Streamlit](https://streamlit.io/)
- **Document Processing**: `PyPDF2` and `LangChain` text splitters
- **Vector Database**: [FAISS](https://github.com/facebookresearch/faiss) (runs locally, no cloud DB required)
- **AI Models**: OpenAI (GPT-3.5-turbo and text-embedding-ada-002)

## Features
- 📄 Upload any PDF document
- 🧠 Automatic text chunking and vector embedding
- 💬 Chat interface to ask questions about the document
- 🔍 Accurate answers based *only* on the document's content

## Setup Instructions

### 1. Prerequisites
- Python 3.9+
- An OpenAI API Key

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/Rishikesan05/AI-Document-Search.git
cd AI-Document-Search
pip install -r requirements.txt
```

### 3. Environment Variables
Rename `.env.example` to `.env` and add your OpenAI API key:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Run the Application
Start the Streamlit server:
```bash
streamlit run app.py
```
The application will open in your browser at `http://localhost:8501`.
