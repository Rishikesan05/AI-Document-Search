# 📄 AI Document Search (RAG Chatbot)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36+-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.2.6-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

An intelligent, conversational AI assistant that allows you to chat directly with your PDF documents. Built with **Streamlit**, **LangChain**, and **Google Gemini**, this tool uses **Retrieval-Augmented Generation (RAG)** to provide accurate, context-aware answers based strictly on the uploaded document's contents.

---

## ✨ Features

- **Document Upload**: Seamlessly upload any PDF file.
- **Automated Processing**: Instantly extracts, chunks, and creates vector embeddings of the text.
- **Local Vector Database**: Utilizes FAISS for local, fast semantic search without needing a cloud database.
- **Conversational Interface**: Ask questions naturally and get precise answers derived *only* from the document.
- **Modern & Lightweight**: Uses purely Python and Streamlit—no complex frontend frameworks required.

---

## 🛠️ Tech Stack

- **Frontend / UI**: [Streamlit](https://streamlit.io/)
- **Document Processing**: `PyPDF2` & `LangChain` Text Splitters
- **Vector Store**: [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
- **AI Engine**: Google Gemini API (`gemini-3.5-flash` for chat, `gemini-embedding-2` for embeddings)

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.9+** (We recommend using `uv` or `venv` for environment management)
- A **Google Gemini API Key** (Get yours free from [Google AI Studio](https://aistudio.google.com/))

### 2. Installation

Clone the repository and navigate into the project directory:

```bash
git clone https://github.com/Rishikesan05/AI-Document-Search.git
cd AI-Document-Search
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```
*(Tip: If you use `uv`, you can run `uv pip install -r requirements.txt`)*

### 3. Configuration

1. Create a `.env` file in the root directory (you can copy `.env.example`).
2. Add your Google API key:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

### 4. Running the App

Start the Streamlit development server:

```bash
streamlit run app.py
```
*(If using `uv`, run `uv run streamlit run app.py`)*

The app will launch automatically in your browser at `http://localhost:8501`.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Rishikesan05/AI-Document-Search/issues).

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).
