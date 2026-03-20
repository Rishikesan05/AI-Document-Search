# AI Document Search (RAG Chatbot)

An intelligent conversational assistant that allows users to chat with PDF documents using Large Language Models and semantic search to extract context-aware answers directly from uploaded files.

## Features

- **PDF Document Ingestion**: Upload and parse PDFs, extract text, and store vector embeddings in a database
- **Semantic Search**: Uses vector similarity search to find the most relevant document sections for any query
- **Streaming Responses**: Real-time streaming of AI-generated responses for a smooth user experience
- **LangGraph Integration**: Built using LangGraph's state machine approach to orchestrate ingestion and retrieval workflows
- **Modern Frontend**: Next.js React UI with file uploads, real-time chat, and Tailwind CSS styling

## Tech Stack

- **Frontend**: React, Next.js, Tailwind CSS
- **Backend**: Node.js, TypeScript, LangGraph
- **AI/ML**: OpenAI GPT, LangChain
- **Vector Database**: Supabase (pgvector)
- **Deployment**: Vercel + Docker

## Architecture

```
┌─────────────────────┐    1. Upload PDFs    ┌───────────────────────────┐
│ Frontend (Next.js)  │ ──────────────────> │ Backend (LangGraph)       │
│ - React UI w/ chat  │                      │ - Ingestion Graph         │
│ - Upload .pdf files │ <────────────────── │   + Vector embedding      │
└─────────────────────┘    2. Confirmation   └───────────────────────────┘

┌─────────────────────┐    3. Ask questions  ┌───────────────────────────┐
│ Frontend (Next.js)  │ ──────────────────> │ Backend (LangGraph)       │
│ - Chat + SSE stream │                      │ - Retrieval Graph         │
│ - Display sources   │ <────────────────── │   + Chat model (OpenAI)   │
└─────────────────────┘ 4. Streamed answers  └───────────────────────────┘
```

## Getting Started

### Prerequisites

- Node.js v18+ (recommended: v20)
- Yarn package manager
- Supabase project (for vector storage)
- OpenAI API Key

### Installation

1. Clone the repository:

```bash
git clone https://github.com/Rishikesan05/AI-Document-Search.git
cd AI-Document-Search
```

2. Install dependencies:

```bash
yarn install
```

3. Configure environment variables:

```bash
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
```

### Environment Variables

**Backend (.env)**
```
OPENAI_API_KEY=your-openai-api-key
SUPABASE_URL=your-supabase-url
SUPABASE_SERVICE_ROLE_KEY=your-supabase-key
```

**Frontend (.env)**
```
NEXT_PUBLIC_LANGGRAPH_API_URL=http://localhost:2024
```

### Running Locally

**Start the backend:**
```bash
cd backend
yarn langgraph:dev
```

**Start the frontend:**
```bash
cd frontend
yarn dev
```

Access the app at `http://localhost:3000`

## Usage

1. Upload PDF documents via the file upload button
2. Wait for ingestion to complete (text extraction + embedding)
3. Ask questions in the chat input
4. Get AI-powered answers with source references

## Project Structure

```
├── backend/
│   ├── src/
│   │   ├── ingestion_graph.ts    # PDF processing pipeline
│   │   ├── retrieval_graph.ts    # Question answering pipeline
│   │   └── shared/
│   │       ├── configuration.ts  # App configuration
│   │       └── retrieval.ts      # Vector store retrieval
│   └── langgraph.json
├── frontend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── chat/             # Chat API route
│   │   │   └── ingest/           # PDF upload API route
│   │   └── page.tsx              # Main chat UI
│   ├── components/               # React components
│   └── styles/                   # Tailwind styles
└── package.json
```

## License

MIT
