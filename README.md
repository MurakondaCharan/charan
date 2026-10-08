# Document Q&A Chatbot (RAG)

Answers questions from your own PDFs using retrieval-augmented generation.

**Stack:** Python, LangChain, ChromaDB, sentence-transformers embeddings, Google Gemini API

## Setup
```
python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env         # then paste your API key into .env
```

## Run
1. Put PDF files in `data/`
2. `python ingest.py`   (builds the vector database)
3. `python chat.py`     (ask questions)

## How it works
PDF -> split into chunks -> embeddings -> ChromaDB -> retrieve top matches for a question -> LLM answers using only those chunks.
