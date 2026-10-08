"""Step 2: ask questions. Retrieves the most relevant chunks and has the LLM answer from them."""
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

DB_DIR = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # must match ingest.py
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
TOP_K = 4

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Answer the question using ONLY the context below. "
     "If the answer is not in the context, say: \"I could not find this in the documents.\"\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])

_embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
_db = Chroma(persist_directory=DB_DIR, embedding_function=_embeddings)
_retriever = _db.as_retriever(search_kwargs={"k": TOP_K})
_llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)


def ask(question: str) -> dict:
    """Returns {"answer": str, "sources": [str, ...]}. Reused later by the FastAPI step."""
    docs = _retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    answer = (PROMPT | _llm).invoke({"context": context, "question": question}).content
    sources = sorted({f"{d.metadata.get('source', '?')} (page {d.metadata.get('page', '?')})" for d in docs})
    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    print("Ask questions about your documents. Type 'exit' to quit.")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        result = ask(q)
        print("\nBot:", result["answer"])
        print("Sources:", "; ".join(result["sources"]))
