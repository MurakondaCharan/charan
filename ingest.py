"""Step 1: read PDFs from data/, split into chunks, embed them, store in ChromaDB."""
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path("data")
DB_DIR = "chroma_db"
CHUNK_SIZE = 1000      # experiment with this later (500, 1000, 1500) and note the results
CHUNK_OVERLAP = 200
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # free, runs on your laptop


def main():
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("Put at least one PDF inside the data/ folder first.")

    pages = []
    for pdf in pdfs:
        pages.extend(PyPDFLoader(str(pdf)).load())
    print(f"Loaded {len(pages)} pages from {len(pdfs)} PDF(s)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks")

    shutil.rmtree(DB_DIR, ignore_errors=True)  # rebuild from scratch each run
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)
    print(f"Stored embeddings in {DB_DIR}/")


if __name__ == "__main__":
    main()
