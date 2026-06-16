import glob
import os
from importlib import metadata
from pathlib import Path
from typing import Iterable, List

import pypdf
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core import embeddings
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

model = "nomic-embed-text"
load_dotenv(override=True)
KNOWLEDGE_BASE = str(Path(__file__).parent / "knowledge-base")
DATABASE_URL = os.getenv("DATABASE_URL")


embed = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")


# fetch documents
def fetch_documents() -> list[Document]:
    dirs = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
    documents = []

    for dir in dirs:
        doc_type = os.path.basename(dir)
        loader = DirectoryLoader(
            dir,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        dir_docs = loader.load()
        for doc in dir_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    return documents


# Create chunk
def create_chunks(docs: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    return chunks


# Create Embeddings
def create_embeddings(chunks):
    vs = PGVector(
        embeddings=embed,
        collection_name="kb",
        connection=DATABASE_URL,
    )

    # safety batch insert
    batch_size = 20
    for i in range(0, len(chunks), batch_size):
        vs.add_documents(chunks[i : i + batch_size])

    dim = len(embed.embed_query("test"))
    print(f"Embedding dimension: {dim}")


if __name__ == "__main__":
    # fetch documents
    docs = fetch_documents()
    # Chunk documents
    chunks = create_chunks(docs)
    # create embedding
    create_embeddings(chunks)
    print("Ingestion completed")
