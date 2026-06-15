import os
from importlib import metadata
from typing import List

import pypdf
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

embed = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

# Load pdf


def load_pdf_pages(file_path: str) -> List[Document]:
    reader = pypdf.PdfReader(file_path)

    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": file_path, "page": i},
        )
        for i, page in enumerate(reader.pages)
    ]


# Load file
file_path = "./docs/caf.pdf"
docs = load_pdf_pages(file_path)

# Initialize splitters
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(f"Len: {len(all_splits)}")

vs = PGVector(embeddings=embed, collection_name="kb", connection=DATABASE_URL)
vs.add_documents(documents=all_splits)

results = vs.similarity_search("Why was the virtual card moved to phase II?")
print(results[0])
