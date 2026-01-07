"""
rag_utils.py
Core utilities for Retrieval-Augmented Generation (RAG)
"""

import os
from typing import List, TypedDict, Any
import numpy as np
import faiss
import tiktoken

from pypdf import PdfReader
from docx import Document
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Models
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"


# Types 
class DocumentChunk(TypedDict):
    text: str
    source: str
    chunk_id: int
    page: int


# OpenAI 
def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    return OpenAI(api_key=api_key)


# File Readers
def read_txt(file) -> str:
    return file.read().decode("utf-8", errors="ignore")


def read_md(file) -> str:
    return file.read().decode("utf-8", errors="ignore")


def read_docx(file) -> str:
    doc = Document(file)
    return "\n".join(p.text for p in doc.paragraphs)


def read_pdf(file) -> List[tuple]:
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append((text, i + 1))
    return pages


# Chunking 
def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 200,
) -> List[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        start = end - overlap

    return chunks


#Build Document Store
def build_document_store(files: List[Any]) -> List[DocumentChunk]:
    documents: List[DocumentChunk] = []

    for f in files:
        name = f.name.lower()

        if name.endswith(".pdf"):
            pages = read_pdf(f)
            for text, page_num in pages:
                chunks = chunk_text(text)
                for i, chunk in enumerate(chunks):
                    documents.append({
                        "text": chunk,
                        "source": f.name,
                        "chunk_id": i,
                        "page": page_num,
                    })

        else:
            if name.endswith(".txt"):
                text = read_txt(f)
            elif name.endswith(".docx"):
                text = read_docx(f)
            elif name.endswith(".md"):
                text = read_md(f)
            else:
                continue

            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                documents.append({
                    "text": chunk,
                    "source": f.name,
                    "chunk_id": i,
                    "page": 1,
                })

    return documents


# Embeddings & FAISS 
def embed_texts(texts: List[str], client: OpenAI) -> np.ndarray:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    vectors = [d.embedding for d in response.data]
    return np.array(vectors, dtype="float32")


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return vectors / norms


def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


# Index Builder
def build_index_from_files(files: List[Any]):
    client = get_openai_client()
    documents = build_document_store(files)

    texts = [d["text"] for d in documents]
    embeddings = embed_texts(texts, client)
    embeddings = normalize(embeddings)

    index = build_faiss_index(embeddings)
    return documents, index


# Retrieval + Answer
def answer_question_rag(
    question: str,
    index: faiss.IndexFlatIP,
    documents: List[DocumentChunk],
    top_k: int = 5,
) -> str:
    client = get_openai_client()

    q_emb = embed_texts([question], client)
    q_emb = normalize(q_emb)

    scores, indices = index.search(q_emb, top_k)
    indices = indices[0]

    retrieved = [documents[i] for i in indices if i != -1]

    if not retrieved:
        return "I couldn’t find this information in the uploaded document."

    context = "\n\n".join(
        f"[Source: {d['source']} | Page {d['page']}]\n{d['text']}"
        for d in retrieved
    )

    prompt = f"""
<context>
{context}
</context>

Question: {question}

Rules:
- Answer ONLY from the context.
- If the answer is not present, say:
"I couldn’t find this information in the uploaded document."
"""

    response = client.responses.create(
        model=LLM_MODEL,
        instructions="You are a document-based assistant.",
        input=prompt.strip(),
    )

    citations = "\n".join(
        sorted({f"{d['source']} (page {d['page']})" for d in retrieved})
    )

    return response.output_text + f"\n\n**Sources:**\n{citations}"
