import os
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from config import config
import shutil
import hashlib
# from hybrid_search import update_bm25_index, build_bm25_index

# Mapping extensions to loaders
LOADERS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
}

def compute_checksum(file_path: str) -> str:
    """Compute MD5 hash of file contents."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_vector_store(tenant_id: str):
    embeddings = OllamaEmbeddings(
        model=config.OLLAMA_EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL
    )
    return Chroma(
        persist_directory=os.path.join(config.CHROMA_DB_PATH, tenant_id),
        embedding_function=embeddings,
        collection_name=f"tenant_{tenant_id}"
    )

async def process_document(file_path: str, tenant_id: str, doc_id: int, version: int = 1):
    _, ext = os.path.splitext(file_path)
    loader_cls = LOADERS.get(ext.lower())
    if not loader_cls:
        raise ValueError(f"Unsupported file extension: {ext}")

    loader = loader_cls(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(documents)

    # Add metadata
    for chunk in chunks:
        chunk.metadata["tenant_id"] = tenant_id
        chunk.metadata["doc_id"] = doc_id
        chunk.metadata["filename"] = os.path.basename(file_path)
        chunk.metadata["version"] = version
        chunk.metadata["active"] = True

    vector_store = get_vector_store(tenant_id)
    vector_store.add_documents(chunks)
    
    # Update BM25 index
    from hybrid_search import update_bm25_index
    update_bm25_index(tenant_id, chunks)
    
    return len(chunks)

async def delete_document_from_vector_store(tenant_id: str, doc_id: int):
    vector_store = get_vector_store(tenant_id)
    # Chroma allows filtering by metadata on deletion in some versions, 
    # but the simplest reliable way is to use get() then delete() if needed.
    # Here we use the generic delete by metadata if possible.
    vector_store.delete(where={"doc_id": doc_id})
    
    # Rebuild BM25 index after deletion to stay in sync
    from hybrid_search import build_bm25_index
    build_bm25_index(tenant_id)

def get_all_chunks_for_doc(tenant_id: str, doc_id: int) -> List[Dict]:
    """Retrieves all active chunks for a document from ChromaDB."""
    vector_store = get_vector_store(tenant_id)
    # Chroma returns a dict with 'documents' and 'metadatas'
    results = vector_store.get(where={"doc_id": doc_id, "active": True})
    
    chunks = []
    if results["documents"]:
        for text, meta in zip(results["documents"], results["metadatas"]):
            chunks.append({"content": text, "metadata": meta})
    return chunks
