import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from config import config
import shutil

# Mapping extensions to loaders
LOADERS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
}

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

async def process_document(file_path: str, tenant_id: str, doc_id: int):
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

    vector_store = get_vector_store(tenant_id)
    vector_store.add_documents(chunks)
    
    return len(chunks)

async def delete_document_from_vector_store(tenant_id: str, doc_id: int):
    vector_store = get_vector_store(tenant_id)
    # Chroma allows filtering by metadata on deletion in some versions, 
    # but the simplest reliable way is to use get() then delete() if needed.
    # Here we use the generic delete by metadata if possible.
    vector_store.delete(where={"doc_id": doc_id})
