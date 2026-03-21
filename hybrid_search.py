import os
import pickle
import hashlib
import asyncio
import logging
from typing import List, Tuple, Dict, Any
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
# from ingestion import get_vector_store
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
BM25_INDEX_DIR = "./bm25_indexes"
RRF_K = 60
TOP_K_RETRIEVAL = 20
TOP_K_FINAL = 6

# Create BM25 index directory if not exists
os.makedirs(BM25_INDEX_DIR, exist_ok=True)

# Load CrossEncoder model once at module level
logger.info("Loading CrossEncoder model...")
try:
    reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
except Exception as e:
    logger.error(f"Failed to load CrossEncoder model: {e}")
    reranker_model = None

def get_bm25_path(tenant_id: str) -> str:
    return os.path.join(BM25_INDEX_DIR, f"{tenant_id}.pkl")

def get_content_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def build_bm25_index(tenant_id: str):
    """Builds or rebuilds BM25 index for a tenant from all chunks in ChromaDB."""
    from ingestion import get_vector_store
    try:
        vector_store = get_vector_store(tenant_id)
        # Get all documents from ChromaDB for this tenant
        result = vector_store.get()
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])
        
        if not documents:
            logger.warning(f"No documents found in ChromaDB for tenant {tenant_id}. BM25 index not built.")
            return

        tokenized_corpus = [doc.split() for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)
        
        data = {
            "bm25": bm25,
            "documents": documents,
            "metadatas": metadatas
        }
        
        with open(get_bm25_path(tenant_id), "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Successfully built BM25 index for tenant {tenant_id}")
    except Exception as e:
        logger.error(f"Error building BM25 index for tenant {tenant_id}: {e}")

def update_bm25_index(tenant_id: str, new_chunks: List[Any]):
    """Appends new chunks to existing index and re-saves."""
    try:
        bm25_path = get_bm25_path(tenant_id)
        new_texts = [chunk.page_content for chunk in new_chunks]
        new_metadatas = [chunk.metadata for chunk in new_chunks]
        
        if os.path.exists(bm25_path):
            with open(bm25_path, "rb") as f:
                data = pickle.load(f)
            
            existing_docs = data.get("documents", [])
            existing_metadatas = data.get("metadatas", [])
            
            all_docs = existing_docs + new_texts
            all_metadatas = existing_metadatas + new_metadatas
            
            tokenized_corpus = [doc.split() for doc in all_docs]
            bm25 = BM25Okapi(tokenized_corpus)
            
            updated_data = {
                "bm25": bm25,
                "documents": all_docs,
                "metadatas": all_metadatas
            }
        else:
            # If index doesn't exist, build a new one
            tokenized_corpus = [doc.split() for doc in new_texts]
            bm25 = BM25Okapi(tokenized_corpus)
            updated_data = {
                "bm25": bm25,
                "documents": new_texts,
                "metadatas": new_metadatas
            }
            
        with open(bm25_path, "wb") as f:
            pickle.dump(updated_data, f)
        logger.info(f"Successfully updated BM25 index for tenant {tenant_id}")
    except Exception as e:
        logger.error(f"Error updating BM25 index for tenant {tenant_id}: {e}")

async def dense_retrieval(tenant_id: str, query: str) -> List[Tuple[str, Dict, float]]:
    """Step 1: Dense Retrieval (ChromaDB)"""
    from ingestion import get_vector_store
    try:
        vector_store = get_vector_store(tenant_id)
        # Chroma's similarity_search_with_relevance_scores or similarity_search_with_score
        # For cosine similarity, Chroma's score is often distance (lower is better) or similarity.
        # We'll use similarity_search_with_score and normalize if needed, but RRF only needs rank.
        loop = asyncio.get_event_loop()
        docs_with_scores = await loop.run_in_executor(
            None, 
            lambda: vector_store.similarity_search_with_score(query, k=TOP_K_RETRIEVAL)
        )
        
        results = []
        for doc, score in docs_with_scores:
            results.append((doc.page_content, doc.metadata, score))
        return results
    except Exception as e:
        logger.error(f"Dense retrieval error: {e}")
        return []

async def sparse_retrieval(tenant_id: str, query: str) -> List[Tuple[str, Dict, float]]:
    """Step 2: Sparse Retrieval (BM25)"""
    try:
        bm25_path = get_bm25_path(tenant_id)
        if not os.path.exists(bm25_path):
            logger.info(f"BM25 index for tenant {tenant_id} missing, falling back to dense-only.")
            return []
            
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, lambda: pickle.load(open(bm25_path, "rb")))
        
        bm25 = data["bm25"]
        documents = data["documents"]
        metadatas = data["metadatas"]
        
        tokenized_query = query.split()
        scores = bm25.get_scores(tokenized_query)
        
        # Get top K indices
        import numpy as np
        top_n_indices = np.argsort(scores)[::-1][:TOP_K_RETRIEVAL]
        
        results = []
        for idx in top_n_indices:
            if scores[idx] > 0:
                results.append((documents[idx], metadatas[idx], float(scores[idx])))
        return results
    except Exception as e:
        logger.error(f"Sparse retrieval error: {e}")
        return []

def rrf_merge(dense_results: List[Tuple], sparse_results: List[Tuple]) -> List[Dict]:
    """Step 3: Reciprocal Rank Fusion (RRF)"""
    scores = {} # text_hash -> score
    doc_map = {} # text_hash -> (text, metadata)
    
    # Process Dense
    for rank, (text, metadata, _) in enumerate(dense_results):
        txt_hash = get_content_hash(text)
        if txt_hash not in scores:
            scores[txt_hash] = 0
            doc_map[txt_hash] = (text, metadata)
        scores[txt_hash] += 1.0 / (RRF_K + rank + 1)
        
    # Process Sparse
    for rank, (text, metadata, _) in enumerate(sparse_results):
        txt_hash = get_content_hash(text)
        if txt_hash not in scores:
            scores[txt_hash] = 0
            doc_map[txt_hash] = (text, metadata)
        scores[txt_hash] += 1.0 / (RRF_K + rank + 1)
        
    # Sort by RRF score
    sorted_hashes = sorted(scores.keys(), key=lambda h: scores[h], reverse=True)
    
    ranked_docs = []
    for h in sorted_hashes[:TOP_K_RETRIEVAL]:
        text, metadata = doc_map[h]
        ranked_docs.append({
            "content": text,
            "metadata": metadata,
            "rrf_score": scores[h]
        })
    return ranked_docs

async def cross_encoder_rerank(query: str, ranked_docs: List[Dict]) -> List[Dict]:
    """Step 4: CrossEncoder Reranking"""
    if not reranker_model or not ranked_docs:
        return ranked_docs[:TOP_K_FINAL]
        
    try:
        pairs = [[query, doc["content"]] for doc in ranked_docs]
        
        loop = asyncio.get_event_loop()
        ce_scores = await loop.run_in_executor(None, lambda: reranker_model.predict(pairs))
        
        for doc, score in zip(ranked_docs, ce_scores):
            doc["relevance_score"] = float(score)
            
        # Re-sort by cross-encoder score
        ranked_docs.sort(key=lambda x: x["relevance_score"], reverse=True)
        return ranked_docs[:TOP_K_FINAL]
    except Exception as e:
        logger.error(f"CrossEncoder reranking error: {e}")
        return ranked_docs[:TOP_K_FINAL]

async def hybrid_search_pipeline(tenant_id: str, query: str) -> List[Dict]:
    """Full 4-step hybrid search pipeline."""
    # Step 1 & 2: Retrieval (Concurrent)
    dense_task = asyncio.create_task(dense_retrieval(tenant_id, query))
    sparse_task = asyncio.create_task(sparse_retrieval(tenant_id, query))
    
    dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)
    
    # Step 3: RRF
    rrf_docs = rrf_merge(dense_results, sparse_results)
    
    # Step 4: Rerank
    final_docs = await cross_encoder_rerank(query, rrf_docs)
    
    return final_docs
