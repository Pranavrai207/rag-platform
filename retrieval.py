from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from config import config
from hybrid_search import hybrid_search_pipeline

SYSTEM_PROMPT = """You are a helpful assistant for a company.
Answer ONLY using the provided context below.
If the answer is not found in the context, respond with:
'I don't have enough information in your uploaded documents to answer this.' 
Never make anything up.

Context:
{context}

Question: {question}
"""



async def query_rag(tenant_id: str, question: str):
    # 1. Call hybrid search pipeline
    top_chunks = await hybrid_search_pipeline(tenant_id, question)
    
    # 2. Format chunks as context
    context_text = "\n\n".join([
        f"Source: {c['metadata'].get('filename')}, Page: {c['metadata'].get('page','N/A')}\nContent: {c['content']}" 
        for c in top_chunks
    ])
    
    # 3. Build the prompt with context
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT).partial(context=context_text)
    
    # 4. Define LLM and Chain
    llm = OllamaLLM(
        model=config.OLLAMA_LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL
    )
    
    chain = prompt | llm | StrOutputParser()
    
    # 5. Return citations
    citations = [
        {
            "filename": c["metadata"].get("filename"), 
            "page": c["metadata"].get("page", "N/A"),
            "relevance_score": round(c.get("relevance_score", c.get("rrf_score", 0)), 4)
        } 
        for c in top_chunks
    ]
    
    return chain, citations
