from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from ingestion import get_vector_store
from config import config

SYSTEM_PROMPT = """You are a helpful assistant for a company.
Answer ONLY using the provided context below.
If the answer is not found in the context, respond with:
'I don't have enough information in your uploaded documents to answer this.' 
Never make anything up.

Context:
{context}

Question: {question}
"""

def get_rag_chain(tenant_id: str):
    llm = OllamaLLM(
        model=config.OLLAMA_LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL
    )
    
    vector_store = get_vector_store(tenant_id)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
    
    def format_docs(docs):
        return "\n\n".join([f"Source: {d.metadata.get('filename')}, Page: {d.metadata.get('page','N/A')}\nContent: {d.page_content}" for d in docs])
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain, retriever

async def query_rag(tenant_id: str, question: str):
    chain, retriever = get_rag_chain(tenant_id)
    
    # Get relevant documents first to show citations
    docs = retriever.invoke(question)
    citations = [{"filename": d.metadata.get("filename"), "page": d.metadata.get("page", "N/A")} for d in docs]
    
    # We use stream for the actual response in chat.py, 
    # but here is a simple invoke if needed.
    # response = chain.invoke(question)
    return chain, citations
