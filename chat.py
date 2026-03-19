import json
from sse_starlette.sse import EventSourceResponse
from retrieval import query_rag
from sqlalchemy.orm import Session
import models
import asyncio

async def chat_streaming_response(tenant_id: str, user_id: int, question: str, db: Session):
    chain, citations = await query_rag(tenant_id, question)
    
    # Save user message
    user_msg = models.ChatMessage(
        tenant_id=tenant_id,
        user_id=user_id,
        role="user",
        content=question
    )
    db.add(user_msg)
    db.commit()

    async def event_generator():
        full_response = ""
        # Send citations first
        yield {"data": json.dumps({"type": "citations", "content": citations})}
        
        # Stream the answer
        try:
            async for chunk in chain.astream(question):
                full_response += chunk
                yield {"data": json.dumps({"type": "chunk", "content": chunk})}
        except Exception as e:
            yield {"data": json.dumps({"type": "error", "content": str(e)})}
            return

        # Save assistant message
        assistant_msg = models.ChatMessage(
            tenant_id=tenant_id,
            user_id=user_id,
            role="assistant",
            content=full_response
        )
        db.add(assistant_msg)
        db.commit()
        
        yield {"data": "[DONE]"}

    return EventSourceResponse(event_generator())
