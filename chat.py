import json
from sse_starlette.sse import EventSourceResponse
from retrieval import query_rag
from sqlalchemy.orm import Session
import models
import asyncio

async def chat_streaming_response(tenant_id: str, user_id: int, question: str, session_id: int, db: Session):
    chain, citations = await query_rag(tenant_id, question)
    
    # Create session if not provided
    is_new_session = False
    if not session_id:
        title = " ".join(question.split()[:5]) + "..." if len(question.split()) > 5 else question
        new_session = models.ChatSession(tenant_id=tenant_id, user_id=user_id, title=title)
        db.add(new_session)
        db.flush() # Get the new ID before commit
        session_id = new_session.id
        is_new_session = True
    
    # Save user message
    user_msg = models.ChatMessage(
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
        role="user",
        content=question
    )
    db.add(user_msg)
    db.commit()

    async def event_generator():
        full_response = ""
        # Send session ID if newly created
        if is_new_session:
            yield {"data": json.dumps({"type": "session_id", "content": session_id})}
            
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
            session_id=session_id,
            role="assistant",
            content=full_response
        )
        db.add(assistant_msg)
        db.commit()
        
        yield {"data": "[DONE]"}

    return EventSourceResponse(event_generator())
