from typing import Dict
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from .models import ChatInput, ChatResponse, Conversation, SetHistoryInput, Message
from .irt_app import process_chat_message_stream, process_chat_message

import logging
from fastapi.responses import StreamingResponse
import json
from .logging_config import setup_logging

# Initialize logging
setup_logging()

logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for conversations
conversations: Dict[str, Conversation] = {}


@app.get("/")
async def root():
    """Root endpoint - redirects to health check."""
    return {
        "message": "IRT Chatbot API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "chat_stream": "/chat/stream",
            "chat": "/chat",
            "conversation": "/conversation/{session_id}"
        }
    }


@app.get("/health")
async def health_check():
    """Lightweight health endpoint for Cloud Run."""
    return {"status": "ok"}


@app.post("/chat/stream")
async def stream_chat_endpoint(request: Request):
    try:
        body = await request.json()
        chat_input = ChatInput.from_dict(body)
        
        logger.info(f"Processing streaming chat request for session: {chat_input.session_id[:8]}...")
        
        # Get or create conversation
        conversation = conversations.get(chat_input.session_id)
        if not conversation:
            conversation = Conversation(
                session_id=chat_input.session_id,
                user_id=chat_input.user_id
            )
            conversations[chat_input.session_id] = conversation
        
        return StreamingResponse(
            process_chat_message_stream(chat_input, conversation),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable buffering in Nginx
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        body = await request.json()
        chat_input = ChatInput.from_dict(body)
        
        # Get or create conversation
        conversation = conversations.get(chat_input.session_id)
        if not conversation:
            conversation = Conversation(
                session_id=chat_input.session_id,
                user_id=chat_input.user_id
            )
            conversations[chat_input.session_id] = conversation
        
        response = await process_chat_message(chat_input, conversation)
        return response.to_dict()
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversation/{session_id}")
async def get_conversation_endpoint(session_id: str):
    """Retrieve the conversation history for a session"""
    try:
        # Get or create conversation
        conversation = conversations.get(session_id)
        if not conversation:
            conversation = Conversation(session_id=session_id)
            conversations[session_id] = conversation
        
        return {
            "session_id": session_id,
            "messages": [msg.dict() for msg in conversation.messages],
            "stages": conversation.stages,
            "language": conversation.language
        }
    except Exception as e:
        logger.error(f"Error retrieving conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/conversation/{session_id}")
async def delete_conversation_endpoint(session_id: str):
    """Delete a conversation session"""
    if session_id in conversations:
        del conversations[session_id]
        return {"status": "success", "message": f"Session {session_id} deleted"}
    else:
        return {"status": "not_found", "message": f"Session {session_id} not found"}


@app.post("/chat/set_history")
async def set_history_endpoint(history_input: SetHistoryInput):
    """Sets the history for a given session ID."""
    try:
        session_id = history_input.session_id
        user_id = history_input.user_id
        messages = history_input.messages
        stages = history_input.stages
        language = history_input.language
        
        logger.info(f"Setting history for session: {session_id[:8]}... ({len(messages)} messages)")
        
        # Get or create conversation
        conversation = conversations.get(session_id)
        if not conversation:
            conversation = Conversation(
                session_id=session_id,
                user_id=user_id # Use provided user_id if available
            )
            conversations[session_id] = conversation
            logger.info(f"Created new conversation for session {session_id[:8]} during set_history.")
        else:
            # If conversation exists, update user_id if provided and different
            if user_id and conversation.user_id != user_id:
                logger.warning(f"Updating user_id for existing session {session_id[:8]} from {conversation.user_id} to {user_id}")
                conversation.user_id = user_id

        # Validate and set messages (ensure they are Message objects)
        validated_messages = []
        for msg_data in messages:
            if isinstance(msg_data, Message):
                validated_messages.append(msg_data)
            elif isinstance(msg_data, dict): # Handle case where client sends dicts
                try:
                    validated_messages.append(Message(**msg_data))
                except Exception as e:
                    logger.error(f"Error validating message item during set_history: {msg_data}, Error: {e}")
                    raise HTTPException(status_code=400, detail=f"Invalid message format in history: {msg_data}")
            else:
                 raise HTTPException(status_code=400, detail=f"Invalid message type in history: {type(msg_data)}")

        conversation.messages = validated_messages
        
        # Set stages if provided, otherwise keep existing or default
        if stages is not None:
            conversation.stages = stages
        
        # Set language if provided, otherwise keep existing or default
        if language is not None:
            conversation.language = language
        
        return {
            "status": "success", 
            "message": f"History set for session {session_id[:8]}", 
            "session_id": session_id, 
            "messages_count": len(conversation.messages)
        }
        
    except HTTPException as http_exc:
        logger.error(f"HTTP Error in set_history: {http_exc.detail}")
        raise http_exc # Re-raise FastAPI exceptions
    except Exception as e:
        logger.error(f"Error setting history for session {history_input.session_id[:8]}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error setting history: {str(e)}")

