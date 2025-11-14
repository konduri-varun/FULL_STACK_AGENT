from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import os
import sys
import importlib.util
from dotenv import load_dotenv
import uvicorn
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# Load environment variables
load_dotenv()

# Get port from environment (for Render deployment)
PORT = int(os.getenv("PORT", 7860))

# MongoDB setup
MONGODB_URI = os.environ.get("MONGODB_URI")
mongo_client = None
db = None
chat_history_collection = None

try:
    mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    mongo_client.admin.command('ping')
    db = mongo_client['fullstack_agent']
    chat_history_collection = db['chat_history']
    print("✅ Connected to MongoDB successfully!")
except ConnectionFailure as e:
    print(f"⚠️  MongoDB connection failed: {e}")
    print("📝 Chat history will not be persisted to database")

# Import the agent
agent_path = os.path.join(os.path.dirname(__file__), "my_first_agent", "agent.py")
spec = importlib.util.spec_from_file_location("agent_module", agent_path)
agent_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_module)

from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import uuid

# Initialize agent
root_agent = agent_module.root_agent
session_service = InMemorySessionService()
runner = Runner(agent=root_agent, app_name="fullstack_python_agent", session_service=session_service)

# Track sessions
sessions = {}

# Create FastAPI app
app = FastAPI(title="CodeFlow AI - Professional Development Assistant")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://codeflow-ai-frontend.vercel.app",
        "https://*.vercel.app"  # Allow all Vercel preview deployments
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    userId: Optional[str] = "default_user"
    history: Optional[List[Message]] = []

class ChatResponse(BaseModel):
    response: str
    userId: str
    sessionId: str

@app.get("/")
async def root():
    return {
        "message": "Full-Stack Python AI Agent API",
        "version": "1.0.0",
        "agents": 12,
        "status": "online",
        "mongodb_connected": mongo_client is not None
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "api_key_set": bool(os.environ.get("GOOGLE_API_KEY")),
        "mongodb_connected": mongo_client is not None
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Check API key
        if not os.environ.get("GOOGLE_API_KEY"):
            raise HTTPException(
                status_code=500,
                detail="GOOGLE_API_KEY not set. Please configure your API key."
            )
        
        # Get or create session for user
        user_id = request.userId
        if user_id not in sessions:
            session_id = f"session_{uuid.uuid4().hex[:8]}"
            session_service.create_session_sync(
                user_id=user_id,
                session_id=session_id,
                app_name="fullstack_python_agent"
            )
            sessions[user_id] = session_id
        else:
            session_id = sessions[user_id]
        
        # Create content object
        content = types.Content(parts=[types.Part(text=request.message)])
        
        # Run agent and collect response
        response_text = ""
        for event in runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=content
        ):
            if hasattr(event, 'content') and event.content:
                if hasattr(event.content, 'parts'):
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_text += part.text
        
        if not response_text:
            response_text = "No response generated. Please try rephrasing your question."
        
        # Save to MongoDB if connected
        if chat_history_collection is not None:
            try:
                chat_history_collection.update_one(
                    {"userId": user_id},
                    {
                        "$push": {
                            "messages": {
                                "$each": [
                                    {"role": "user", "content": request.message, "timestamp": datetime.utcnow()},
                                    {"role": "assistant", "content": response_text, "timestamp": datetime.utcnow()}
                                ]
                            }
                        },
                        "$set": {"lastUpdated": datetime.utcnow()}
                    },
                    upsert=True
                )
            except Exception as db_error:
                print(f"⚠️  Error saving to MongoDB: {db_error}")
        
        return ChatResponse(
            response=response_text,
            userId=user_id,
            sessionId=session_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"🚀 Starting FastAPI backend on port {port}...")
    print(f"📡 CORS enabled for React frontend (localhost:3000)")
    print(f"🔑 GOOGLE_API_KEY is set: {bool(os.environ.get('GOOGLE_API_KEY'))}")
    print(f"💾 MongoDB connected: {mongo_client is not None}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
