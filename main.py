from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import sqlite3
import time
from pathlib import Path

# Load .env
api_key = "AIzaSyBnlJnbN5kve35EARmEeWX1PfcOBNK3G3o"

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Initialize Gemini
chat = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key
)

app = FastAPI(title="Gemini Chat Bot", description="A chat interface for Google's Gemini AI")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Database Setup ----------------
DB_FILE = "tmp_chats.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_name TEXT,
        timestamp INTEGER
    )""")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER,
        sender TEXT,
        text TEXT,
        timestamp INTEGER,
        FOREIGN KEY(chat_id) REFERENCES chats(id)
    )""")
    conn.commit()
    conn.close()

init_db()

# ---------------- Models ----------------
class PromptRequest(BaseModel):
    chat_id: int
    prompt: str

class NewChatRequest(BaseModel):
    chat_name: str

# ---------------- Static Files and HTML ----------------

# Serve the main HTML page
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the main chat interface"""
    try:
        html_path = Path("index.html")
        if html_path.exists():
            return FileResponse("index.html", media_type="text/html")
        else:
            # Fallback HTML if index.html doesn't exist
            return HTMLResponse("""
            <html>
                <head><title>Gemini Chat - File Not Found</title></head>
                <body>
                    <h1>Error: index.html not found</h1>
                    <p>Please make sure index.html is in the same directory as main.py</p>
                    <p>API endpoints are still available at:</p>
                    <ul>
                        <li><a href="/docs">/docs - API Documentation</a></li>
                        <li><a href="/chats">/chats - Get all chats</a></li>
                    </ul>
                </body>
            </html>
            """)
    except Exception as e:
        return HTMLResponse(f"<html><body><h1>Error loading page</h1><p>{str(e)}</p></body></html>")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Gemini Chat Bot"}

# ---------------- API Routes ----------------

@app.post("/new_chat")
def new_chat(req: NewChatRequest):
    """Create a new chat session"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        ts = int(time.time())
        cursor.execute("INSERT INTO chats (chat_name, timestamp) VALUES (?, ?)", (req.chat_name, ts))
        conn.commit()
        chat_id = cursor.lastrowid
        conn.close()
        return {"chat_id": chat_id, "chat_name": req.chat_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create chat: {str(e)}")

@app.get("/chats")
def get_chats():
    """Get all chat sessions"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT id, chat_name FROM chats ORDER BY timestamp DESC")
        chats = cursor.fetchall()
        conn.close()
        return [{"chat_id": c[0], "chat_name": c[1]} for c in chats]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chats: {str(e)}")

@app.get("/messages/{chat_id}")
def get_messages(chat_id: int):
    """Get all messages for a specific chat"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        # Check if chat exists
        cursor.execute("SELECT id FROM chats WHERE id=?", (chat_id,))
        if not cursor.fetchone():
            conn.close()
            raise HTTPException(status_code=404, detail="Chat not found")
        
        cursor.execute("SELECT sender, text FROM messages WHERE chat_id=? ORDER BY timestamp ASC", (chat_id,))
        msgs = cursor.fetchall()
        conn.close()
        return [{"sender": m[0], "text": m[1]} for m in msgs]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve messages: {str(e)}")

@app.post("/ask")
def ask_gemini(req: PromptRequest):
    """Send a message to Rahul's chat and get response"""
    try:
        # Validate chat exists
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM chats WHERE id=?", (req.chat_id,))
        if not cursor.fetchone():
            conn.close()
            raise HTTPException(status_code=404, detail="Chat not found")

        # Save user message
        ts = int(time.time())
        cursor.execute("INSERT INTO messages (chat_id, sender, text, timestamp) VALUES (?, ?, ?, ?)",
                       (req.chat_id, "user", req.prompt, ts))

        # Get Gemini response
        try:
            response = chat.invoke(req.prompt)
            answer = response.content
        except Exception as e:
            answer = f"Sorry, I encountered an error: {str(e)}"

        # Save bot response
        cursor.execute("INSERT INTO messages (chat_id, sender, text, timestamp) VALUES (?, ?, ?, ?)",
                       (req.chat_id, "bot", answer, int(time.time())))

        conn.commit()
        conn.close()

        return {"answer": answer}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process request: {str(e)}")

@app.delete("/delete_chat/{chat_id}")
def delete_chat(chat_id: int):
    """Delete a chat and all its messages"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Check if chat exists
        cursor.execute("SELECT id FROM chats WHERE id=?", (chat_id,))
        if not cursor.fetchone():
            conn.close()
            raise HTTPException(status_code=404, detail="Chat not found")
        
        # Delete messages first (foreign key constraint)
        cursor.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
        # Delete chat
        cursor.execute("DELETE FROM chats WHERE id=?", (chat_id,))
        
        conn.commit()
        conn.close()
        return {"message": "Chat deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete chat: {str(e)}")

@app.post("/clear_chats")
def clear_all_chats():
    """Clear all chats and messages"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages")
        cursor.execute("DELETE FROM chats")
        conn.commit()
        conn.close()
        return {"message": "All chats cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear chats: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Gemini Chat Bot...")
    print("📝 Open your browser and go to: http://localhost:8000")
    print("📚 API Documentation available at: http://localhost:8000/docs")

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

