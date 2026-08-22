from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import chat

app = FastAPI(title="Simple Chatbot API", version="1.0.0")

# Allow the frontend (served from a different origin/port) to call this API.
# For production, replace "*" with the exact origin(s) you trust.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api", tags=["chat"])


@app.get("/")
def root():
    return {"message": "Chatbot API is running. See /docs for the API docs."}
