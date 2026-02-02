from fastapi import FastAPI
from app.api.chat import router as chat_router
from app.core.rag import load_rag_resources

app = FastAPI(
    title="Agri AI Expert Chatbot",
    version="1.0.0"
)

@app.on_event("startup")
def startup_event():
    load_rag_resources()

app.include_router(chat_router, prefix="/api/v1")

@app.get("/")
def root():
    return {"status": "Agri AI Chatbot is running"}
