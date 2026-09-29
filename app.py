from fastapi import FastAPI
from pydantic import BaseModel

from src.graph import graph


app = FastAPI(
    title="Agentic AI RAG Chatbot",
    description="RAG chatbot using the Agentic AI ebook",
    version="1.0.0"
)


class ChatRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "Agentic AI RAG Chatbot is running"
    }


@app.post("/chat")
def chat(request: ChatRequest):

    result = graph.invoke({
        "question": request.question,
        "context": [],
        "answer": "",
        "score": 0.0
    })

    return {
        "final_answer": result["answer"],
        "retrieved_context": result["context"],
        "confidence_score": result["score"]
    }