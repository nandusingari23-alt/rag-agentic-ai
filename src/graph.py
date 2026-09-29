from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama

from src.config import PINECONE_INDEX_NAME


class AgentState(TypedDict):
    question: str
    context: list
    answer: str
    score: float


# Local embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Pinecone vector store
vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings
)


# Local Ollama LLM
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


def retrieve(state: AgentState):
    question = state["question"]

    results = vectorstore.similarity_search_with_score(
        question,
        k=5
    )

    context = [
        document.page_content
        for document, score in results
    ]

    scores = [
        score
        for document, score in results
    ]

    if scores:
        confidence = sum(scores) / len(scores)
    else:
        confidence = 0.0

    return {
        "context": context,
        "score": confidence
    }


def generate(state: AgentState):
    context = "\n\n".join(state["context"])
    question = state["question"]

    prompt = f"""
You are a RAG chatbot for the Agentic AI ebook.

Answer the user's question ONLY using the provided context from the ebook.

Do not use outside knowledge.

If the context does not contain enough information to answer the question,
respond exactly:

"I could not find this information in the Agentic AI ebook."

Be concise and factual.

Context from the ebook:
{context}

User question:
{question}
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

graph = workflow.compile()