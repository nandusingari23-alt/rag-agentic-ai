import os
from typing import TypedDict

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langgraph.graph import StateGraph, START, END
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore

from src.config import PINECONE_INDEX_NAME


load_dotenv()


class AgentState(TypedDict):
    question: str
    context: list
    answer: str
    score: float


class HFEmbeddings(Embeddings):

    def __init__(self):
        self.client = InferenceClient(
            api_key=os.getenv("HF_TOKEN")
        )
        self.model = "sentence-transformers/all-MiniLM-L6-v2"

    def embed_query(self, text: str):
        result = self.client.feature_extraction(
            text,
            model=self.model
        )

        # Convert returned embedding to a simple list
        if hasattr(result, "tolist"):
            result = result.tolist()

        while result and isinstance(result[0], list):
            result = result[0]

        return result

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


embeddings = HFEmbeddings()


vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings
)


hf_client = InferenceClient(
    api_key=os.getenv("HF_TOKEN"),
    provider="auto"
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

    confidence = sum(scores) / len(scores) if scores else 0.0

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

    response = hf_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=500
    )

    return {
        "answer": response.choices[0].message.content
    }


workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

graph = workflow.compile()