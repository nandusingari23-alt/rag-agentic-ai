# Agentic AI RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions using the Agentic AI ebook as its knowledge source.

## Project Overview

This project implements a RAG pipeline that:

1. Loads the Agentic AI ebook PDF.
2. Splits the ebook into smaller chunks.
3. Generates embeddings for the chunks.
4. Stores the embeddings in Pinecone.
5. Retrieves relevant chunks for a user question.
6. Uses LangGraph to orchestrate retrieval and generation.
7. Uses Ollama with Llama 3.2 3B for local answer generation.
8. Returns the final answer, retrieved context, and confidence score.
9. Refuses to answer questions that are not supported by the ebook.

## Technologies Used

- Python
- LangChain
- LangGraph
- Pinecone
- Hugging Face Sentence Transformers
- Ollama
- Llama 3.2 3B
- FastAPI
- Uvicorn
- PyPDF
- Streamlit

## Project Structure

```text
RAG-AGENTIC-AI/
│
├── data/
│   └── Ebook-Agentic-AI.pdf
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── ingestion.py
│   └── graph.py
│
├── app.py
├── tests_sample_queries.py
├── tests_retrieval.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md

RAG Pipeline
Agentic AI Ebook
       ↓
PDF Loading
       ↓
Text Chunking
       ↓
Hugging Face Embeddings
       ↓
Pinecone Vector Database
       ↓
User Question
       ↓
Similarity Search
       ↓
Relevant Context
       ↓
LangGraph
       ↓
Ollama Llama 3.2 3B
       ↓
Final Answer
Setup

Create and activate a virtual environment:

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
Environment Variables

Create a .env file:

OPENAI_API_KEY=your_key
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=agentic-ai-index

Do not commit .env to GitHub.

Data Ingestion

Run:

python -m src.ingestion

The ebook is loaded, split into chunks, embedded, and stored in Pinecone.

Run Ollama

Make sure Ollama is installed and the model is available:

ollama pull llama3.2:3b
Run the API

Start FastAPI:

uvicorn app:app --reload

Open:

http://127.0.0.1:8000/docs

The Swagger UI can be used to test the /chat endpoint.

API Request

Example:

{
  "question": "What is Agentic AI?"
}
API Response

The API returns:

{
  "final_answer": "...",
  "retrieved_context": [
    "..."
  ],
  "confidence_score": 0.0
}
Testing

Run the sample queries:

python tests_sample_queries.py

The test set includes questions about:

Agentic AI
Core characteristics
Building blocks
Autonomous decision making
Practical applications
An out-of-context question about the 2022 FIFA World Cup

For questions outside the ebook's knowledge, the chatbot responds:

I could not find this information in the Agentic AI ebook.
Grounding

The chatbot is instructed to answer only from the retrieved ebook context and not use outside knowledge.

This helps reduce unsupported or hallucinated answers.

##Author
Nandu Singari

