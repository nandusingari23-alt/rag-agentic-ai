from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from src.config import PINECONE_INDEX_NAME


PDF_PATH = "data/Ebook-Agentic-AI.pdf"

def load_pdf():
    loader = PyMuPDFLoader(PDF_PATH)
    documents = loader.load()

    print("Number of pages:", len(documents))

    for i, doc in enumerate(documents[:3]):
        print(f"\n--- Page {i + 1} ---")
        print(repr(doc.page_content[:500]))

    return documents



def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)
    return chunks


def create_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings


def store_in_pinecone(chunks, embeddings):
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME
    )
if __name__ == "__main__":
    documents = load_pdf()
    print(f"Loaded {len(documents)} pages")

    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    embeddings = create_embeddings()

    store_in_pinecone(chunks, embeddings)
    print("Successfully stored chunks in Pinecone!")