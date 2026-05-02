from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os

load_dotenv()

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
URLS =[
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Get docs and split into chunks
try:
    docs = WebBaseLoader(URLS).load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    doc_splits = text_splitter.split_documents(docs)
    print(f"Total chunks created: {len(doc_splits)}")
except Exception as e:
    print(f"Error loading or processing documents: {e}")

embeddings = HuggingFaceEndpointEmbeddings(
        repo_id=EMBEDDING_MODEL,
        task="feature-extraction"
    )

# Check if the ChromaDB directory already exists to avoid re-creating the vectorstore
if not os.path.exists("./chroma_db"):
    try:
        vectorstore = Chroma.from_documents(
            documents=doc_splits, 
            collection_name="agentic-rag-docs",
            persist_directory="./chroma_db",
            embedding=embeddings
        )
        print("Vectorstore created and persisted to disk.")
    except Exception as e:
        print(f"Error creating vectorstore: {e}")

retriever = Chroma( 
    collection_name="agentic-rag-docs",
    persist_directory="./chroma_db",
    embedding_function=embeddings
    ).as_retriever()    
print("Retriever created successfully.")
