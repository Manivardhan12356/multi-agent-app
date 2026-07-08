import os
import sys
from pathlib import Path
from langchain_groq import ChatGroq

# Ensure the project root is in the Python search path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load .env file manually to avoid external dependency issues
env_path = project_root / ".env"
if env_path.exists():
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                key, val = line.strip().split("=", 1)
                os.environ[key] = val.strip().strip('"').strip("'")


# pyrefly: ignore [missing-import]
from src.converDocument import process_all_pdfs
# pyrefly: ignore [missing-import]
from src.splitDocuments import split_documents
# pyrefly: ignore [missing-import]
from src.embeedingChunks import EmbeddingManager
# pyrefly: ignore [missing-import]
from src.vectorstore import VectorStore


def main():
    print("==================================================")
    print("Starting Custom RAG Pipeline (using new src folder)")
    print("==================================================")

    # 1. Initialize Clients
    data_dir = project_root / "data"
    pdf_dir = data_dir / "pdf"
    persist_dir = data_dir / "vector_store"

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        print("[WARNING] 'GROQ_API_KEY' not found in environment or .env file.")
        print("Please configure GROQ_API_KEY to generate LLM summary answers.\n")

    # Initialize Embedding Manager
    embedding_manager = EmbeddingManager(model_name="all-MiniLM-L6-v2")

    # Initialize Vector Store (using the persist directory)
    vector_store = VectorStore(
        collection_name="pdf_documents",
        persist_directory=str(persist_dir)
    )

    # 2. Check if the database needs population
    doc_count = vector_store.collection.count()
    print(f"Current documents in ChromaDB: {doc_count}")

    if doc_count == 0:
        print("\nVector database is empty. Ingesting documents...")
        
        # Load PDF documents
        documents = process_all_pdfs(str(pdf_dir))
        
        if not documents:
            print("[ERROR] No documents loaded. Please add PDF files to data/pdf/ directory.")
            return

        # Split documents into chunks
        chunks = split_documents(documents, chunk_size=1000, chunk_overlap=200)

        # Generate embeddings
        texts = [doc.page_content for doc in chunks]
        embeddings = embedding_manager.generate_embeddings(texts)

        # Add to vector store
        vector_store.add_documents(chunks, embeddings)
        print("Ingestion completed successfully!")
    else:
        print("Using existing collection in ChromaDB.")

    # 3. Perform RAG query
    query = "What is XGBoost and when should I use it?"
    print("\n" + "=" * 60)
    print(f"Query: '{query}'")
    print("=" * 60)

    # Generate query embedding
    query_embeddings = embedding_manager.generate_embeddings([query])
    query_vector = query_embeddings[0]

    # Retrieve similar chunks
    retrieved_results = vector_store.query(query_vector.tolist(), top_k=3)

    print("\n--- Retrieved Chunks ---")
    for rank, res in enumerate(retrieved_results, 1):
        source = res['metadata'].get('source_file', 'Unknown')
        snippet = res['content'].replace('\n', ' ')[:120] + "..."
        print(f"Rank {rank} - Source: {source} | Snippet: {snippet}")

    # 4. Generate Answer using Groq LLM
    if groq_api_key and retrieved_results:
        print("\n--- Generating Summary Answer (using Llama 3.3 70B on Groq) ---")
        
        # Build Context
        context = "\n\n".join([res['content'] for res in retrieved_results])
        
        prompt = f"""You are a helpful assistant. Use the following pieces of context to answer the user's question.
If you do not know the answer or if the context does not contain the answer, say "I cannot find the answer in the provided documents."

Context:
{context}

Question:
{query}

Answer:"""

        try:
            # Initialize Groq LLM
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                api_key=groq_api_key
            )
            
            response = llm.invoke(prompt)
            print("\nAnswer:")
            print(response.content)
            
        except Exception as e:
            print(f"\n[ERROR] Failed to generate response from Groq: {e}")
    else:
        print("\nSkipping answer generation (missing API key or no retrieved context).")

if __name__ == "__main__":
    main()
