import os
import sys
import contextvars
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from deepagents import create_deep_agent

# Ensure the project root is in the Python search path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load .env file manually
env_path = project_root / ".env"
if env_path.exists():
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                key, val = line.strip().split("=", 1)
                os.environ[key] = val.strip().strip('"').strip("'")

# Import custom modules
# pyrefly: ignore [missing-import]
from src.converDocument import process_all_pdfs
# pyrefly: ignore [missing-import]
from src.splitDocuments import split_documents
# pyrefly: ignore [missing-import]
from src.embeedingChunks import EmbeddingManager
# pyrefly: ignore [missing-import]
from src.vectorstore import VectorStore

app = FastAPI(title="RAG Chatbot API")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models and components
embedding_manager = EmbeddingManager(model_name="all-MiniLM-L6-v2")
vector_store = VectorStore(
    collection_name="pdf_documents",
    persist_directory=str(project_root / "data" / "vector_store")
)

# Configure parameters to launch our separate mcp_server.py stdio process
python_executable = str(project_root / ".venv" / "bin" / "python")
mcp_script = str(project_root / "mcp_server.py")
mcp_params = StdioServerParameters(
    command=python_executable,
    args=[mcp_script],
    env=None
)

# Thread-safe request scope context variable to capture citations during agent execution
request_sources = contextvars.ContextVar("request_sources", default=[])

async def call_mcp_tool(tool_name: str, arguments: dict) -> str:
    """Connect to the separate mcp_server.py stdio process and execute a tool"""
    try:
        async with stdio_client(mcp_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                response = await session.call_tool(tool_name, arguments)
                if response.content and len(response.content) > 0:
                    return response.content[0].text
                return "Empty response from math tool."
    except Exception as e:
        return f"Error executing MCP tool: {str(e)}"

# --- Define LLM Wrapper Tools linking to the separate mcp_server.py ---

@tool
async def add(a: float, b: float) -> str:
    """Add two numbers together. Use this tool only when the query requires adding numbers or performing addition."""
    return await call_mcp_tool("add", {"a": a, "b": b})

@tool
async def multiply(a: float, b: float) -> str:
    """Multiply two numbers together. Use this tool only when the query requires multiplication, multiplying numbers, or computing products."""
    return await call_mcp_tool("multiply", {"a": a, "b": b})

@tool
async def divide(a: float, b: float) -> str:
    """Divide a by b. Use this tool only when the query requires division, dividing numbers, or calculating quotients."""
    return await call_mcp_tool("divide", {"a": a, "b": b})

@tool
def retrieve_pdf_context(query: str) -> str:
    """
    Search your indexed machine learning and Python PDF files for semantic matches.
    Use this tool when the user asks questions about machine learning concepts, algorithms (e.g. XGBoost, Random Forests), coding, Python guides, resumes, or any details inside your document library.
    """
    query_embeddings = embedding_manager.generate_embeddings([query])
    query_vector = query_embeddings[0]
    retrieved = vector_store.query(query_vector.tolist(), top_k=3)
    
    # Store retrieved metadata in request-scoped contextvar
    current_sources = request_sources.get()
    for res in retrieved:
        current_sources.append({
            "content": res["content"],
            "source": res["metadata"].get("source_file", "Unknown File"),
            "page": res["metadata"].get("page", "N/A"),
            "distance": res.get("distance")
        })
    request_sources.set(current_sources)
    
    if not retrieved:
        return "No relevant context found in documents."
        
    formatted = []
    for r in retrieved:
        formatted.append(f"[Source: {r['metadata'].get('source_file')}, Page: {r['metadata'].get('page')}]\n{r['content']}")
    return "\n\n".join(formatted)


class ChatRequest(BaseModel):
    message: str
    top_k: Optional[int] = 3

@app.get("/api/status")
def get_status():
    """Retrieve database status and document count"""
    try:
        count = vector_store.collection.count()
        return {"status": "ready", "document_chunks": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest")
def trigger_ingest():
    """Force re-ingest all PDFs in the data/pdf directory"""
    try:
        pdf_dir = project_root / "data" / "pdf"
        documents = process_all_pdfs(str(pdf_dir))
        if not documents:
            return {"status": "no_documents", "message": "No PDFs found in data/pdf"}
        
        chunks = split_documents(documents, chunk_size=1000, chunk_overlap=200)
        texts = [doc.page_content for doc in chunks]
        embeddings = embedding_manager.generate_embeddings(texts)
        
        vector_store.add_documents(chunks, embeddings)
        return {
            "status": "success",
            "message": f"Successfully ingested {len(chunks)} chunks from {len(documents)} PDFs."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Query the RAG pipeline using the imported deepagents library, calling mcp_server for math operations"""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Query message cannot be empty")
        
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY environment variable not configured on server."
        )

    try:
        # Reset context variable for this request
        request_sources.set([])

        # Initialize Groq LLM
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            api_key=groq_api_key
        )

        # Initialize the Deep Agent from deepagents library
        agent = create_deep_agent(
            model=llm,
            tools=[add, multiply, divide, retrieve_pdf_context],
            system_prompt=(
                "You are a helpful Deep Reasoning Agent. You have access to tools: add, multiply, divide, and retrieve_pdf_context.\n"
                "IMPORTANT: When calling a tool, you MUST output the tool arguments directly in the JSON object inside the function tags, e.g. <function=multiply>{\"a\": 2, \"b\": 5}</function>.\n"
                "DO NOT wrap the arguments in a \"parameters\" key or a \"type\" key. The JSON object must contain only the raw parameters defined by the tool."
            )
        )

        # Invoke the agent asynchronously
        response = await agent.ainvoke({"messages": [("user", request.message)]})
        answer = response["messages"][-1].content
        sources = request_sources.get()

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
