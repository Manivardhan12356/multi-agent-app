import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

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

# Configure parameters to launch our separate mcp_server.py
python_executable = str(project_root / ".venv" / "bin" / "python")
mcp_script = str(project_root / "mcp_server.py")
mcp_params = StdioServerParameters(
    command=python_executable,
    args=[mcp_script],
    env=None
)

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

# --- Specialized Agent Workers ---

async def run_math_agent(query: str, llm) -> str:
    """Worker Agent specialized in mathematical operations using MCP Calculator Server"""
    try:
        math_parse_prompt = f"""You are a specialized Math Worker Agent. 
Extract the math operation ('add', 'multiply', or 'divide') and the arguments 'a' and 'b' from this query:
"{query}"

Respond in JSON format: {{"tool": "add/multiply/divide", "a": number, "b": number}}
Do not write any other text, code, or explanation."""
        
        parse_res = (await llm.ainvoke(math_parse_prompt)).content.strip()
        if parse_res.startswith("```"):
            parse_res = parse_res.split("```")[1]
            if parse_res.startswith("json"):
                parse_res = parse_res[4:]
        
        parsed = json.loads(parse_res.strip())
        tool_name = parsed["tool"]
        args = {"a": float(parsed["a"]), "b": float(parsed["b"])}
        
        # Execute math tool on the separate MCP Server
        result = await call_mcp_tool(tool_name, args)
        return f"Math Agent: Calculated result for '{query}' is: {result}"
    except Exception as e:
        return f"Math Agent Error: Failed to perform math calculation: {str(e)}"

async def run_rag_agent(query: str, top_k: int, llm) -> tuple[str, list[dict]]:
    """Worker Agent specialized in retrieving and summarizing context from PDF documents"""
    try:
        # Generate query embedding
        query_embeddings = embedding_manager.generate_embeddings([query])
        query_vector = query_embeddings[0]
        retrieved = vector_store.query(query_vector.tolist(), top_k=top_k)

        sources = []
        for res in retrieved:
            sources.append({
                "content": res["content"],
                "source": res["metadata"].get("source_file", "Unknown File"),
                "page": res["metadata"].get("page", "N/A"),
                "distance": res.get("distance")
            })

        if not retrieved:
            return "RAG Agent: No relevant context found in documents.", []

        context = "\n\n".join([res["content"] for res in retrieved])
        prompt = f"""You are a specialized RAG Worker Agent. Use the following context to answer the user query.
If the context does not contain the answer, state that clearly.

Context:
{context}

Query:
{query}

Answer:"""
        
        response = await llm.ainvoke(prompt)
        return f"RAG Agent PDF Context Report:\n\n{response.content}", sources
    except Exception as e:
        return f"RAG Agent Error: Failed to retrieve document context: {str(e)}", []

# --- Supervisor / Orchestrator Agent ---

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Query the RAG pipeline using a stateful Supervisor-Worker Multi-Agent architecture"""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Query message cannot be empty")
        
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY environment variable not configured on server."
        )

    try:
        # Initialize Groq LLM
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            api_key=groq_api_key
        )

        # 1. Supervisor classification & task delegation step
        supervisor_prompt = f"""You are an orchestrating Supervisor Agent. Your job is to coordinate worker agents:
1. 'RAG Agent' (searches PDF documents for information).
2. 'Math Agent' (performs arithmetic calculations: addition, multiplication, division).

Review the user query: "{request.message}"
Decide which agent(s) need to be activated. If the query requires BOTH document info and math (e.g. "find years of experience in my CV and add them up"), activate both.

Respond with a JSON structure containing:
{{
  "activate_rag": true/false,
  "rag_subquery": "subquery text for document search or null",
  "activate_math": true/false,
  "math_subquery": "subquery text for calculation or null"
}}

Respond with ONLY the raw JSON string (do not include markdown syntax or formatting)."""

        sup_res = (await llm.ainvoke(supervisor_prompt)).content.strip()
        if sup_res.startswith("```"):
            sup_res = sup_res.split("```")[1]
            if sup_res.startswith("json"):
                sup_res = sup_res[4:]
                
        delegation = json.loads(sup_res.strip())
        
        worker_reports = []
        sources = []

        # 2. Run workers concurrently based on Supervisor delegation
        tasks = []
        if delegation.get("activate_rag"):
            tasks.append(run_rag_agent(delegation["rag_subquery"], request.top_k, llm))
        if delegation.get("activate_math"):
            tasks.append(run_math_agent(delegation["math_subquery"], llm))

        if tasks:
            results = await asyncio.gather(*tasks)
            for res in results:
                if isinstance(res, tuple):  # RAG worker returns (report, sources)
                    worker_reports.append(res[0])
                    sources.extend(res[1])
                else:  # Math worker returns report string
                    worker_reports.append(res)
        
        # 3. Compile the final unified answer
        if worker_reports:
            reports_str = "\n\n".join(worker_reports)
            compilation_prompt = f"""You are the Supervisor Agent. Compile a final, friendly, and complete response for the user based on the reports from your worker agents.
Use the worker reports to answer the query accurately.

User Query: "{request.message}"

Worker Reports:
{reports_str}

Final Answer:"""
            
            final_response = await llm.ainvoke(compilation_prompt)
            answer = final_response.content
        else:
            # Fallback to direct conversational response if no workers were activated
            response = await llm.ainvoke(request.message)
            answer = response.content

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
