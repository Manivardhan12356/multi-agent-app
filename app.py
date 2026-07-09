import os
import sys
import json
import contextvars
from pathlib import Path
from typing import Optional, List, Dict, Any, TypedDict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langgraph.graph import StateGraph, START, END

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


# --- Lightweight LangGraph Multi-Agent Architecture ---

class AgentState(TypedDict):
    messages: List[BaseMessage]
    next_agent: str
    sources: List[Dict[str, Any]]
    reports: List[str]
    query: str
    subquery: str

def build_lightweight_workflow(llm):
    async def supervisor_node(state: AgentState) -> Dict[str, Any]:
        """LangGraph Node: Lightweight Supervisor routes user query to Math or RAG directly"""
        supervisor_prompt = f"""You are the orchestrating Supervisor Agent. Decide who should handle this task:
1. 'math_agent' (for additions, multiplications, divisions).
2. 'rag_agent' (for document searches: resumes, python guides, ML).
3. 'compiler' (if no tools are needed, or we already have the answers).

User Query: "{state['query']}"

Respond in JSON format:
{{
  "next": "math_agent" or "rag_agent" or "compiler",
  "subquery": "query/parameters to send to the agent or null"
}}

Respond with ONLY the raw JSON string."""

        res = (await llm.ainvoke(supervisor_prompt)).content.strip()
        if res.startswith("```"):
            res = res.split("```")[1]
            if res.startswith("json"):
                res = res[4:]
                
        delegation = json.loads(res.strip())
        next_step = delegation.get("next", "compiler")
        subquery = delegation.get("subquery", state['query'])

        return {
            "next_agent": next_step,
            "subquery": subquery
        }

    async def math_agent_node(state: AgentState) -> Dict[str, Any]:
        """LangGraph Node: Directly calls math MCP wrapper without heavy deepagents wrappers"""
        subquery = state.get("subquery") or state['query']
        # Ask LLM to format args for call_mcp_tool
        parse_prompt = f"""Parse this math query: "{subquery}"
Respond in JSON: {{"tool": "add/multiply/divide", "a": number, "b": number}}
Respond with ONLY raw JSON."""
        
        parse_res = (await llm.ainvoke(parse_prompt)).content.strip()
        if parse_res.startswith("```"):
            parse_res = parse_res.split("```")[1]
            if parse_res.startswith("json"):
                parse_res = parse_res[4:]
        
        parsed = json.loads(parse_res.strip())
        tool_name = parsed["tool"]
        args = {"a": float(parsed["a"]), "b": float(parsed["b"])}
        
        result = await call_mcp_tool(tool_name, args)
        reports = list(state['reports'])
        reports.append(f"Math calculation result: {result}")
        
        return {
            "reports": reports,
            "next_agent": "compiler"
        }

    async def rag_agent_node(state: AgentState) -> Dict[str, Any]:
        """LangGraph Node: Directly calls vector store lookup without heavy deepagents wrappers"""
        subquery = state.get("subquery") or state['query']
        result = retrieve_pdf_context.invoke(subquery)
        
        reports = list(state['reports'])
        reports.append(f"Document search result:\n{result}")
        
        sources = list(state['sources'])
        sources.extend(request_sources.get())
        
        return {
            "sources": sources,
            "reports": reports,
            "next_agent": "compiler"
        }

    async def compiler_node(state: AgentState) -> Dict[str, Any]:
        """LangGraph Node: Compiles worker reports or answers directly"""
        if not state['reports']:
            response = await llm.ainvoke(state['query'])
            ans = response.content
        else:
            reports_str = "\n\n".join(state['reports'])
            compilation_prompt = f"""You are the Supervisor Agent. Compile a final response for the user based on the reports from your worker agents.
Use the worker reports to answer the query accurately.

User Query: "{state['query']}"

Worker Reports:
{reports_str}

Final Answer:"""

            response = await llm.ainvoke(compilation_prompt)
            ans = response.content
            
        new_messages = list(state['messages'])
        new_messages.append(AIMessage(content=ans))
        
        return {
            "messages": new_messages
        }

    # Assemble graph
    workflow = StateGraph(AgentState)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("math_agent", math_agent_node)
    workflow.add_node("rag_agent", rag_agent_node)
    workflow.add_node("compiler", compiler_node)
    
    workflow.set_entry_point("supervisor")
    
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next_agent"],
        {
            "math_agent": "math_agent",
            "rag_agent": "rag_agent",
            "compiler": "compiler"
        }
    )
    
    workflow.add_edge("math_agent", "compiler")
    workflow.add_edge("rag_agent", "compiler")
    workflow.add_edge("compiler", END)
    
    return workflow.compile()


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
    """Query the RAG pipeline using a lightweight LangGraph Multi-Agent architecture"""
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

        # Build and compile graph
        graph = build_lightweight_workflow(llm)

        # Initialize state variables
        initial_state = {
            "messages": [HumanMessage(content=request.message)],
            "next_agent": "",
            "sources": [],
            "reports": [],
            "query": request.message,
            "subquery": ""
        }

        # Execute LangGraph workflow
        final_state = await graph.ainvoke(initial_state)
        
        answer = final_state["messages"][-1].content
        sources = final_state["sources"]

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
