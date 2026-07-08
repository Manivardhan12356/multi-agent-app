import sys
from fastmcp import FastMCP

# Initialize MCP Server
mcp = FastMCP("Math-Calculator-Server")

@mcp.tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

@mcp.tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

@mcp.tool
def divide(a: float, b: float) -> str:
    """Divide a by b."""
    if b == 0:
        return "Error: Division by zero."
    return str(a / b)

if __name__ == "__main__":
    # Run the server (defaults to stdio transport)
    mcp.run()
