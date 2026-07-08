import subprocess
import sys
import time
import os
from pathlib import Path

def run():
    project_root = Path(__file__).resolve().parent
    
    print("==================================================")
    print("Starting RAG Application Services")
    print("==================================================")

    # 1. Start FastAPI Backend
    print("\n[1/2] Starting FastAPI Backend on port 8000...")
    backend_cmd = [str(project_root / ".venv" / "bin" / "uvicorn"), "app:app", "--host", "127.0.0.1", "--port", "8000", "--reload"]
    backend_process = subprocess.Popen(
        backend_cmd,
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Give the backend a brief moment to start
    time.sleep(2)
    if backend_process.poll() is not None:
        print("[ERROR] FastAPI backend failed to start. Output:")
        print(backend_process.stdout.read())
        return

    # 2. Start Next.js Frontend
    print("[2/2] Starting Next.js Frontend on port 3000...")
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(project_root / "frontend"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    time.sleep(2)
    if frontend_process.poll() is not None:
        print("[ERROR] Next.js frontend failed to start. Output:")
        print(frontend_process.stdout.read())
        backend_process.terminate()
        return

    print("\n" + "=" * 50)
    print("🚀 Both services are running successfully!")
    print("   👉 Chat UI:   http://localhost:3000")
    print("   👉 API Docs:  http://127.0.0.1:8000/docs")
    print("=" * 50)
    print("\nPress Ctrl+C to stop both services.\n")

    try:
        # Keep monitoring and printing outputs
        while True:
            # We can read non-blocking if needed, but simple sleep keeps it clean
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nStopping services gracefully...")
        frontend_process.terminate()
        backend_process.terminate()
        
        # Wait for them to exit
        frontend_process.wait()
        backend_process.wait()
        print("Services stopped.")

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"\nAn error occurred: {e}")
