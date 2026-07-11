"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  Send,
  Bot,
  User,
  RefreshCw,
  FileText,
  Database,
  ChevronRight,
  AlertCircle,
  Sparkles,
  BookOpen,
  CheckCircle2,
  ExternalLink
} from "lucide-react";

interface Source {
  content: string;
  source: string;
  page: string | number;
  distance?: number;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  sources?: Source[];
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      text: "Hello! I am your RAG-augmented AI assistant. Ask me anything about the uploaded machine learning and Python guidebooks, and I'll answer using precise context from the documents.",
    }
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [dbChunks, setDbChunks] = useState<number | null>(null);
  const [isIngesting, setIsIngesting] = useState(false);
  const [ingestStatus, setIngestStatus] = useState<string | null>(null);
  const [selectedSource, setSelectedSource] = useState<Source | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "";


  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  // Fetch status of ChromaDB count
  const fetchDbStatus = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/status`);
      if (res.ok) {
        const data = await res.json();
        setDbChunks(data.document_chunks);
      }
    } catch (err) {
      console.error("Error fetching db status:", err);
    }
  };

  useEffect(() => {
    fetchDbStatus();
    // Poll status every 10 seconds
    const interval = setInterval(fetchDbStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  // Handle document ingestion
  const handleIngestion = async () => {
    setIsIngesting(true);
    setIngestStatus("Ingesting PDF documents and indexing embeddings...");
    try {
      const res = await fetch(`${API_BASE_URL}/api/ingest`, { method: "POST" });
      const data = await res.json();
      if (res.ok) {
        setIngestStatus(`Success: ${data.message}`);
        fetchDbStatus();
      } else {
        setIngestStatus(`Error: ${data.detail || "Ingestion failed"}`);
      }
    } catch (err) {
      setIngestStatus("Error: Failed to connect to backend server.");
    } finally {
      setIsIngesting(false);
      setTimeout(() => setIngestStatus(null), 5000);
    }
  };

  // Handle chat submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      text: input,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);
    setSelectedSource(null);

    try {
      const res = await fetch(`${API_BASE_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMessage.text }),
      });

      const data = await res.json();

      if (res.ok) {
        setMessages((prev) => [
          ...prev,
          {
            id: (Date.now() + 1).toString(),
            role: "assistant",
            text: data.answer,
            sources: data.sources,
          },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            id: (Date.now() + 1).toString(),
            role: "assistant",
            text: `Error: ${data.detail || "Something went wrong. Please check if your GROQ_API_KEY is configured."}`,
          },
        ]);
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          text: "Failed to connect to the backend server. Make sure the FastAPI backend is running and accessible.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSuggestionClick = (text: string) => {
    setInput(text);
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-transparent">

      {/* 1. Main Chat Area */}
      <div className="flex flex-col flex-1 h-full min-w-0 border-r border-white/5">

        {/* Header */}
        <header className="flex items-center justify-between px-6 py-4 glass-panel border-b border-white/5">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-purple-600/20 text-purple-400 border border-purple-500/30">
              <Bot className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight text-white flex items-center gap-2">
                RAG Learning Assistant
                <span className="flex h-2 w-2 relative">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                </span>
              </h1>
              <p className="text-xs text-gray-400">Powered by Llama 3.3 70B & SentenceTransformers</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 text-xs text-gray-300 border border-white/5">
              <Database className="w-4 h-4 text-purple-400" />
              <span>Chunks: <strong className="text-white">{dbChunks !== null ? dbChunks : "Loading..."}</strong></span>
            </div>

            <button
              onClick={handleIngestion}
              disabled={isIngesting}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-purple-600 hover:bg-purple-700 disabled:bg-purple-900/50 text-white text-xs font-semibold transition-all border border-purple-500/30 cursor-pointer shadow-lg shadow-purple-600/10 hover:shadow-purple-600/20"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isIngesting ? "animate-spin" : ""}`} />
              <span>{isIngesting ? "Ingesting..." : "Sync Docs"}</span>
            </button>
          </div>
        </header>

        {/* Message Panel */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {ingestStatus && (
            <div className="flex items-center gap-3 px-4 py-3 rounded-xl bg-purple-950/40 text-purple-300 border border-purple-500/20 text-sm">
              <Sparkles className="w-4 h-4 text-purple-400 animate-pulse" />
              <span>{ingestStatus}</span>
            </div>
          )}

          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex gap-4 max-w-3xl ${msg.role === "user" ? "ml-auto flex-row-reverse" : "mr-auto"
                }`}
            >
              {/* Avatar */}
              <div
                className={`flex items-center justify-center w-8 h-8 rounded-lg shrink-0 border ${msg.role === "user"
                    ? "bg-purple-600/10 border-purple-500/30 text-purple-400"
                    : "bg-white/5 border-white/5 text-gray-400"
                  }`}
              >
                {msg.role === "user" ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
              </div>

              {/* Message bubble */}
              <div className="space-y-3">
                <div
                  className={`px-4 py-3 rounded-2xl text-sm leading-relaxed border ${msg.role === "user"
                      ? "bg-gradient-to-br from-purple-700/80 to-purple-800/80 border-purple-500/20 text-white shadow-md shadow-purple-900/10"
                      : "glass-card text-gray-200"
                    }`}
                >
                  <p className="whitespace-pre-wrap">{msg.text}</p>
                </div>

                {/* Sources list */}
                {msg.role === "assistant" && msg.sources && msg.sources.length > 0 && (
                  <div className="flex flex-wrap gap-2 pt-1 pl-1">
                    {msg.sources.map((src, i) => (
                      <button
                        key={i}
                        onClick={() => setSelectedSource(src)}
                        className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs transition-all border cursor-pointer ${selectedSource === src
                            ? "bg-purple-600/20 border-purple-500/40 text-purple-300 shadow-sm"
                            : "bg-white/5 border-white/5 text-gray-400 hover:bg-white/10 hover:text-white"
                          }`}
                      >
                        <FileText className="w-3 h-3 text-purple-400" />
                        <span className="max-w-[120px] truncate">{src.source}</span>
                        <span className="text-[10px] opacity-75">(P{src.page})</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}

          {/* Typing indicator */}
          {isLoading && (
            <div className="flex gap-4 max-w-3xl mr-auto">
              <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-white/5 border border-white/5 text-gray-400 shrink-0">
                <Bot className="w-4 h-4" />
              </div>
              <div className="px-4 py-3 rounded-2xl glass-card text-gray-400 text-sm flex items-center gap-2">
                <span className="h-1.5 w-1.5 bg-purple-400 rounded-full animate-bounce"></span>
                <span className="h-1.5 w-1.5 bg-purple-400 rounded-full animate-bounce [animation-delay:0.2s]"></span>
                <span className="h-1.5 w-1.5 bg-purple-400 rounded-full animate-bounce [animation-delay:0.4s]"></span>
                <span className="ml-1 text-xs">Analyzing documents...</span>
              </div>
            </div>
          )}

          {/* Welcome Screen suggestions */}
          {messages.length === 1 && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-2xl mx-auto pt-8">
              <div className="col-span-2 text-center text-xs text-gray-500 uppercase tracking-widest font-semibold pb-2">
                Suggested Prompts
              </div>
              <button
                onClick={() => handleSuggestionClick("What is XGBoost and when should I use it?")}
                className="p-4 rounded-xl text-left glass-card hover:bg-white/5 transition-all duration-300 group cursor-pointer border border-white/5"
              >
                <div className="flex items-start justify-between">
                  <span className="text-sm font-semibold text-purple-300 group-hover:text-purple-200">XGBoost Details</span>
                  <ChevronRight className="w-4 h-4 text-gray-500 group-hover:translate-x-1 transition-transform" />
                </div>
                <p className="text-xs text-gray-400 mt-1">"What is XGBoost and when should I use it?"</p>
              </button>

              <button
                onClick={() => handleSuggestionClick("What is Supervised Learning?")}
                className="p-4 rounded-xl text-left glass-card hover:bg-white/5 transition-all duration-300 group cursor-pointer border border-white/5"
              >
                <div className="flex items-start justify-between">
                  <span className="text-sm font-semibold text-blue-300 group-hover:text-blue-200">Supervised Learning</span>
                  <ChevronRight className="w-4 h-4 text-gray-500 group-hover:translate-x-1 transition-transform" />
                </div>
                <p className="text-xs text-gray-400 mt-1">"What is Supervised Learning?"</p>
              </button>

              <button
                onClick={() => handleSuggestionClick("How are Random Forests trained?")}
                className="p-4 rounded-xl text-left glass-card hover:bg-white/5 transition-all duration-300 group cursor-pointer border border-white/5"
              >
                <div className="flex items-start justify-between">
                  <span className="text-sm font-semibold text-emerald-300 group-hover:text-emerald-200">Random Forests</span>
                  <ChevronRight className="w-4 h-4 text-gray-500 group-hover:translate-x-1 transition-transform" />
                </div>
                <p className="text-xs text-gray-400 mt-1">"How are Random Forests trained?"</p>
              </button>

              <button
                onClick={() => handleSuggestionClick("Summarize the machine learning order guide.")}
                className="p-4 rounded-xl text-left glass-card hover:bg-white/5 transition-all duration-300 group cursor-pointer border border-white/5"
              >
                <div className="flex items-start justify-between">
                  <span className="text-sm font-semibold text-pink-300 group-hover:text-pink-200">General Overview</span>
                  <ChevronRight className="w-4 h-4 text-gray-500 group-hover:translate-x-1 transition-transform" />
                </div>
                <p className="text-xs text-gray-400 mt-1">"Summarize the machine learning order guide."</p>
              </button>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Bar */}
        <footer className="p-4 glass-panel border-t border-white/5">
          <form onSubmit={handleSubmit} className="flex gap-2 max-w-3xl mx-auto">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question about your documents..."
              disabled={isLoading}
              className="flex-1 bg-white/5 text-sm text-white placeholder-gray-500 px-4 py-3 rounded-xl border border-white/5 focus:outline-none focus:border-purple-500/50 focus:ring-1 focus:ring-purple-500/50 transition-all disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="flex items-center justify-center w-11 h-11 bg-purple-600 hover:bg-purple-700 disabled:bg-purple-900/50 text-white rounded-xl transition-all border border-purple-500/30 shrink-0 cursor-pointer disabled:cursor-not-allowed"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>
        </footer>
      </div>

      {/* 2. Right Side Panel: Source Citation Inspector */}
      {selectedSource && (
        <div className="w-80 h-full glass-panel flex flex-col shrink-0 animate-in slide-in-from-right duration-300">
          <header className="p-4 border-b border-white/5 flex items-center justify-between bg-purple-950/20">
            <div className="flex items-center gap-2">
              <BookOpen className="w-4 h-4 text-purple-400" />
              <span className="text-sm font-bold text-white">Source Inspector</span>
            </div>
            <button
              onClick={() => setSelectedSource(null)}
              className="text-xs text-gray-400 hover:text-white px-2 py-1 rounded bg-white/5 hover:bg-white/10 transition-all cursor-pointer"
            >
              Close
            </button>
          </header>

          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            <div className="p-3.5 rounded-xl bg-white/5 border border-white/5 space-y-2">
              <div className="text-[10px] uppercase font-bold tracking-wider text-purple-400">File Name</div>
              <div className="text-sm font-semibold text-white break-words">{selectedSource.source}</div>

              <div className="flex justify-between items-center pt-2 text-xs text-gray-400">
                <span>Page: <strong className="text-white">{selectedSource.page}</strong></span>
                {selectedSource.distance !== undefined && (
                  <span>Distance: <strong className="text-white">{(selectedSource.distance).toFixed(4)}</strong></span>
                )}
              </div>
            </div>

            <div className="p-3.5 rounded-xl bg-white/5 border border-white/5 space-y-2 flex-1">
              <div className="text-[10px] uppercase font-bold tracking-wider text-purple-400">Retrieved Chunk Content</div>
              <div className="text-xs leading-relaxed text-gray-300 whitespace-pre-wrap font-mono max-h-[400px] overflow-y-auto p-2 bg-black/35 rounded-lg border border-white/5">
                {selectedSource.content}
              </div>
            </div>
          </div>

          <footer className="p-4 border-t border-white/5 text-[10px] text-gray-500 text-center">
            This chunk was matched via cosine similarity vectors.
          </footer>
        </div>
      )}

    </div>
  );
}
