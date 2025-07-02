# 🩺 MediBot AI
MediBot is an AI-powered medical chatbot designed to help people get trusted, fact-based medical information using RAG (Retrieval Augmented Generation), LangChain, Qdrant, and Supabase — all wrapped in a simple Streamlit app.

🚀 Features

✅ Smart Retrieval — Reads medical PDFs and stores the knowledge in a Qdrant vector database.

✅ Accurate Answers — Uses Hugging Face embeddings + Gemini 1.5 Flash LLM to answer questions only from trusted context.

✅ Conversational Memory — Stores chat history in Supabase for a context-aware experience.

✅ Easy Interface — Built with Streamlit for a clean, user-friendly chat UI.

⚙️ Tech Stack

LangChain — For orchestrating the RAG pipeline.

Qdrant — Vector store for fast semantic search over medical documents.

Hugging Face Embeddings — To convert text into meaningful vectors.

Gemini 1.5 Flash — LLM for safe, fact-grounded answers.

Supabase — Manages persistent chat history.

Streamlit — Frontend for user interaction.

💡 Why MediBot?

Not everyone has immediate access to medical professionals. MediBot helps bridge that gap by giving people reliable, AI-assisted answers from trusted sources — safely and privately.

📂 Files

app.py — Streamlit app

helper.py — Helper functions & RAG chain

requirements.txt — Dependencies
hello 123