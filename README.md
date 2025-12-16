# 📚 Study Buddy: AI-Powered RAG for Engineering Students

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://studybuddyyyy.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-v0.3-green)
![Llama 3](https://img.shields.io/badge/Model-Llama_3.1-orange)

**Study Buddy** is a Retrieval-Augmented Generation (RAG) application designed to help students efficiently review lecture notes. Unlike generic chatbots, this tool answers questions strictly based on uploaded PDF/PPT documents and provides **page-level citations** for verification.

🔗 **Live Demo:** [https://studybuddyyyy.streamlit.app/](https://studybuddyyyy.streamlit.app/)

## 🚀 Features

* **📄 Multi-Document Ingestion:** Upload multiple PDF or PowerPoint files simultaneously (e.g., an entire semester's worth of notes).
* **🔍 Accurate Retrieval:** Uses `ChromaDB` and `HuggingFace Embeddings` to find the most relevant context for every query.
* **🧠 Advanced Reasoning:** Powered by **Llama-3.1-8b** (via Groq) for high-speed, accurate summarization.
* **📍 Source Attribution:** Every answer includes a "View Sources" dropdown, citing the exact document and page number to prevent hallucinations.
* **🔒 Secure:** API keys are handled via environment variables, ensuring no credentials are leaked in the codebase.

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **Orchestration:** [LangChain](https://www.langchain.com/)
* **Vector Database:** [ChromaDB](https://www.trychroma.com/) (Local persistent storage)
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)
* **LLM Inference:** [Groq API](https://groq.com/) (Llama 3.1 8B Instant)

## 🏗️ Architecture

1.  **Ingestion:** Documents are loaded and split into chunks (1000 characters).
2.  **Embedding:** Text chunks are converted into vector embeddings using HuggingFace.
3.  **Storage:** Vectors are stored locally in ChromaDB.
4.  **Retrieval:** The system performs a similarity search to find the top 3 chunks relevant to the user's question.
5.  **Generation:** The retrieved context + user question are sent to Llama-3 to generate the final answer.

## 💻 Installation & Local Setup

If you want to run this locally:

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/study-buddy.git](https://github.com/yourusername/study-buddy.git)
cd study-buddy

**2. Install Dependencies**

Bash

pip install -r requirements.txt
3. Set up Environment Variables Create a .env file in the root directory and add your Groq API key:

Ini, TOML

GROQ_API_KEY=gsk_your_actual_key_here
4. Run the App

Bash

streamlit run app.py
