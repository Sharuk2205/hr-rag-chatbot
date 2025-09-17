# HR Policy RAG Chatbot

<img width="1454" height="623" alt="image" src="https://github.com/user-attachments/assets/710eca8a-e684-4c99-8359-dec874e91254" />


---

## Overview

This project is a **RAG-based HR Policy Chatbot** that helps employees quickly retrieve information from your company's HR policy document. The backend is built with **FastAPI** and the frontend is a **Streamlit** interface. The system uses **LangChain**, **FAISS**, and **Groq LLM** for document retrieval and question answering.

## Features

- Upload your HR Policy PDF and query it using natural language.
- Retrieves the most relevant sections using vector similarity search.
- Answers are formatted in **bullet points** for clarity.
- Simple in-memory caching for faster repeated queries.
- Full-stack setup with **FastAPI** backend and **Streamlit** frontend.

## Tech Stack

- **Python 3.11**
- **FastAPI** for backend API
- **Streamlit** for frontend
- **LangChain** (core & community) for RAG setup
- **FAISS** for vector database
- **Groq LLM** for language model
- **HuggingFace Embeddings**
- **PyPDFLoader** for PDF document loading

## Project Structure

```
RAG_HR_Chatbot/
├── main.py              # FastAPI backend
├── streamlit_app.py     # Streamlit frontend
├── HR-Policy.pdf        # HR policy document
├── requirements.txt     # Dependencies

```

## Installation (Local)

1. Clone the repository:

```bash
git clone (https://github.com/Sharuk2205/hr-rag-chatbot)
cd RAG_HR_Chatbot
```

2. Create and activate virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Add your environment variables in a `.env` file:

```
HF_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_api_key
```

5. Run the FastAPI backend:

```bash
uvicorn main:app --reload
```

6. In a separate terminal, run the Streamlit frontend:

```bash
streamlit run streamlit_app.py
```

Visit http://localhost:8501 to interact with the chatbot.



