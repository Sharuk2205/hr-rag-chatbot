import os
import re
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq.chat_models import ChatGroq
from dotenv import load_dotenv


# Loading environment variables
load_dotenv()
# API Key
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")


# Utility Function to Clean Text
def clean_text(text):
    """
    Removes extra newlines, tabs, and excessive spaces from a text string.
    """
    return " ".join(text.split())

def format_as_bullets(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    bullets = [f"- {sentence}" for sentence in sentences if sentence]
    return "\n".join(bullets)


# Initializing the Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Loading the HR Policy Document
loader=PyPDFLoader("HR-Policy.pdf")
documents=loader.load()
# Split and create embeddings for the documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
splits = text_splitter.split_documents(documents)
# Creating Vectore Store
vectorstore = FAISS.from_documents(documents=splits, embedding=embedding_model)
retriever = vectorstore.as_retriever()


# Prompt
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert HR policy assistant. Based on the context provided, give a thorough and detailed answer to the question.

Context:
{context}

Question:
{question}

Provide a detailed, well-explained answer, and list key points if needed.
"""
)


# Initializing LLM Model
llm=ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"),model_name="Gemma2-9b-It")

# Create RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
)


# Simple in-memory query cache
query_cache: Dict[str, Dict] = {}


def get_qa_response_with_cache(query: str):
    # Check cache first
    if query in query_cache:
        return query_cache[query]

    # If not cached, compute result
    # Step 1: Retrieve top 10 docs from FAISS
    docs = retriever.get_relevant_documents(query)
    # Step 2: Embed the query
    query_embedding = embedding_model.embed_query(query)
    # Step 3: Embed documents
    doc_embeddings = [embedding_model.embed_query(doc.page_content) for doc in docs]
    # Step 4: Calculate cosine similarity
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    # Step 5: Rank docs by similarity
    ranked_docs = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)
    # Select top 5 re-ranked docs
    top_docs = [doc for doc, _ in ranked_docs[:5]]
    # Step 6: Run QA chain with top re-ranked documents
    result = qa_chain({"query": query, "input_documents": top_docs})

    # Save result in cache
    query_cache[query] = result

    return result


# FastAPI app
app = FastAPI(title="HR Policy RAG Backend")

# Request models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]


 # API Endpoint
@app.post("/query", response_model=QueryResponse)
def query_hr_policy(data: QueryRequest):
    result = get_qa_response_with_cache(data.query)
    answer = format_as_bullets(clean_text(result['result']))
    sources = [
        f"Source {i+1}: {'. '.join(clean_text(doc.page_content).split('. ')[:3])}..."
        for i, doc in enumerate(result['source_documents'])
    ]
    return QueryResponse(answer=answer, sources=sources)