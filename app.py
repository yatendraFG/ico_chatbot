from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load environment variables from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Step 1: Load content from a text file
def load_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Step 2: Process and store embeddings
def create_vector_store(content):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_text(content)
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vector_store = FAISS.from_texts(documents, embeddings)
    return vector_store

# Step 3: Set up RAG chatbot using RetrievalQA
def get_rag_chain(vector_store):
    llm = ChatOpenAI(model_name="gpt-4", api_key=api_key)
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

# Define FastAPI app
app = FastAPI()

# Define request schema
class QueryRequest(BaseModel):
    query: str

# Load and prepare data
file_path = "content.txt"
content = load_text_file(file_path)
vector_store = create_vector_store(content)
qa_chain = get_rag_chain(vector_store)

# POST endpoint for chatbot
@app.post("/chat")
def chat(request: QueryRequest):
    try:
        print("🔍 Input query:", request.query)
        response = qa_chain.invoke({"query": request.query})  # ✅ This works with RetrievalQA
        print("✅ Output response:", response)
        return {"response": response["result"]}
    except Exception as e:
        print("❌ Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint for testing
@app.get("/")
def read_root():
    return {"message": "RAG Chatbot API is running"}
