from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Define input and output schema
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = []

# Initialize FastAPI app
app = FastAPI(title="BDRAG API", version="1.0")

# Load embedding model and vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embedding_model)

# Create retriever and QA chain
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0), 
    chain_type="stuff", 
    retriever=retriever,
    return_source_documents=True
)

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    try:
        result = qa_chain({"query": request.query})
        sources = [doc.metadata.get("source", "") for doc in result["source_documents"]]
        return QueryResponse(answer=result["result"], sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the BDRAG RAG API"}
