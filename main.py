import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
import shutil

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI(title="DocuChat - AI Document Q&A")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.2
)

vectorstore = None


@app.get("/")
def home():
    return {"message": "DocuChat API is running"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore

    os.makedirs("temp", exist_ok=True)
    file_path = f"temp/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    os.remove(file_path)

    return {"message": "PDF processed successfully", "chunks_created": len(chunks)}


@app.post("/ask")
async def ask_question(question: str = Form(...)):
    global vectorstore

    if vectorstore is None:
        return {"error": "No document uploaded yet. Please upload a PDF first."}

    relevant_docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = "Answer the question based only on the following context from the document.\n"
    prompt += "If the answer isn't in the context, say you don't know based on the document.\n\n"
    prompt += "Context:\n" + context + "\n\n"
    prompt += "Question: " + question + "\n\n"
    prompt += "Answer:"

    response = llm.invoke(prompt)

    return {"question": question, "answer": response.content}