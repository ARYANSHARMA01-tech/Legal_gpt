from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import os
import uuid
from dotenv import load_dotenv 
load_dotenv() 
os.environ['GROQ_API_KY']=os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="RAG PDF Chatbot API", version="1.0.0")

# Global storage
chat_sessions = {}  # Store chat histories by session_id
vectorstores = {}  # Store vector stores by session_id
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Pydantic models
class UploadResponse(BaseModel):
    session_id: str
    message: str
    files_processed: List[str]

class ChatRequest(BaseModel):
    session_id: str
    question: str
    groq_api_key: str

class ChatResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    chat_history: List[dict]

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create chat history for a session"""
    if session_id not in chat_sessions:
        chat_sessions[session_id] = ChatMessageHistory()
    return chat_sessions[session_id]

@app.post("/upload", response_model=UploadResponse)
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """
    Upload PDF files and return a session_id for chatting
    """
    try:
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Validate files
        pdf_files = [f for f in files if f.content_type == "application/pdf"]
        if not pdf_files:
            raise HTTPException(
                status_code=400, 
                detail="No valid PDF files uploaded. Please upload PDF files only."
            )
        
        # Process PDF files
        documents = []
        processed_files = []
        
        for file in pdf_files:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            
            try:
                # Load PDF
                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()
                documents.extend(docs)
                processed_files.append(file.filename)
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
        
        if not documents:
            raise HTTPException(
                status_code=400, 
                detail="No content could be extracted from the PDF files"
            )
        
        # Create vector store
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000, 
            chunk_overlap=500
        )
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        
        # Store vector store for this session
        vectorstores[session_id] = vectorstore
        
        return UploadResponse(
            session_id=session_id,
            message=f"Successfully processed {len(processed_files)} PDF files",
            files_processed=processed_files
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_with_pdf(request: ChatRequest):
    """
    Chat with the uploaded PDF using the session_id from upload
    """
    try:
        # Validate session exists
        if request.session_id not in vectorstores:
            raise HTTPException(
                status_code=404, 
                detail=f"Session {request.session_id} not found. Please upload PDF files first using /upload endpoint."
            )
        
        # Initialize LLM
        llm = ChatGroq(model_name="Gemma2-9b-It")
        
        # Get retriever
        retriever = vectorstores[request.session_id].as_retriever()
        
        # Create contextualize question prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # Create history-aware retriever
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        
        # Create QA prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # Create chains
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        # Create conversational chain with history
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        # Get response
        response = conversational_rag_chain.invoke(
            {"input": request.question},
            config={"configurable": {"session_id": request.session_id}}
        )
        
        # Get updated chat history
        session_history = get_session_history(request.session_id)
        chat_history = [
            {
                "type": msg.type,
                "content": msg.content
            }
            for msg in session_history.messages
        ]
        
        return ChatResponse(
            session_id=request.session_id,
            question=request.question,
            answer=response['answer'],
            chat_history=chat_history
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    return {
        "active_sessions": list(vectorstores.keys()),
        "total_sessions": len(vectorstores)
    }

@app.get("/sessions/{session_id}/history")
async def get_chat_history(session_id: str):
    """Get chat history for a specific session"""
    if session_id not in chat_sessions:
        return {"session_id": session_id, "messages": []}
    
    session_history = chat_sessions[session_id]
    chat_history = [
        {
            "type": msg.type,
            "content": msg.content
        }
        for msg in session_history.messages
    ]
    
    return {
        "session_id": session_id,
        "messages": chat_history
    }

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its associated data"""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
    if session_id in vectorstores:
        del vectorstores[session_id]
    
    return {"message": f"Session {session_id} deleted successfully"}

@app.get("/")
async def root():
    """API documentation"""
    return {
        "message": "RAG PDF Chatbot API",
        "description": "Upload PDFs to get a session_id, then chat with the content",
        "workflow": {
            "1": "POST /upload - Upload PDF files, get session_id",
            "2": "POST /chat - Chat with PDF content using session_id"
        },
        "endpoints": {
            "POST /upload": "Upload PDF files and get session_id",
            "POST /chat": "Chat with uploaded PDF content",
            "GET /sessions": "List all active sessions",
            "GET /sessions/{session_id}/history": "Get chat history for session",
            "DELETE /sessions/{session_id}": "Delete session data"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)