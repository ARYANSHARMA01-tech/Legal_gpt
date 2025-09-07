from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from typing import Optional, List
import asyncio
import os
import io
import base64
import datetime
import logging
import sys
from contextlib import asynccontextmanager
import pdfplumber
from mistralai import Mistral
from crewai import Agent, Task, Crew, Process
from crewai.flow.flow import Flow, start, listen
from crewai.llm import LLM
import uvicorn
from dotenv import load_dotenv
import uuid
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('legal_gpt.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Settings and Configuration
class Settings:
    def __init__(self):
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.MISTRAL_API = os.getenv("MISTRAL_API")
        self.MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB default
        self.ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        
        # Validate required environment variables
        if not self.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable is required")
        if not self.MISTRAL_API:
            raise ValueError("MISTRAL_API environment variable is required")

settings = Settings()

# Global clients
mistral_client = None
groq_llm = None
groq_llm2 = None  # Fixed: Added missing global variable

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events"""
    global mistral_client, groq_llm, groq_llm2
    
    # Startup
    logger.info("Starting LegalGPT API...")
    try:
        mistral_client = Mistral(api_key=settings.MISTRAL_API)
        groq_llm = LLM(model="groq/llama-3.3-70b-versatile", api_key=settings.GROQ_API_KEY)
        groq_llm2 = LLM(model="groq/gemma2-9b-it", api_key=settings.GROQ_API_KEY)  # Fixed: typo in api_key
        logger.info("Clients initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize clients: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down LegalGPT API...")

# FastAPI app initialization
app = FastAPI(
    title="LegalGPT", 
    description="Legal AI Assistant API", 
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Response schemas
class LegalGPTResponse(BaseModel):
    status: str
    message: str
    result: Optional[str] = None
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str = "1.0.0"

# Helper functions
async def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF file asynchronously"""
    try:
        def _extract():
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                return "\n".join([page.extract_text() or "" for page in pdf.pages])
        
        # Run PDF extraction in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _extract)
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise HTTPException(status_code=400, detail="Failed to extract text from PDF")

async def extract_text_from_image(file_bytes: bytes, content_type: str) -> str:
    """Extract text from image using OCR"""
    try:
        base64_image = base64.b64encode(file_bytes).decode("utf-8")
        
        def _ocr():
            response = mistral_client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "image_url",
                    "image_url": f"data:{content_type};base64,{base64_image}"
                },
                include_image_base64=True
            )
            print(response.pages[0].markdown)
            return response.pages[0].markdown
        
        # Run OCR in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _ocr)
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        raise HTTPException(status_code=400, detail="Failed to extract text from image")

# Legal agents and tasks classes
class LegalAgents:
    def __init__(self, llm, llm2):
        self.llm = llm
        self.llm2 = llm2
        
    def user_query_agent(self):
        return Agent(
            role="User Query Classifier",
            goal="Classify and understand user legal queries",
            backstory="Expert at understanding legal questions and categorizing them",
            llm=self.llm2,
            verbose=True
        )
    
    def doc_retrieval_agent(self):
        return Agent(
            role="Document Retrieval Specialist",
            goal="Retrieve relevant legal documents and information",
            backstory="Skilled at finding relevant legal documents and precedents",
            llm=self.llm2,
            verbose=True
        )
    
    def legal_analysis_agent(self):
        return Agent(
            role="Legal Analysis Expert",
            goal="Analyze legal documents and provide insights",
            backstory="Experienced legal analyst with deep knowledge of law",
            llm=self.llm2,
            verbose=True
        )
    
    def drafting_agent(self):
        return Agent(
            role="Legal Document Drafter",
            goal="Draft legal responses and documents",
            backstory="Expert at drafting clear and accurate legal documents",
            llm=self.llm2,
            verbose=True
        )
    
    def validator_agent(self):
        return Agent(
            role="Legal Validator",
            goal="Validate and review legal outputs",
            backstory="Meticulous reviewer ensuring legal accuracy and compliance",
            llm=self.llm2,
            verbose=True
        )

class LegalTasks:
    def classify_user_query_task(self, agent, user_input, doc_content):
        return Task(
            description=f"""
            Classify this legal query and identify key legal areas: {user_input}
            
            Document content provided: {doc_content[:500]}...
            
            Analyze what type of legal assistance is needed and identify the main legal areas involved.
            """,
            agent=agent,
            expected_output="Classified query with category, key legal areas, and relevant context"
        )
    
    def retrieve_legal_documents_task(self, agent, user_input, doc_content):
        return Task(
            description=f"""
            Based on the user query: {user_input}
            And the document content: {doc_content[:500]}...
            
            Retrieve and organize relevant legal concepts, precedents, and framework that apply to this situation.
            """,
            agent=agent,
            expected_output="Organized legal concepts, precedents, and relevant legal framework"
        )
    
    def analyze_legal_documents_task(self, agent, user_input, doc_content):
        return Task(
            description=f"""
            Analyze the provided document content and legal concepts to identify key legal issues, rights, and obligations.
            
            User Query: {user_input}
            Document Content: {doc_content}
            
            Provide a comprehensive legal analysis identifying:
            1. Key legal issues
            2. Rights and obligations
            3. Legal implications
            4. Relevant laws and regulations
            """,
            agent=agent,
            expected_output="Comprehensive legal analysis with key findings, rights, obligations, and legal implications"
        )
    
    def draft_legal_response_task(self, agent, user_input, doc_content):
        return Task(
            description=f"""
            Draft a comprehensive legal response based on the previous analysis.
            
            Original User Query: {user_input}
            Document Content: {doc_content[:500]}...
            
            Create a clear, professional legal response that:
            1. Addresses the user's specific question
            2. Explains relevant legal concepts
            3. Provides actionable recommendations
            4. Uses the analysis from the previous task
            
            Use the output from the previous legal analysis task to inform your response.
            """,
            agent=agent,
            expected_output="Professional legal response addressing the user's query with clear explanations and recommendations"
        )
    
    def validate_legal_output_task(self, agent, user_input):
        return Task(
            description=f"""
            Review and validate the drafted legal response for accuracy, completeness, and clarity.
            
            Original User Query: {user_input}
            
            Ensure the response:
            1. Properly addresses the user's query
            2. Provides actionable guidance
            3. Is legally sound and accurate
            4. Is clear and understandable
            
            Use the output from the previous drafting task to create the final response.
            """,
            agent=agent,
            expected_output="Final validated and refined legal response ready for delivery and also provide some real-life steps to take by the person"
        )

# LegalGPT crew flow
class LegalGPT(Flow):
    def __init__(self, user_query: str, processed_document: str):
        super().__init__()
        self.user_query = user_query
        self.processed_document = processed_document
        logger.info(f"LegalGPT flow initialized for query: {user_query[:100]}...")

    @start()
    def get_user_input(self):
        """Process user input and prepare for analysis"""
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant designed to clarify user legal questions."},
                {"role": "user", "content": self.user_query}
            ]
            response = groq_llm.call(messages=messages)
            self.llm_query = response
            logger.info("User input processed successfully")
            return {"llm_query": self.llm_query, "doc_text": self.processed_document}
        except Exception as e:
            logger.error(f"Failed to process user input: {e}")
            raise

    @listen(get_user_input)
    def response(self, inputs):
        """Generate legal response using crew of agents"""
        try:
            llm_query = inputs["llm_query"]
            doc_text = inputs["doc_text"]

            agents = LegalAgents(llm=groq_llm, llm2=groq_llm2)
            tasks = LegalTasks()

            # Create agents
            user_query_agent = agents.user_query_agent()
            doc_retrieval_agent = agents.doc_retrieval_agent()
            legal_analysis_agent = agents.legal_analysis_agent()
            drafting_agent = agents.drafting_agent()
            validator_agent = agents.validator_agent()

            # Create tasks with full context - each task gets the original query and document
            task1 = tasks.classify_user_query_task(
                agent=user_query_agent, 
                user_input=self.user_query,
                doc_content=doc_text
            )
            
            task2 = tasks.retrieve_legal_documents_task(
                agent=doc_retrieval_agent, 
                user_input=self.user_query,
                doc_content=doc_text
            )
            
            task3 = tasks.analyze_legal_documents_task(
                agent=legal_analysis_agent, 
                user_input=self.user_query,
                doc_content=doc_text
            )
            
            task4 = tasks.draft_legal_response_task(
                agent=drafting_agent, 
                user_input=self.user_query,
                doc_content=doc_text
            )
            
            task5 = tasks.validate_legal_output_task(
                agent=validator_agent, 
                user_input=self.user_query
            )

            # Create and run crew
            crew = Crew(
                agents=[user_query_agent, doc_retrieval_agent, legal_analysis_agent, drafting_agent, validator_agent],
                tasks=[task1, task2, task3, task4, task5],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            logger.info("Legal analysis completed successfully")
            return result
        except Exception as e:
            logger.error(f"Failed to generate legal response: {e}")
            raise

# Utility functions
def validate_file_size(file_size: int) -> bool:
    """Validate file size against maximum allowed size"""
    return file_size <= settings.MAX_FILE_SIZE

def validate_file_type(content_type: str) -> bool:
    """Validate file type"""
    allowed_types = ["application/pdf", "image/jpeg", "image/png", "image/gif", "image/webp"]
    return content_type in allowed_types

# API endpoints
@app.post("/process", response_model=LegalGPTResponse)
async def process_legal_query(
    file: UploadFile = File(...), 
    user_query: str = Form(...),
    # token: str = Depends(security)  # Uncomment for authentication
):
    """Process legal document and query"""
    start_time = datetime.datetime.now()
    
    try:
        # Validate input
        if not user_query.strip():
            raise HTTPException(status_code=400, detail="User query cannot be empty")
        
        # Read file
        file_bytes = await file.read()
        
        # Validate file size
        if not validate_file_size(len(file_bytes)):
            raise HTTPException(
                status_code=413, 
                detail=f"File size exceeds maximum allowed size of {settings.MAX_FILE_SIZE} bytes"
            )
        
        # Validate file type
        if not validate_file_type(file.content_type):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file.content_type}"
            )
        
        logger.info(f"Processing file: {file.filename}, type: {file.content_type}, size: {len(file_bytes)} bytes")
        
        # Extract text based on file type
        if file.content_type == "application/pdf":
            extracted_text = await extract_text_from_pdf(file_bytes)
        elif file.content_type.startswith("image/"):
            extracted_text = await extract_text_from_image(file_bytes, file.content_type)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the file")
        
        logger.info(f"Extracted {len(extracted_text)} characters from document")
        
        # Process with LegalGPT
        legal_gpt = LegalGPT(user_query=user_query, processed_document=extracted_text)
        
        # Run the flow asynchronously
        def run_flow():
            return legal_gpt.kickoff()
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_flow)
        
        processing_time = (datetime.datetime.now() - start_time).total_seconds()
        logger.info(f"Legal query processed successfully in {processing_time:.2f} seconds")
        
        return LegalGPTResponse(
            status="success",
            message="Legal query processed successfully",
            result=str(result)
        )

    except HTTPException:
        raise
    except Exception as e:
        processing_time = (datetime.datetime.now() - start_time).total_seconds()
        logger.error(f"Processing failed after {processing_time:.2f} seconds: {e}")
        
        return LegalGPTResponse(
            status="error",
            message="Failed to process legal query",
            error=str(e)
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.datetime.now().isoformat()
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LegalGPT API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail, "timestamp": datetime.datetime.now().isoformat()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error", 
            "message": "Internal server error", 
            "timestamp": datetime.datetime.now().isoformat()
        }
    )
# In your FastAPI app.py

# ... (existing imports and setup) ...

# Global dictionary to store document content by ID (for demonstration, use a proper DB/cache in production)
document_store = {}

class DocumentUploadResponse(BaseModel):
    status: str
    message: str
    document_id: str
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())

@app.post("/upload-document", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        if not validate_file_size(len(file_bytes)):
            raise HTTPException(status_code=413, detail=f"File size exceeds {settings.MAX_FILE_SIZE} bytes")
        if not validate_file_type(file.content_type):
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

        if file.content_type == "application/pdf":
            extracted_text = await extract_text_from_pdf(file_bytes)
        elif file.content_type.startswith("image/"):
            extracted_text = await extract_text_from_image(file_bytes, file.content_type)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the file")
        
        document_id = str(uuid.uuid4()) # Generate a unique ID
        document_store[document_id] = extracted_text
        logger.info(f"Document uploaded and processed with ID: {document_id}")

        return DocumentUploadResponse(
            status="success",
            message="Document uploaded and processed successfully",
            document_id=document_id
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {e}")

class ChatQuery(BaseModel):
    document_id: str
    user_query: str

@app.post("/chat", response_model=LegalGPTResponse)
async def chat_with_document(query: ChatQuery):
    start_time = datetime.datetime.now()
    try:
        if query.document_id not in document_store:
            raise HTTPException(status_code=404, detail="Document ID not found.")
        
        processed_document = document_store[query.document_id]

        legal_gpt = LegalGPT(user_query=query.user_query, processed_document=processed_document)
        def run_flow():
            return legal_gpt.kickoff()
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_flow)
        
        processing_time = (datetime.datetime.now() - start_time).total_seconds()
        logger.info(f"Legal query processed successfully in {processing_time:.2f} seconds for doc ID: {query.document_id}")
        
        return LegalGPTResponse(
            status="success",
            message="Legal query processed successfully",
            result=str(result)
        )
    except HTTPException:
        raise
    except Exception as e:
        processing_time = (datetime.datetime.now() - start_time).total_seconds()
        logger.error(f"Chat processing failed after {processing_time:.2f} seconds: {e}")
        return LegalGPTResponse(
            status="error",
            message="Failed to process legal query",
            error=str(e)
        )
if __name__ == "__main__":
    uvicorn.run(
        "api_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Set to True for development
        log_level="info"
    )