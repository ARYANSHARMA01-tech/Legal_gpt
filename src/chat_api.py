from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import asyncio
import os
import datetime
import logging
import sys
from contextlib import asynccontextmanager
from crewai import Agent, Task, Crew, Process
from crewai.flow.flow import Flow, start, listen
from crewai.llm import LLM
import uvicorn
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
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
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Alternative to Groq
        self.ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.CHATBOT_NAME = os.getenv("CHATBOT_NAME", "WebsiteBot")
        self.CHATBOT_DESCRIPTION = os.getenv("CHATBOT_DESCRIPTION", "Helpful website assistant")
        
        # Validate required environment variables
        if not self.GROQ_API_KEY and not self.OPENAI_API_KEY:
            raise ValueError("Either GROQ_API_KEY or OPENAI_API_KEY environment variable is required")

settings = Settings()

# Global clients
main_llm = None
assistant_llm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events"""
    global main_llm, assistant_llm
    
    # Startup
    logger.info("Starting Chatbot API...")
    try:
        if settings.GROQ_API_KEY:
            main_llm = LLM(model="groq/llama-3.3-70b-versatile", api_key=settings.GROQ_API_KEY)
            assistant_llm = LLM(model="groq/gemma2-9b-it", api_key=settings.GROQ_API_KEY)
        else:
            main_llm = LLM(model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY)
            assistant_llm = LLM(model="gpt-3.5-turbo", api_key=settings.OPENAI_API_KEY)
        
        logger.info("LLM clients initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize LLM clients: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Chatbot API...")

# FastAPI app initialization
app = FastAPI(
    title="CrewAI Chatbot API", 
    description="Multi-agent chatbot API powered by CrewAI", 
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Request/Response schemas
class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(default_factory=lambda: datetime.datetime.now().isoformat())

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    user_id: Optional[str] = Field(None, description="User ID for personalization")
    context: Optional[Dict] = Field(None, description="Additional context (website info, user preferences, etc.)")

class ChatResponse(BaseModel):
    status: str
    message: str
    response: Optional[str] = None
    conversation_id: Optional[str] = None
    suggestions: Optional[List[str]] = None
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str = "1.0.0"

# Conversation storage (in production, use Redis or database)
conversations: Dict[str, List[ChatMessage]] = {}

# Chatbot agents
class ChatbotAgents:
    def __init__(self, main_llm, assistant_llm):
        self.main_llm = main_llm
        self.assistant_llm = assistant_llm
        
    def intent_classifier_agent(self):
        return Agent(
            role="Intent Classification Specialist",
            goal="Analyze user messages to understand intent and categorize requests",
            backstory="""You are an expert at understanding user intentions and categorizing 
            different types of requests. You can identify whether users need information, 
            support, want to make transactions, or just casual conversation.""",
            llm=self.assistant_llm,
            verbose=True
        )
    
    def context_manager_agent(self):
        return Agent(
            role="Context Manager",
            goal="Maintain conversation context and extract relevant information",
            backstory="""You specialize in maintaining conversation flow and context. 
            You remember important details from previous messages and can reference 
            relevant information to provide coherent responses.""",
            llm=self.assistant_llm,
            verbose=True
        )
    
    def response_generator_agent(self):
        return Agent(
            role="Response Generator",
            goal="Generate helpful, accurate, and engaging responses",
            backstory=f"""You are {settings.CHATBOT_NAME}, a {settings.CHATBOT_DESCRIPTION}. 
            You provide helpful information, answer questions, and assist users with their needs. 
            You are friendly, professional, and knowledgeable about the website and its services.""",
            llm=self.main_llm,
            verbose=True
        )
    
    def quality_assurance_agent(self):
        return Agent(
            role="Quality Assurance Specialist",
            goal="Review and improve responses for accuracy and helpfulness",
            backstory="""You are responsible for ensuring all responses are accurate, 
            helpful, appropriate, and align with the website's tone and policies. 
            You also suggest follow-up questions or actions when relevant.""",
            llm=self.assistant_llm,
            verbose=True
        )

# Chatbot tasks
class ChatbotTasks:
    def classify_intent_task(self, agent, user_message, context):
        return Task(
            description=f"""
            Analyze the user message and classify the intent:
            
            User Message: {user_message}
            Context: {context}
            
            Identify:
            1. Primary intent (information, support, transaction, casual, etc.)
            2. Urgency level (low, medium, high)
            3. Required information or actions
            4. Emotional tone
            """,
            agent=agent,
            expected_output="Classified intent with category, urgency, requirements, and emotional tone"
        )
    
    def manage_context_task(self, agent, user_message, conversation_history, context):
        return Task(
            description=f"""
            Manage conversation context and extract relevant information:
            
            Current Message: {user_message}
            Conversation History: {conversation_history}
            Additional Context: {context}
            
            Extract and organize:
            1. Key information from current message
            2. Relevant context from conversation history
            3. Important details to remember
            4. Any references that need clarification
            """,
            agent=agent,
            expected_output="Organized context with key information, relevant history, and important details"
        )
    
    def generate_response_task(self, agent, user_message, context):
        return Task(
            description=f"""
            Generate a helpful and engaging response:
            
            User Message: {user_message}
            Context from previous tasks: Use the context and intent analysis from previous tasks
            
            Create a response that:
            1. Directly addresses the user's question or need
            2. Is appropriate for the identified intent and tone
            3. Provides clear and actionable information
            4. Maintains conversation flow
            5. Reflects the website's personality and brand
            """,
            agent=agent,
            expected_output="Helpful, engaging response that addresses the user's needs"
        )
    
    def quality_review_task(self, agent, user_message):
        return Task(
            description=f"""
            Review and enhance the generated response:
            
            Original User Message: {user_message}
            Generated Response: Use the response from the previous task
            
            Ensure the response:
            1. Accurately addresses the user's question
            2. Is helpful and actionable
            3. Maintains appropriate tone and style
            4. Follows website policies and guidelines
            5. Includes relevant follow-up suggestions if appropriate
            
            Provide the final polished response and any suggested follow-up questions.
            """,
            agent=agent,
            expected_output="Final polished response with optional follow-up suggestions"
        )

# Chatbot crew flow
class ChatbotFlow(Flow):
    def __init__(self, user_message: str, conversation_history: List[ChatMessage], context: Dict):
        super().__init__()
        self.user_message = user_message
        self.conversation_history = conversation_history
        self.context = context
        logger.info(f"ChatbotFlow initialized for message: {user_message[:100]}...")

    @start()
    def process_message(self):
        """Process the user message and prepare for analysis"""
        try:
            # Convert conversation history to string for context
            history_str = "\n".join([
                f"{msg.role}: {msg.content}" 
                for msg in self.conversation_history[-5:]  # Last 5 messages for context
            ])
            
            return {
                "user_message": self.user_message,
                "conversation_history": history_str,
                "context": self.context
            }
        except Exception as e:
            logger.error(f"Failed to process message: {e}")
            raise

    @listen(process_message)
    def generate_response(self, inputs):
        """Generate chatbot response using crew of agents"""
        try:
            user_message = inputs["user_message"]
            conversation_history = inputs["conversation_history"]
            context = inputs["context"]

            # Initialize agents and tasks
            agents = ChatbotAgents(main_llm, assistant_llm)
            tasks = ChatbotTasks()

            # Create agents
            intent_agent = agents.intent_classifier_agent()
            context_agent = agents.context_manager_agent()
            response_agent = agents.response_generator_agent()
            qa_agent = agents.quality_assurance_agent()

            # Create tasks
            task1 = tasks.classify_intent_task(intent_agent, user_message, context)
            task2 = tasks.manage_context_task(context_agent, user_message, conversation_history, context)
            task3 = tasks.generate_response_task(response_agent, user_message, context)
            task4 = tasks.quality_review_task(qa_agent, user_message)

            # Create and run crew
            crew = Crew(
                agents=[intent_agent, context_agent, response_agent, qa_agent],
                tasks=[task1, task2, task3, task4],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            logger.info("Chatbot response generated successfully")
            return result
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise

# Utility functions
def get_or_create_conversation(conversation_id: str) -> List[ChatMessage]:
    """Get existing conversation or create new one"""
    if conversation_id not in conversations:
        conversations[conversation_id] = []
    return conversations[conversation_id]

def add_message_to_conversation(conversation_id: str, message: ChatMessage):
    """Add message to conversation history"""
    conversation = get_or_create_conversation(conversation_id)
    conversation.append(message)
    
    # Keep only last 50 messages to prevent memory issues
    if len(conversation) > 50:
        conversations[conversation_id] = conversation[-50:]

def generate_conversation_id() -> str:
    """Generate unique conversation ID"""
    return f"conv_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"

# API endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    start_time = datetime.datetime.now()
    
    try:
        # Validate input
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or generate_conversation_id()
        
        # Get conversation history
        conversation_history = get_or_create_conversation(conversation_id)
        
        # Add user message to conversation
        user_message = ChatMessage(role="user", content=request.message)
        add_message_to_conversation(conversation_id, user_message)
        
        logger.info(f"Processing chat message for conversation: {conversation_id}")
        
        # Process with ChatbotFlow
        chatbot_flow = ChatbotFlow(
            user_message=request.message,
            conversation_history=conversation_history,
            context=request.context or {}
        )
        
        # Run the flow asynchronously
        def run_flow():
            return chatbot_flow.kickoff()
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_flow)
        
        # Add assistant response to conversation
        assistant_message = ChatMessage(role="assistant", content=str(result))
        add_message_to_conversation(conversation_id, assistant_message)
        
        processing_time = (datetime.datetime.now() - start_time).total_seconds()
        logger.info(f"Chat processed successfully in {processing_time:.2f} seconds")
        
        return ChatResponse(
            status="success",
            message="Response generated successfully",
            response=str(result),
            conversation_id=conversation_id
        )

    except HTTPException:
        raise
    except Exception as e:
        processing_time = (datetime.datetime.now() - start_time).total_seconds()
        logger.error(f"Chat processing failed after {processing_time:.2f} seconds: {e}")
        
        return ChatResponse(
            status="error",
            message="Failed to process chat message",
            error=str(e),
            conversation_id=request.conversation_id
        )

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history"""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "conversation_id": conversation_id,
        "messages": conversations[conversation_id],
        "message_count": len(conversations[conversation_id])
    }

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete conversation"""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    del conversations[conversation_id]
    return {"message": "Conversation deleted successfully"}

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
        "message": "CrewAI Chatbot API",
        "version": "1.0.0",
        "chatbot_name": settings.CHATBOT_NAME,
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

if __name__ == "__main__":
    uvicorn.run(
        "chatbot_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )