"""
🎓 Yazd University Intelligent Assistant Backend
================================================

This module provides a FastAPI-based backend service for the Yazd University
Intelligent Assistant. It integrates with various AI models and vector databases
to provide intelligent responses to user queries about university professors,
courses, and academic information.

Key Features:
- Persian language processing with specialized embeddings
- Intelligent query reformatting and enhancement
- Context-aware document retrieval with reranking
- Enhanced metadata support with Persian field names
- Self-query retriever for structured searches
- Comprehensive logging and monitoring
- Robust error handling and validation
- Performance metrics and analytics
- Metadata-based search capabilities

Author: Sepehr Masoudizad
Version: 3.0.0
Last Updated: 2024
"""

import os
import sys
import time
import json
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

# Third-party imports
import dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from pydantic.json import pydantic_encoder

# LangChain and AI model imports
from langchain_openai import ChatOpenAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain.retrievers.self_query.base import SelfQueryRetriever


# PyTorch for model operations
import torch

# =============================================================================
# CONFIGURATION AND SETUP
# =============================================================================

# Load environment variables
dotenv.load_dotenv()

# Application configuration
APP_CONFIG = {
    "name": "Yazd University Intelligent Assistant",
    "version": "3.0.0",
    "description": "AI-powered assistant for Yazd University queries with enhanced metadata support",
    "host": "0.0.0.0",
    "port": 8000,
    "debug": os.getenv("DEBUG", "False").lower() == "true"
}

# Model configuration
MODEL_CONFIG = {
    "embeddings_model": "heydariAI/persian-embeddings",
    "llm_model": "gpt-4.1",
    "llm_base_url": "https://api.avalai.ir/v1",
    "reranker_model": "jinaai/jina-reranker-v2-base-multilingual",
    "temperature": 0.0,
    "max_tokens": 2000,
    "retriever_k": 20,
    "reranker_top_n": 3,
    "metadata_fields": ["نام", "دانشکده", "آدرس بخش", "آدرس شخصی", "file_id"]
}

# Database configuration
DB_CONFIG = {
    "persist_directory": "chroma_db",
    "collection_name": "professors"
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def setup_logging() -> logging.Logger:
    """
    Configure comprehensive logging for the application.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Create logger
    logger = logging.getLogger("yazd_university_assistant")
    logger.setLevel(logging.DEBUG if APP_CONFIG["debug"] else logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(log_format, date_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler for all logs
    file_handler = logging.FileHandler(
        f"logs/app_{datetime.now().strftime('%Y%m%d')}.log",
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(log_format, date_format)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Error file handler
    error_handler = logging.FileHandler(
        f"logs/errors_{datetime.now().strftime('%Y%m%d')}.log",
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    logger.addHandler(error_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

# =============================================================================
# PYDANTIC MODELS FOR REQUEST/RESPONSE VALIDATION
# =============================================================================

class QueryRequest(BaseModel):
    """
    Pydantic model for validating incoming query requests.
    
    Attributes:
        query (str): The user's question or query
        user_id (Optional[str]): Optional user identifier for tracking
        session_id (Optional[str]): Optional session identifier
    """
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="پرسش کاربر",
        example="ایمیل استاد جهانگرد چیست؟"
    )
    user_id: Optional[str] = Field(
        None,
        description="شناسه کاربر (اختیاری)",
        example="user_123"
    )
    session_id: Optional[str] = Field(
        None,
        description="شناسه نشست (اختیاری)",
        example="session_456"
    )
    
    @validator('query')
    def validate_query(cls, v):
        """Validate that query is not empty and contains meaningful content."""
        if not v.strip():
            raise ValueError("پرسش نمی‌تواند خالی باشد یا فقط شامل فاصله باشد.")
        return v.strip()

class QueryResponse(BaseModel):
    """
    Pydantic model for structured query responses.
    
    Attributes:
        answer (str): The AI-generated answer
        context (List[str]): Retrieved context documents
        original_query (str): The original user query
        reformatted_query (str): The reformatted query used for processing
        processing_time (float): Time taken to process the query
        confidence_score (Optional[float]): Confidence score of the response
        sources_count (int): Number of sources used
        retrieval_method (Optional[str]): Retrieval method used
        structured_query (Optional[Dict]): Structured query used
    """
    answer: str = Field(..., description="AI-generated answer")
    context: List[str] = Field(..., description="Retrieved context documents")
    original_query: str = Field(..., description="Original user query")
    reformatted_query: str = Field(..., description="Reformatted query")
    processing_time: float = Field(..., description="Processing time in seconds")
    confidence_score: Optional[float] = Field(None, description="Confidence score")
    sources_count: int = Field(..., description="Number of sources used")
    retrieval_method: Optional[str] = Field(None, description="Retrieval method used")
    structured_query: Optional[Dict] = Field(None, description="Structured query used")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class HealthResponse(BaseModel):
    """
    Health check response model.
    """
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Current timestamp")
    version: str = Field(..., description="Application version")
    uptime: float = Field(..., description="Uptime in seconds")

# =============================================================================
# APPLICATION STATE AND METRICS
# =============================================================================

class ApplicationState:
    """Manages application state and metrics."""
    
    def __init__(self):
        self.start_time = time.time()
        self.total_queries = 0
        self.successful_queries = 0
        self.failed_queries = 0
        self.average_processing_time = 0.0
        self.processing_times = []
        
    def record_query(self, processing_time: float, success: bool = True, use_self_query: bool = False):
        """Record query metrics."""
        self.total_queries += 1
        if success:
            self.successful_queries += 1
        else:
            self.failed_queries += 1
            
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 100:  # Keep last 100 queries
            self.processing_times.pop(0)
            
        self.average_processing_time = sum(self.processing_times) / len(self.processing_times)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current application metrics."""
        return {
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "success_rate": (self.successful_queries / self.total_queries * 100) if self.total_queries > 0 else 0,
            "average_processing_time": self.average_processing_time,
            "uptime": time.time() - self.start_time
        }

# Initialize application state
app_state = ApplicationState()

# =============================================================================
# AI MODEL INITIALIZATION
# =============================================================================

def initialize_ai_models() -> Dict[str, Any]:
    """
    Initialize all AI models and components.
    
    Returns:
        Dict containing initialized models and components
    """
    logger.info("Initializing AI models and components...")
    
    try:
        # Validate API key
        api_key = os.getenv("AVALAI_API_KEY")
        if not api_key:
            raise ValueError("AVALAI_API_KEY not found in environment variables")
        
        os.environ["AVALAI_API_KEY"] = api_key
        logger.info("API key validated successfully")
        
        # Initialize Persian language embeddings
        logger.info(f"Loading embeddings model: {MODEL_CONFIG['embeddings_model']}")
        embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_CONFIG["embeddings_model"],
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )
        logger.info("Embeddings model loaded successfully")
        
        # Initialize vector store
        logger.info(f"Initializing vector store at: {DB_CONFIG['persist_directory']}")
        vectorstore = Chroma(
            persist_directory=DB_CONFIG["persist_directory"],
            embedding_function=embeddings,
            collection_name=DB_CONFIG["collection_name"]
        )
        logger.info("Vector store initialized successfully")
        
        # Set up retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": MODEL_CONFIG["retriever_k"]}
        )
        logger.info(f"Retriever configured with k={MODEL_CONFIG['retriever_k']}")
        
        # Initialize reranker
        logger.info(f"Loading reranker model: {MODEL_CONFIG['reranker_model']}")
        jina_model = HuggingFaceCrossEncoder(
            model_name=MODEL_CONFIG["reranker_model"],
            model_kwargs={"trust_remote_code": True}
        )
        compressor = CrossEncoderReranker(
            model=jina_model,
            top_n=MODEL_CONFIG["reranker_top_n"]
        )
        logger.info("Reranker model loaded successfully")
        
        # Set up compression retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever
        )
        logger.info("Compression retriever configured successfully")
        
        # Initialize language model
        logger.info(f"Initializing LLM: {MODEL_CONFIG['llm_model']}")
        llm = ChatOpenAI(
            base_url=MODEL_CONFIG["llm_base_url"],
            model=MODEL_CONFIG["llm_model"],
            temperature=MODEL_CONFIG["temperature"],
            max_tokens=MODEL_CONFIG["max_tokens"],
            api_key=api_key
        )
        logger.info("Language model initialized successfully")
        
        # Initialize self-query retriever
        logger.info("Initializing self-query retriever...")
        
        try:
            # Define metadata schema for self-query retriever
            metadata_field_info = [
                {"name": "نام", "description": "نام استاد", "type": "string"},
                {"name": "دانشکده", "description": "دانشکده محل تدریس", "type": "string"},
                {"name": "آدرس بخش", "description": "آدرس بخش یا دپارتمان", "type": "string"},
                {"name": "آدرس شخصی", "description": "آدرس شخصی استاد", "type": "string"},
                {"name": "file_id", "description": "شناسه فایل", "type": "string"}
            ]
            
            self_query_retriever = SelfQueryRetriever.from_llm(
                llm,
                vectorstore,
                "text",
                metadata_field_info
            )
            logger.info("Self-query retriever initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize self-query retriever: {str(e)}")
            logger.info("Falling back to regular retriever only")
            self_query_retriever = None
        
        return {
            "embeddings": embeddings,
            "vectorstore": vectorstore,
            "retriever": retriever,
            "compression_retriever": compression_retriever,
            "llm": llm,
            "self_query_retriever": self_query_retriever
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize AI models: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

# Query reformatting prompt for better search results
QUERY_REFORMAT_PROMPT = """
شما یک دستیار هوشمند برای دانشگاه یزد هستید. وظیفه شما تبدیل سوالات کاربر به فرمت مناسب برای جستجو در اطلاعات اساتید است.

سوال کاربر: {user_query}

لطفا سوال را به شکلی واضح و با جمله بندی ساختاریافته به مانند زیر تبدیل کنید.

مثال:
Question: استاد لطیف
Thought: در اینجا جمله بسیار کوتاه است و ممکن است در یافتن جملات مشابه دچار مشکل شویم بنابرین آنرا به ساختاری بلند تر تغییر میدهیم.
Reformatted Question: لطیف استاد دانشگاه یزد چه کسی است

Question: ایمیل استاد جهانگرد را به من بده
Thought: این جمله شامل کلماتی است که ممکن است باعث سردرگمی شوند مانند "من" و "بده" بنابرین آنها را حذف کرده و جمله را بازنویسی میکنیم.
Reformatted Question: ایمیل جهانگرد استاد دانشگاه یزد چیست

Question: دانشکده استاد احمدی
Thought: این سوال مربوط به فیلد دانشکده است و باید به شکل واضح‌تری بیان شود.
Reformatted Question: احمدی استاد در کدام دانشکده دانشگاه یزد تدریس می‌کند

Question: آدرس بخش استاد محمدی
Thought: این سوال مربوط به فیلد آدرس بخش است و باید به شکل واضح‌تری بیان شود.
Reformatted Question: آدرس بخش محل کار محمدی استاد دانشگاه یزد چیست

فقط جمله نهایی را برگردانید، بدون توضیح اضافی.
"""

# Main system prompt for the university assistant
SYSTEM_PROMPT = """
شما یک دستیار هوشمند برای دانشگاه یزد هستید. وظیفه شما پاسخگویی به سوالات در مورد اساتید، برنامه کلاسی، سوابق تحصیلی و تخصص‌های آنها براساس انتخاب هوشمندانه از بافتار داده شده است.

بافتار: {context}

راهنمایی‌های مهم:
۱. پاسخ‌های خود را به زبان فارسی و با لحن رسمی و محترمانه ارائه دهید
۲. اطلاعات دقیق از جمله ساعات کلاس، شماره کلاس و جزئیات تماس را به طور دقیق ذکر کنید
۳. اگر اطلاعاتی در مورد موضوعی ندارید، صادقانه اعلام کنید
۴. در مورد زمینه‌های تخصصی و پژوهشی اساتید با دقت و جزئیات پاسخ دهید
۵. برنامه کلاسی را با ذکر روز، ساعت و نام درس به طور منظم ارائه دهید
۶. از اطلاعات موجود در بافتار استفاده کنید و اطلاعات اضافی ارائه ندهید
۷. پاسخ‌ها را به صورت ساختاریافته و خوانا ارائه دهید
۸. در صورتی که اطلاعات موجود در بافتار با سوال کاربر تفاوت فاحشی دارد از پاسخ دادن بپرهیزید و اظهار بی اطلاعی کنید
۹. برای جستجوی اساتید از فیلدهای موجود استفاده کنید: نام، دانشکده، آدرس بخش، آدرس شخصی
۱۰. اطلاعات تماس و آدرس‌ها را با دقت و کامل ارائه دهید
"""

def create_prompt_chains(llm: ChatOpenAI) -> Dict[str, Any]:
    """
    Create LangChain prompt chains for query processing.
    
    Args:
        llm: Initialized language model
        
    Returns:
        Dict containing prompt chains
    """
    logger.info("Creating prompt chains...")
    
    # Query reformatting chain
    query_reformat_chain = ChatPromptTemplate.from_messages([
        ("system", QUERY_REFORMAT_PROMPT),
        ("human", "{user_query}")
    ]) | llm
    
    # Main question-answering chain
    main_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}")
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, main_prompt)
    
    logger.info("Prompt chains created successfully")
    
    return {
        "query_reformat_chain": query_reformat_chain,
        "question_answer_chain": question_answer_chain
    }

# =============================================================================
# FASTAPI APPLICATION SETUP
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    logger.info("Starting Yazd University Intelligent Assistant...")
    logger.info(f"Version: {APP_CONFIG['version']}")
    logger.info(f"Debug mode: {APP_CONFIG['debug']}")
    
    # Initialize AI models
    global ai_models, prompt_chains, main_chain, self_query_chain
    ai_models = initialize_ai_models()
    prompt_chains = create_prompt_chains(ai_models["llm"])
    
    # Create main processing chain
    main_chain = create_retrieval_chain(
        ai_models["compression_retriever"],
        prompt_chains["question_answer_chain"]
    )
    
    # Create self-query chain
    self_query_chain = ai_models["self_query_retriever"]
    
    logger.info("Application startup completed successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")

# Initialize FastAPI application
app = FastAPI(
    title=APP_CONFIG["name"],
    version=APP_CONFIG["version"],
    description=APP_CONFIG["description"],
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# MIDDLEWARE AND REQUEST PROCESSING
# =============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and their processing time."""
    start_time = time.time()
    
    # Log request details
    logger.info(f"Request: {request.method} {request.url}")
    logger.info(f"Client: {request.client.host if request.client else 'Unknown'}")
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Log response details
    logger.info(f"Response: {response.status_code} - {processing_time:.3f}s")
    
    # Add processing time to response headers
    response.headers["X-Processing-Time"] = str(processing_time)
    
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(f"Request: {request.method} {request.url}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if APP_CONFIG["debug"] else "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint providing basic application information.
    """
    return {
        "message": "Welcome to Yazd University Intelligent Assistant",
        "version": APP_CONFIG["version"],
        "status": "operational",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint for monitoring application status.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version=APP_CONFIG["version"],
        uptime=time.time() - app_state.start_time
    )

@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """
    Get application metrics and performance statistics.
    """
    return {
        "application_metrics": app_state.get_metrics(),
        "model_info": {
            "embeddings_model": MODEL_CONFIG["embeddings_model"],
            "llm_model": MODEL_CONFIG["llm_model"],
            "reranker_model": MODEL_CONFIG["reranker_model"],
            "metadata_fields": MODEL_CONFIG["metadata_fields"]
        },
        "database_info": {
            "path": DB_CONFIG["persist_directory"],
            "collection": DB_CONFIG["collection_name"]
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def process_query(request: QueryRequest):
    """
    پردازش پرسش کاربر و بازگرداندن پاسخ هوشمند.
    این endpoint همیشه از self-query retriever استفاده می‌کند.
    """
    start_time = time.time()
    try:
        logger.info(f"در حال پردازش پرسش: {request.query[:100]}...")
        logger.info(f"شناسه کاربر: {request.user_id}, شناسه نشست: {request.session_id}")
        # مرحله ۱: بازنویسی پرسش
        reformatted_query = prompt_chains["query_reformat_chain"].invoke({
            "user_query": request.query
        })
        reformatted_query_text = reformatted_query.content.strip()
        logger.info(f"پرسش اصلی: {request.query}")
        logger.info(f"پرسش بازنویسی‌شده: {reformatted_query_text}")
        # مرحله ۲: استفاده از self-query chain یا retriever معمولی
        if self_query_chain is not None:
            logger.info("استفاده از self-query retriever برای جستجوی ساختاریافته")
            try:
                context_docs = self_query_chain.invoke({"input": reformatted_query_text})
                retrieval_method = "self-query"
                structured_query = None
                # تلاش برای استخراج structured query (در صورت امکان)
                try:
                    structured_query = {
                        "method": "self-query",
                        "parsed_query": "پرسش ساختاریافته توسط self-query retriever تولید شده است."
                    }
                except Exception as e:
                    logger.warning(f"امکان استخراج structured query وجود ندارد: {e}")
            except Exception as e:
                logger.warning(f"خطا در self-query retriever: {e}")
                logger.info("استفاده از retriever معمولی به عنوان جایگزین")
                context_docs = ai_models["retriever"].invoke(reformatted_query_text)
                retrieval_method = "regular-retriever"
                structured_query = {
                    "method": "fallback",
                    "error": str(e)
                }
        else:
            logger.info("استفاده از retriever معمولی")
            context_docs = ai_models["retriever"].invoke(reformatted_query_text)
            retrieval_method = "regular-retriever"
            structured_query = {
                "method": "regular",
                "reason": "self-query retriever not available"
            }
        
        # مرحله ۳: تولید پاسخ با استفاده از LLM
        logger.info("تولید پاسخ با استفاده از LLM")
        context_texts = [doc.page_content for doc in context_docs]
        
        # Create a prompt for answer generation
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "پرسش: {query}\n\nبافتار: {context}")
        ])
        
        answer_chain = answer_prompt | ai_models["llm"]
        answer_response = answer_chain.invoke({
            "query": reformatted_query_text,
            "context": "\n\n".join(context_texts) if context_texts else "اطلاعاتی یافت نشد."
        })
        answer = answer_response.content
        processing_time = time.time() - start_time
        confidence_score = min(1.0, len(context_texts) / 5.0) if context_texts else 0.0
        use_self_query = (retrieval_method == "self-query") and (self_query_chain is not None)
        app_state.record_query(processing_time, success=True, use_self_query=use_self_query)
        logger.info(f"پرسش با موفقیت در {processing_time:.3f} ثانیه پردازش شد.")
        logger.info(f"روش بازیابی: {retrieval_method}")
        logger.info(f"تعداد منابع: {len(context_texts)}")
        logger.info(f"طول پاسخ: {len(answer)} کاراکتر")
        response_data = QueryResponse(
            answer=answer,
            context=context_texts,
            original_query=request.query,
            reformatted_query=reformatted_query_text,
            processing_time=processing_time,
            confidence_score=confidence_score,
            sources_count=len(context_texts),
            retrieval_method=retrieval_method,
            structured_query=structured_query
        )
        return response_data
    except Exception as e:
        processing_time = time.time() - start_time
        app_state.record_query(processing_time, success=False, use_self_query=False)
        logger.error(f"خطا در پردازش پرسش: {str(e)}")
        logger.error(f"زمان پردازش: {processing_time:.3f} ثانیه")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                "error": "خطا در پردازش پرسش",
                "message": str(e),
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            }
        )

class MetadataSearchRequest(BaseModel):
    """
    Pydantic model for metadata-based search requests.
    """
    field: str = Field(..., description="Metadata field to search in", example="نام")
    value: str = Field(..., description="Value to search for", example="احمدی")
    limit: Optional[int] = Field(5, description="Maximum number of results", example=5)

@app.post("/search/metadata", tags=["Search"])
async def search_by_metadata(request: MetadataSearchRequest):
    """
    جستجو بر اساس فیلدهای متادیتا.
    """
    try:
        logger.info(f"جستجو در فیلد {request.field} برای مقدار {request.value}")
        
        # Validate field name
        valid_fields = MODEL_CONFIG["metadata_fields"]
        if request.field not in valid_fields:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "فیلد نامعتبر",
                    "message": f"فیلد {request.field} معتبر نیست. فیلدهای معتبر: {valid_fields}",
                    "valid_fields": valid_fields
                }
            )
        
        # Get collection
        collection = ai_models["vectorstore"]._collection
        
        # Search by metadata
        results = collection.get(
            where={request.field: {"$contains": request.value}},
            limit=request.limit
        )
        
        # Format results
        formatted_results = []
        if results and 'metadatas' in results and 'documents' in results:
            for i, (metadata, document) in enumerate(zip(results['metadatas'], results['documents'])):
                formatted_results.append({
                    "id": i,
                    "metadata": metadata,
                    "content_preview": document[:200] + "..." if len(document) > 200 else document
                })
        
        return {
            "field": request.field,
            "search_value": request.value,
            "total_results": len(formatted_results),
            "results": formatted_results,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"خطا در جستجوی متادیتا: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "خطا در جستجوی متادیتا",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

# =============================================================================
# UTILITY ENDPOINTS
# =============================================================================

@app.get("/info", tags=["Information"])
async def get_system_info():
    """
    Get detailed system information and configuration.
    """
    return {
        "application": APP_CONFIG,
        "models": MODEL_CONFIG,
        "database": DB_CONFIG,
        "system_info": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/database/info", tags=["Database"])
async def get_database_info():
    """
    Get database information and metadata statistics.
    """
    try:
        # Get collection info
        collection = ai_models["vectorstore"]._collection
        count = collection.count()
        
        # Get sample documents to analyze metadata
        sample_docs = collection.get(limit=min(10, count))
        metadata_fields = set()
        if sample_docs and 'metadatas' in sample_docs:
            for metadata in sample_docs['metadatas']:
                if metadata:
                    metadata_fields.update(metadata.keys())
        
        return {
            "database_path": DB_CONFIG["persist_directory"],
            "collection_name": DB_CONFIG["collection_name"],
            "total_documents": count,
            "available_metadata_fields": list(metadata_fields),
            "sample_metadata": sample_docs['metadatas'][0] if sample_docs and 'metadatas' in sample_docs and sample_docs['metadatas'] else None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting database info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "خطا در دریافت اطلاعات پایگاه داده",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/database/metadata/fields", tags=["Database"])
async def get_metadata_fields():
    """
    Get all available metadata fields and their sample values.
    """
    try:
        collection = ai_models["vectorstore"]._collection
        count = collection.count()
        
        # Get all documents to analyze metadata
        all_docs = collection.get(limit=count)
        
        field_values = {}
        if all_docs and 'metadatas' in all_docs:
            for metadata in all_docs['metadatas']:
                if metadata:
                    for field, value in metadata.items():
                        if field not in field_values:
                            field_values[field] = set()
                        if value:
                            field_values[field].add(str(value))
        
        # Convert sets to lists and limit sample values
        field_samples = {}
        for field, values in field_values.items():
            field_samples[field] = {
                "total_unique_values": len(values),
                "sample_values": list(values)[:10]  # Show first 10 unique values
            }
        
        return {
            "total_documents": count,
            "metadata_fields": field_samples,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting metadata fields: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "خطا در دریافت فیلدهای متادیتا",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting application server...")
    logger.info(f"Host: {APP_CONFIG['host']}, Port: {APP_CONFIG['port']}")
    
    uvicorn.run(
        "backend:app",
        host=APP_CONFIG["host"],
        port=APP_CONFIG["port"],
        reload=APP_CONFIG["debug"],
        log_level="info"
    )