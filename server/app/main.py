from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException
from sqlalchemy.ext.asyncio import async_sessionmaker

from fastapi.middleware.cors import CORSMiddleware

from app.core.config import cloudinary_settings, configure_cloudinary
from app.core.logging_config import setup_logging
from app.database.health import check_database
from app.database.session import AsyncSessionLocal, engine
from app.modules.conversation.llm_client import LLMConfigurationError, OpenAIResponseClient
from app.modules.conversation.router import router as conversation_router
from app.modules.conversation.service import ConversationService
from app.modules.ingestion.router import router as ingestion_router
from app.modules.ingestion.service import IngestionService
from app.modules.papers.router import router as papers_router
from app.modules.retrieval import RetrievalRepository, RetrievalService

# Initialize shared structured logging
setup_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Configure Cloudinary SDK
    configure_cloudinary(cloudinary_settings)

    # Initialize the application-scoped retrieval service with a fresh session
    retrieval_service = RetrievalService()
    async with AsyncSessionLocal() as session:
        repository = RetrievalRepository(session)
        await retrieval_service.refresh_index(repository)

    app.state.retrieval_service = retrieval_service

    if isinstance(AsyncSessionLocal, async_sessionmaker):
        app.state.ingestion_service = IngestionService(
            session_factory=AsyncSessionLocal,
            retrieval_service=retrieval_service,
        )

    # Initialize OpenAI client and application-scoped ConversationService
    app.state.openai_client = None
    app.state.conversation_service = None

    try:
        openai_client = OpenAIResponseClient()
        app.state.openai_client = openai_client

        if isinstance(AsyncSessionLocal, async_sessionmaker):
            app.state.conversation_service = ConversationService(
                session_factory=AsyncSessionLocal,
                retrieval_service=retrieval_service,
                llm_client=openai_client,
            )
    except LLMConfigurationError:
        logger.warning(
            "OpenAI API key is missing or not configured. Conversation service will remain unavailable."
        )
        app.state.openai_client = None
        app.state.conversation_service = None

    yield

    # Shutdown: Clean up resources
    if hasattr(app.state, "openai_client") and app.state.openai_client is not None:
        await app.state.openai_client.close()

    await engine.dispose()


app = FastAPI(
    title="AI Research Intelligence Agent",
    description="Research assistant for AI, ML, and NLP papers.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Register routers
app.include_router(ingestion_router, prefix="/api/v1")
app.include_router(papers_router, prefix="/api/v1")
app.include_router(conversation_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "message": "AI Research Intelligence Agent API",
        "status": "running",
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/health/database")
async def database_health_check():
    try:
        is_connected = await check_database(engine)

        if not is_connected:
            raise HTTPException(
                status_code=503,
                detail="Database health check failed",
            )

        return {"database": "connected"}

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable",
        )
