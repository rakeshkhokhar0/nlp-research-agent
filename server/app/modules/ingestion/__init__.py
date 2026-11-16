from app.modules.ingestion.chunker import (
    CharacterTokenCounter,
    ChunkingConfig,
    ChunkingDiagnostics,
    ChunkingResult,
    HuggingFaceTokenCounter,
    ResearchChunk,
    ResearchChunkingEngine,
    TokenCounter,
    WhitespaceTokenCounter,
)
from app.modules.ingestion.embedder import (
    SUPPORTED_PROVIDERS,
    ChunkEmbedding,
    EmbeddingConfig,
    EmbeddingDiagnostics,
    EmbeddingError,
    EmbeddingGenerationError,
    EmbeddingModelLoadError,
    EmbeddingModelManager,
    EmbeddingModelProtocol,
    EmbeddingResult,
    EmbeddingValidationError,
    ResearchEmbeddingGenerator,
)
from app.modules.ingestion.models import (
    ACTIVE_JOB_STATUSES,
    SUPPORTED_JOB_TYPES,
    TERMINAL_JOB_STATUSES,
    IngestionJob,
    PaperChunk,
)
from app.modules.ingestion.parser import (
    AcademicMetadata,
    AuthorCandidate,
    ParsedPage,
    ParsedResearchPaper,
    ParsingDiagnostics,
    PDFMetadata,
    PDFParser,
    ResearchMetadataExtractor,
)
from app.modules.ingestion.preprocessor import (
    PreprocessedPage,
    PreprocessedResearchPaper,
    PreprocessingDiagnostics,
    ResearchTextPreprocessor,
)
from app.modules.ingestion.section_detector import (
    DetectedSection,
    ResearchSectionDetector,
    SectionDetectionDiagnostics,
    SectionDetectionResult,
    SectionParagraph,
    SectionType,
)
from app.modules.ingestion.repository import (
    ChunkActivationError,
    ChunkPersistenceError,
    EmbeddingModelConfigurationError,
    EmbeddingPersistenceError,
    IngestionRepository,
    IngestionRepositoryError,
    JobNotFoundError,
)
from app.modules.ingestion.storage import (
    CloudinaryStorage,
    CloudinaryUploadResult,
)

__all__ = [
    "PaperChunk",
    "IngestionJob",
    "ACTIVE_JOB_STATUSES",
    "TERMINAL_JOB_STATUSES",
    "SUPPORTED_JOB_TYPES",
    "CloudinaryStorage",
    "CloudinaryUploadResult",
    "PDFParser",
    "ResearchMetadataExtractor",
    "PDFMetadata",
    "AcademicMetadata",
    "AuthorCandidate",
    "ParsedPage",
    "ParsingDiagnostics",
    "ParsedResearchPaper",
    "PreprocessedPage",
    "PreprocessingDiagnostics",
    "PreprocessedResearchPaper",
    "ResearchTextPreprocessor",
    "SectionType",
    "SectionParagraph",
    "DetectedSection",
    "SectionDetectionDiagnostics",
    "SectionDetectionResult",
    "ResearchSectionDetector",
    "TokenCounter",
    "WhitespaceTokenCounter",
    "CharacterTokenCounter",
    "HuggingFaceTokenCounter",
    "ChunkingConfig",
    "ResearchChunk",
    "ChunkingDiagnostics",
    "ChunkingResult",
    "ResearchChunkingEngine",
    "SUPPORTED_PROVIDERS",
    "EmbeddingError",
    "EmbeddingModelLoadError",
    "EmbeddingGenerationError",
    "EmbeddingValidationError",
    "EmbeddingConfig",
    "ChunkEmbedding",
    "EmbeddingDiagnostics",
    "EmbeddingResult",
    "EmbeddingModelProtocol",
    "EmbeddingModelManager",
    "ResearchEmbeddingGenerator",
    "IngestionRepository",
    "IngestionRepositoryError",
    "JobNotFoundError",
    "EmbeddingModelConfigurationError",
    "ChunkPersistenceError",
    "EmbeddingPersistenceError",
    "ChunkActivationError",
    "check_active_jobs_for_deletion",
    "IngestionResult",
    "IngestionService",
    "router",
    "get_ingestion_service",
    "IngestionUploadResponse",
    "IngestionJobResponse",
]


def __getattr__(name: str) -> object:
    if name in {"IngestionService", "IngestionResult", "check_active_jobs_for_deletion"}:
        import app.modules.ingestion.service as _service
        return getattr(_service, name)
    if name in {"router", "get_ingestion_service"}:
        import app.modules.ingestion.router as _router
        return getattr(_router, name)
    if name in {"IngestionUploadResponse", "IngestionJobResponse"}:
        import app.modules.ingestion.schemas as _schemas
        return getattr(_schemas, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

