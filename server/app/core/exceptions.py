"""Domain exceptions for AI Research Intelligence Agent."""


class AppError(Exception):
    """Base exception for application domain errors."""

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__


class DatabaseError(AppError):
    """Base exception for database-related operations."""


class EntityNotFoundError(DatabaseError):
    """Raised when an entity is not found."""


class DuplicateEntityError(DatabaseError):
    """Raised when a unique constraint or deduplication rule is violated."""


class ActiveJobConflictError(DatabaseError):
    """Raised when an operation (e.g. file deletion) conflicts with an active ingestion job."""


class StorageError(AppError):
    """Raised when a cloud or local storage operation fails."""


class InvalidPDFError(StorageError):
    """Raised when file content fails PDF validation."""


class StorageUploadError(StorageError):
    """Raised when file upload fails."""


class StorageDownloadError(StorageError):
    """Raised when file download/retrieval fails."""


class StorageDeleteError(StorageError):
    """Raised when file deletion fails."""


class StorageFileNotFoundError(StorageError):
    """Raised when a requested resource is not found."""


class PDFParsingError(AppError):
    """Raised when PDF document parsing or extraction fails."""


class PDFEncryptedError(PDFParsingError):
    """Raised when a PDF is password-protected or encrypted and cannot be parsed."""


class PDFNoExtractableTextError(PDFParsingError):
    """Raised when a PDF contains pages but zero extractable text across all pages."""


class CitationIntegrityError(DatabaseError):
    """Raised when citation consistency is violated."""


class EmbeddingDimensionMismatchError(DatabaseError):
    """Raised when vector embedding dimensions do not match model specifications."""
