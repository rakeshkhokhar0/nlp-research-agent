"""Research paper chunk embedding architecture, model management, and inference.

This module provides deterministic, batched dense vector embedding generation for
research paper chunks using local SentenceTransformer models. It supports lazy model
loading, instance caching, configurable device placement, batching, and comprehensive
numerical and shape validation of output embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
import math
import numbers
import time
from typing import Any, Protocol, runtime_checkable

from app.modules.ingestion.chunker import ResearchChunk


SUPPORTED_PROVIDERS: frozenset[str] = frozenset({"sentence-transformers"})


class EmbeddingError(Exception):
    """Base exception for all embedding-related errors."""


class EmbeddingModelLoadError(EmbeddingError):
    """Raised when an embedding model cannot be loaded or initialized."""


class EmbeddingGenerationError(EmbeddingError):
    """Raised when an error occurs during embedding inference or vector generation."""


class EmbeddingValidationError(EmbeddingError, ValueError):
    """Raised when embedding configuration, input chunks, or vectors fail validation."""


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for embedding model loading and inference.

    Attributes:
        provider: Embedding provider backend. Default is 'sentence-transformers'.
        model_name: HuggingFace model identifier or local path.
        dimensions: Expected dimensionality of generated embedding vectors.
        normalize_embeddings: Whether to L2-normalize vectors to unit length.
        batch_size: Number of chunks encoded per model inference batch.
        device: Target execution device ('cpu', 'cuda', etc.).
        model_revision: Optional Git commit hash or branch revision of the model.
    """

    provider: str = "sentence-transformers"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimensions: int = 384
    normalize_embeddings: bool = True
    batch_size: int = 16
    device: str = "cpu"
    model_revision: str | None = None

    def __post_init__(self) -> None:
        """Validate embedding configuration parameters."""
        if self.provider not in SUPPORTED_PROVIDERS:
            raise EmbeddingValidationError(
                f"Unsupported provider '{self.provider}'. Supported providers: {sorted(SUPPORTED_PROVIDERS)}"
            )
        if not self.model_name or not self.model_name.strip():
            raise EmbeddingValidationError("model_name must not be empty.")
        if isinstance(self.dimensions, bool) or not isinstance(self.dimensions, int):
            raise EmbeddingValidationError(
                f"dimensions must be an integer, got {type(self.dimensions).__name__} ({self.dimensions!r})."
            )
        if self.dimensions <= 0:
            raise EmbeddingValidationError(
                f"dimensions must be an integer greater than zero, got {self.dimensions}."
            )
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise EmbeddingValidationError(
                f"batch_size must be an integer, got {type(self.batch_size).__name__} ({self.batch_size!r})."
            )
        if self.batch_size <= 0:
            raise EmbeddingValidationError(
                f"batch_size must be an integer greater than zero, got {self.batch_size}."
            )
        if not self.device or not self.device.strip():
            raise EmbeddingValidationError("device must not be empty.")
        if self.model_revision is not None and not self.model_revision.strip():
            raise EmbeddingValidationError(
                "model_revision must not be an empty string when provided."
            )


@dataclass(frozen=True)
class ChunkEmbedding:
    """An immutable dense vector embedding associated with a specific research chunk.

    Note:
        This dataclass is shallowly immutable: reassignment of its attributes
        (chunk_index, content_hash, vector) is prohibited by the frozen dataclass
        contract. However, because vector is typed and stored as a Python list[float]
        for interoperability with numerical and database libraries, callers must not
        mutate the underlying list elements in-place after construction.

    Attributes:
        chunk_index: 0-indexed sequential position of the source chunk in the paper.
        content_hash: SHA-256 digest of the chunk's text content.
        vector: Dense embedding vector as a list of floating-point numbers.
    """

    chunk_index: int
    content_hash: str
    vector: list[float]

    def __post_init__(self) -> None:
        """Validate chunk embedding data fields."""
        if self.chunk_index < 0:
            raise EmbeddingValidationError(
                f"chunk_index must be non-negative, got {self.chunk_index}."
            )
        if not self.content_hash or not self.content_hash.strip():
            raise EmbeddingValidationError("content_hash must not be empty.")
        if not isinstance(self.vector, list):
            raise EmbeddingValidationError(
                f"vector must be a list of floats, got {type(self.vector).__name__}."
            )


@dataclass
class EmbeddingDiagnostics:
    """Diagnostics and performance metrics for an embedding generation run.

    Attributes:
        input_chunks: Total number of chunks received for embedding.
        embedded_chunks: Total number of chunks successfully embedded.
        batch_count: Number of inference batches executed.
        elapsed_seconds: Total execution time in seconds.
        warnings: Quality, performance, or truncation warnings encountered.
    """

    input_chunks: int
    embedded_chunks: int
    batch_count: int
    elapsed_seconds: float
    warnings: list[str] = field(default_factory=list)


@dataclass
class EmbeddingResult:
    """Complete output of an embedding generation execution.

    Attributes:
        embeddings: List of ChunkEmbedding objects preserving source chunk ordering.
        model_name: Identifier of the model used for embedding.
        provider: Embedding provider backend name.
        dimensions: Dimensionality of the generated vectors.
        normalized: Whether the vectors were L2-normalized.
        diagnostics: Execution metrics and warnings.
    """

    embeddings: list[ChunkEmbedding]
    model_name: str
    provider: str
    dimensions: int
    normalized: bool
    diagnostics: EmbeddingDiagnostics


@runtime_checkable
class EmbeddingModelProtocol(Protocol):
    """Protocol defining the interface required from underlying embedding model backends.

    Allows the embedding engine to remain decoupled from specific ML libraries,
    supporting pluggable backends and lightweight test doubles.
    """

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int = 16,
        normalize_embeddings: bool = True,
        device: str | None = None,
        convert_to_numpy: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Encode a batch of sentences into numeric embedding representations.

        Args:
            sentences: List of text strings to encode.
            batch_size: Number of sentences per encoding batch.
            normalize_embeddings: Whether to L2-normalize vectors to unit length.
            device: Target execution device (e.g. 'cpu', 'cuda').
            convert_to_numpy: Whether to return embeddings as numpy arrays.
            **kwargs: Backend-specific keyword options.

        Returns:
            Sequence or array of numeric embedding vectors.
        """
        ...


class EmbeddingModelManager:
    """Manages the lifecycle, configuration, lazy loading, and caching of embedding models.

    Ensures that embedding models are loaded lazily upon first access,
    reused across embedding calls, and kept decoupled from high-level
    pipeline coordination.
    """

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        model: EmbeddingModelProtocol | None = None,
    ) -> None:
        """Initialize the model manager.

        Args:
            config: Embedding configuration specifying provider, model_name, etc.
            model: Optional pre-loaded model instance for dependency injection.
        """
        self.config = config if config is not None else EmbeddingConfig()
        self._model = model

    @property
    def is_loaded(self) -> bool:
        """Return True if an embedding model instance is currently loaded in memory."""
        return self._model is not None

    def get_model(self) -> EmbeddingModelProtocol:
        """Retrieve the loaded embedding model, loading it lazily if not already present.

        Returns:
            An object adhering to EmbeddingModelProtocol.

        Raises:
            EmbeddingModelLoadError: If importing the backend or initializing the model fails.
        """
        if self._model is not None:
            return self._model

        # Lazily import sentence_transformers
        try:
            module = import_module("sentence_transformers")
            sentence_transformer_cls = getattr(module, "SentenceTransformer")
        except (ImportError, ModuleNotFoundError) as exc:
            raise EmbeddingModelLoadError(
                "Failed to import 'sentence_transformers'. Please ensure 'sentence-transformers' "
                f"is installed in your environment: {exc}"
            ) from exc
        except Exception as exc:
            raise EmbeddingModelLoadError(
                f"Unexpected error loading 'sentence_transformers' module: {exc}"
            ) from exc

        model_kwargs: dict[str, Any] = {
            "device": self.config.device,
        }
        if self.config.model_revision is not None:
            model_kwargs["revision"] = self.config.model_revision

        try:
            model = sentence_transformer_cls(
                self.config.model_name,
                **model_kwargs,
            )
        except Exception as exc:
            raise EmbeddingModelLoadError(
                f"Failed to initialize SentenceTransformer model '{self.config.model_name}': {exc}"
            ) from exc

        self._model = model
        return self._model

    def unload_model(self) -> None:
        """Unload the cached model to free system resources."""
        self._model = None


class ResearchEmbeddingGenerator:
    """Generates dense vector embeddings for research paper chunks.

    Operates strictly on chunk.content, preserves chunk ordering, associates
    vectors with chunk_index and content_hash, and processes inputs in configurable batches.
    """

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        model_manager: EmbeddingModelManager | None = None,
    ) -> None:
        """Initialize the embedding generator.

        Args:
            config: Embedding configuration parameters.
            model_manager: Optional EmbeddingModelManager instance for dependency injection.

        Raises:
            EmbeddingValidationError: If both config and model_manager are provided
                and their configurations do not match.
        """
        if config is None and model_manager is not None:
            self.config = model_manager.config
            self.model_manager = model_manager
        elif config is not None and model_manager is not None:
            if config != model_manager.config:
                raise EmbeddingValidationError(
                    "Configuration mismatch between ResearchEmbeddingGenerator and injected EmbeddingModelManager: "
                    f"generator config={config}, model_manager config={model_manager.config}"
                )
            self.config = config
            self.model_manager = model_manager
        elif config is not None and model_manager is None:
            self.config = config
            self.model_manager = EmbeddingModelManager(config=config)
        else:
            self.config = EmbeddingConfig()
            self.model_manager = EmbeddingModelManager(config=self.config)

    def embed_chunks(self, chunks: list[ResearchChunk]) -> EmbeddingResult:
        """Generate vector embeddings for a list of research paper chunks.

        Embeds only chunk.content, preserves chunk order, and associates each
        resulting vector with chunk_index and content_hash.

        Args:
            chunks: List of ResearchChunk objects from the chunking engine.

        Returns:
            EmbeddingResult containing ChunkEmbedding objects and diagnostics.

        Raises:
            EmbeddingValidationError: If chunks list fails validation or contains
                empty/whitespace-only chunk content.
            EmbeddingGenerationError: If embedding generation fails during inference
                or vector validation.
        """
        if not isinstance(chunks, list):
            raise EmbeddingValidationError(
                f"Expected chunks to be a list, got {type(chunks).__name__}"
            )
        for i, c in enumerate(chunks):
            if not isinstance(c, ResearchChunk):
                raise EmbeddingValidationError(
                    f"Element at index {i} is not a ResearchChunk, got {type(c).__name__}"
                )
            if not c.content or not c.content.strip():
                raise EmbeddingValidationError(
                    f"Chunk at index {i} has empty or whitespace-only content."
                )

        start_time = time.perf_counter()

        if not chunks:
            elapsed = time.perf_counter() - start_time
            return EmbeddingResult(
                embeddings=[],
                model_name=self.config.model_name,
                provider=self.config.provider,
                dimensions=self.config.dimensions,
                normalized=self.config.normalize_embeddings,
                diagnostics=EmbeddingDiagnostics(
                    input_chunks=0,
                    embedded_chunks=0,
                    batch_count=0,
                    elapsed_seconds=elapsed,
                    warnings=[],
                ),
            )

        model = self.model_manager.get_model()

        all_embeddings: list[ChunkEmbedding] = []
        batch_count = 0

        for batch_start in range(0, len(chunks), self.config.batch_size):
            batch = chunks[batch_start : batch_start + self.config.batch_size]
            batch_texts = [c.content for c in batch]

            try:
                raw_vectors = model.encode(
                    batch_texts,
                    batch_size=len(batch_texts),
                    normalize_embeddings=self.config.normalize_embeddings,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            except EmbeddingError:
                raise
            except Exception as exc:
                raise EmbeddingGenerationError(
                    f"Model inference failed for batch {batch_count}: {exc}"
                ) from exc

            converted_vectors = self._validate_and_convert_vectors(
                raw_vectors, batch, batch_count
            )

            for c, vec in zip(batch, converted_vectors):
                all_embeddings.append(
                    ChunkEmbedding(
                        chunk_index=c.chunk_index,
                        content_hash=c.content_hash,
                        vector=vec,
                    )
                )

            batch_count += 1

        elapsed_seconds = time.perf_counter() - start_time

        return EmbeddingResult(
            embeddings=all_embeddings,
            model_name=self.config.model_name,
            provider=self.config.provider,
            dimensions=self.config.dimensions,
            normalized=self.config.normalize_embeddings,
            diagnostics=EmbeddingDiagnostics(
                input_chunks=len(chunks),
                embedded_chunks=len(all_embeddings),
                batch_count=batch_count,
                elapsed_seconds=elapsed_seconds,
                warnings=[],
            ),
        )

    def _validate_and_convert_vectors(
        self,
        raw_vectors: Any,
        batch_chunks: list[ResearchChunk],
        batch_index: int,
    ) -> list[list[float]]:
        """Validate raw model output and convert vectors to native Python float lists.

        Args:
            raw_vectors: Output produced by the model's encode() method.
            batch_chunks: List of ResearchChunk objects in the current batch.
            batch_index: 0-indexed position of the batch for error diagnostics.

        Returns:
            List of 1D float vectors corresponding to the input chunks.

        Raises:
            EmbeddingGenerationError: If the model output fails structure, shape,
                or numeric validation.
        """
        try:
            n_vectors = len(raw_vectors)
        except TypeError as exc:
            raise EmbeddingGenerationError(
                f"Model output for batch {batch_index} is not a sized sequence: {exc}"
            ) from exc

        if n_vectors != len(batch_chunks):
            raise EmbeddingGenerationError(
                f"Batch {batch_index} output count mismatch: "
                f"expected {len(batch_chunks)} vectors, got {n_vectors}"
            )

        converted_vectors: list[list[float]] = []

        for vec_idx, raw_vec in enumerate(raw_vectors):
            if isinstance(raw_vec, (str, bytes)) or not hasattr(raw_vec, "__len__"):
                raise EmbeddingGenerationError(
                    f"Batch {batch_index}, vector {vec_idx} is not a sequence, expected 1D numeric vector."
                )

            if hasattr(raw_vec, "ndim") and raw_vec.ndim != 1:
                raise EmbeddingGenerationError(
                    f"Batch {batch_index}, vector {vec_idx} has {raw_vec.ndim} dimensions, expected 1D vector."
                )

            if isinstance(raw_vec, (list, tuple)) and any(
                isinstance(x, (list, tuple, dict)) for x in raw_vec
            ):
                raise EmbeddingGenerationError(
                    f"Batch {batch_index}, vector {vec_idx} is nested, expected 1D numeric vector."
                )

            vec_len = len(raw_vec)
            if vec_len != self.config.dimensions:
                raise EmbeddingGenerationError(
                    f"Batch {batch_index}, vector {vec_idx} dimension mismatch: "
                    f"expected {self.config.dimensions}, got {vec_len}."
                )

            float_vec: list[float] = []
            for val_idx, val in enumerate(raw_vec):
                # Reject booleans (bool is a subclass of int in Python)
                if (
                    isinstance(val, bool)
                    or type(val).__name__ in ("bool", "bool_")
                    or (hasattr(val, "dtype") and str(val.dtype).startswith("bool"))
                ):
                    raise EmbeddingGenerationError(
                        f"Batch {batch_index}, vector {vec_idx}, value {val_idx} is boolean ({val!r}), "
                        "expected numeric float."
                    )

                # Validate genuine real numeric scalar (rejects strings, bytes, complex, arbitrary __float__ objects)
                if not isinstance(val, numbers.Real):
                    raise EmbeddingGenerationError(
                        f"Batch {batch_index}, vector {vec_idx}, value {val_idx} is not a valid real numeric value "
                        f"({type(val).__name__}: {val!r})."
                    )

                try:
                    f_val = float(val)
                except (ValueError, TypeError) as exc:
                    raise EmbeddingGenerationError(
                        f"Batch {batch_index}, vector {vec_idx}, value {val_idx} cannot be converted to float "
                        f"({val!r}): {exc}"
                    ) from exc

                if not math.isfinite(f_val):
                    raise EmbeddingGenerationError(
                        f"Batch {batch_index}, vector {vec_idx}, value {val_idx} is non-finite ({f_val})."
                    )

                float_vec.append(f_val)

            converted_vectors.append(float_vec)

        return converted_vectors
