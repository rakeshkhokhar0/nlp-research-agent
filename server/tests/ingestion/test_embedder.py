"""Unit tests for embedding generation architecture, configuration, model management, and inference.

Stage 2 tests: Validates lazy model loading, caching, device/revision configuration,
batched inference, chunk-to-vector association, output validation (shape, booleans,
NaN/inf, non-numeric), error handling, and performance diagnostics using mock models.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
import subprocess
import sys
from typing import Any
import pytest

from app.modules.ingestion.chunker import ResearchChunk
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


# -------------------------------------------------------------------------
# Test Doubles & Fixtures
# -------------------------------------------------------------------------

class MockEmbeddingModel:
    """Mock implementation adhering to EmbeddingModelProtocol for unit testing."""

    def __init__(self, dimensions: int = 384, return_func: Any = None) -> None:
        self.dimensions = dimensions
        self.return_func = return_func
        self.call_history: list[dict[str, Any]] = []

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int = 16,
        normalize_embeddings: bool = True,
        device: str | None = None,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        **kwargs: Any,
    ) -> Any:
        call_info = {
            "sentences": sentences,
            "batch_size": batch_size,
            "normalize_embeddings": normalize_embeddings,
            "device": device,
            "convert_to_numpy": convert_to_numpy,
            "show_progress_bar": show_progress_bar,
            "kwargs": kwargs,
        }
        self.call_history.append(call_info)
        if self.return_func is not None:
            return self.return_func(sentences)
        return [[0.1] * self.dimensions for _ in sentences]


def make_dummy_chunk(
    chunk_index: int = 0,
    content: str = "Test chunk content.",
    content_hash: str | None = None,
) -> ResearchChunk:
    """Helper to create a valid ResearchChunk for testing."""
    return ResearchChunk(
        chunk_index=chunk_index,
        content=content,
        token_count=len(content.split()),
        char_count=len(content),
        page_start=1,
        page_end=1,
        section_title="1. Introduction",
        content_hash=content_hash if content_hash is not None else f"hash_{chunk_index}",
        metadata={"section_id": "sec_1"},
    )


# -------------------------------------------------------------------------
# Test Suite: Configuration & Data Structures (Stage 1 Regression)
# -------------------------------------------------------------------------

class TestEmbeddingConfiguration:
    """Tests for EmbeddingConfig fields, defaults, and validation."""

    def test_default_configuration_values(self) -> None:
        """EmbeddingConfig instantiates with correct approved default values."""
        cfg = EmbeddingConfig()
        assert cfg.provider == "sentence-transformers"
        assert cfg.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert cfg.dimensions == 384
        assert cfg.normalize_embeddings is True
        assert cfg.batch_size == 16
        assert cfg.device == "cpu"
        assert cfg.model_revision is None

    def test_custom_configuration_values(self) -> None:
        """EmbeddingConfig accepts valid custom parameters."""
        cfg = EmbeddingConfig(
            provider="sentence-transformers",
            model_name="BAAI/bge-small-en-v1.5",
            dimensions=384,
            normalize_embeddings=False,
            batch_size=32,
            device="cuda",
            model_revision="main",
        )
        assert cfg.provider == "sentence-transformers"
        assert cfg.model_name == "BAAI/bge-small-en-v1.5"
        assert cfg.dimensions == 384
        assert cfg.normalize_embeddings is False
        assert cfg.batch_size == 32
        assert cfg.device == "cuda"
        assert cfg.model_revision == "main"

    @pytest.mark.parametrize("invalid_dim", [0, -1, -384, 384.5, True, False])
    def test_invalid_dimensions(self, invalid_dim: Any) -> None:
        """EmbeddingConfig rejects non-integer, boolean, zero, or negative dimensions."""
        with pytest.raises(EmbeddingValidationError, match="dimensions must be an integer"):
            EmbeddingConfig(dimensions=invalid_dim)

    @pytest.mark.parametrize("invalid_batch", [0, -1, -16, 16.5, True, False])
    def test_invalid_batch_size(self, invalid_batch: Any) -> None:
        """EmbeddingConfig rejects non-integer, boolean, zero, or negative batch sizes."""
        with pytest.raises(EmbeddingValidationError, match="batch_size must be an integer"):
            EmbeddingConfig(batch_size=invalid_batch)

    @pytest.mark.parametrize("invalid_model", ["", "   ", "\t\n"])
    def test_invalid_model_name(self, invalid_model: str) -> None:
        """EmbeddingConfig rejects empty or whitespace-only model names."""
        with pytest.raises(EmbeddingValidationError, match="model_name must not be empty"):
            EmbeddingConfig(model_name=invalid_model)

    @pytest.mark.parametrize("invalid_provider", ["openai", "cohere", "", "unknown-provider"])
    def test_invalid_provider(self, invalid_provider: str) -> None:
        """EmbeddingConfig rejects unsupported providers."""
        with pytest.raises(EmbeddingValidationError, match="Unsupported provider"):
            EmbeddingConfig(provider=invalid_provider)

    @pytest.mark.parametrize("invalid_device", ["", "   ", "\t"])
    def test_invalid_device(self, invalid_device: str) -> None:
        """EmbeddingConfig rejects empty device strings."""
        with pytest.raises(EmbeddingValidationError, match="device must not be empty"):
            EmbeddingConfig(device=invalid_device)

    @pytest.mark.parametrize("invalid_rev", ["", "   ", "\t"])
    def test_empty_model_revision(self, invalid_rev: str) -> None:
        """EmbeddingConfig rejects empty model revision when provided."""
        with pytest.raises(EmbeddingValidationError, match="model_revision must not be an empty string"):
            EmbeddingConfig(model_revision=invalid_rev)

    def test_config_is_immutable(self) -> None:
        """EmbeddingConfig is a frozen dataclass and prevents field mutations."""
        cfg = EmbeddingConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.dimensions = 512  # type: ignore[misc]


class TestEmbeddingDataStructures:
    """Tests for ChunkEmbedding, EmbeddingDiagnostics, and EmbeddingResult."""

    def test_chunk_embedding_construction(self) -> None:
        """ChunkEmbedding stores vector and metadata shallowly immutably."""
        vec = [0.123, -0.456, 0.789]
        emb = ChunkEmbedding(chunk_index=0, content_hash="hash_abc", vector=vec)
        assert emb.chunk_index == 0
        assert emb.content_hash == "hash_abc"
        assert emb.vector == vec

        with pytest.raises(FrozenInstanceError):
            emb.chunk_index = 1  # type: ignore[misc]

    def test_chunk_embedding_validation(self) -> None:
        """ChunkEmbedding validates indices, content hash, and vector structure."""
        with pytest.raises(EmbeddingValidationError, match="chunk_index must be non-negative"):
            ChunkEmbedding(chunk_index=-1, content_hash="hash", vector=[0.1])

        with pytest.raises(EmbeddingValidationError, match="content_hash must not be empty"):
            ChunkEmbedding(chunk_index=0, content_hash="", vector=[0.1])

        with pytest.raises(EmbeddingValidationError, match="vector must be a list of floats"):
            ChunkEmbedding(chunk_index=0, content_hash="hash", vector="not_a_list")  # type: ignore[arg-type]

    def test_embedding_diagnostics_construction(self) -> None:
        """EmbeddingDiagnostics accurately records execution metrics."""
        diag = EmbeddingDiagnostics(
            input_chunks=20,
            embedded_chunks=20,
            batch_count=2,
            elapsed_seconds=0.42,
            warnings=["Warning test"],
        )
        assert diag.input_chunks == 20
        assert diag.embedded_chunks == 20
        assert diag.batch_count == 2
        assert diag.elapsed_seconds == 0.42
        assert diag.warnings == ["Warning test"]

        diag_default = EmbeddingDiagnostics(
            input_chunks=5, embedded_chunks=5, batch_count=1, elapsed_seconds=0.1
        )
        assert diag_default.warnings == []

    def test_embedding_result_construction(self) -> None:
        """EmbeddingResult encapsulates embeddings, model metadata, and diagnostics."""
        emb1 = ChunkEmbedding(chunk_index=0, content_hash="hash1", vector=[0.1, 0.2])
        emb2 = ChunkEmbedding(chunk_index=1, content_hash="hash2", vector=[0.3, 0.4])
        diag = EmbeddingDiagnostics(
            input_chunks=2, embedded_chunks=2, batch_count=1, elapsed_seconds=0.05
        )

        result = EmbeddingResult(
            embeddings=[emb1, emb2],
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            provider="sentence-transformers",
            dimensions=2,
            normalized=True,
            diagnostics=diag,
        )

        assert len(result.embeddings) == 2
        assert result.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert result.provider == "sentence-transformers"
        assert result.dimensions == 2
        assert result.normalized is True
        assert result.diagnostics.embedded_chunks == 2


class TestEmbeddingExceptions:
    """Tests for exception hierarchy and error propagation."""

    def test_exception_inheritance(self) -> None:
        """All specialized exceptions inherit from EmbeddingError."""
        assert issubclass(EmbeddingModelLoadError, EmbeddingError)
        assert issubclass(EmbeddingGenerationError, EmbeddingError)
        assert issubclass(EmbeddingValidationError, EmbeddingError)
        assert issubclass(EmbeddingValidationError, ValueError)
        assert issubclass(EmbeddingError, Exception)

    def test_exception_handling_polymorphism(self) -> None:
        """EmbeddingValidationError is catchable as both EmbeddingError and ValueError."""
        with pytest.raises(EmbeddingError):
            raise EmbeddingValidationError("Validation failed")

        with pytest.raises(ValueError):
            raise EmbeddingValidationError("Validation failed")


# -------------------------------------------------------------------------
# Test Suite: EmbeddingModelManager (Stage 2 Lazy Loading & Caching)
# -------------------------------------------------------------------------

class TestEmbeddingModelManager:
    """Tests for lazy model loading, caching, device/revision configuration, and error handling."""

    def test_protocol_compatibility_with_mock(self) -> None:
        """A mock adhering to EmbeddingModelProtocol satisfies isinstance checks."""
        mock_model = MockEmbeddingModel(dimensions=384)
        assert isinstance(mock_model, EmbeddingModelProtocol)

        class IncompleteModel:
            pass

        assert not isinstance(IncompleteModel(), EmbeddingModelProtocol)

    def test_injected_models_bypass_loading(self) -> None:
        """Injecting a model directly bypasses sentence_transformers imports."""
        mock_model = MockEmbeddingModel(dimensions=384)
        manager = EmbeddingModelManager(model=mock_model)
        assert manager.is_loaded is True
        assert manager.get_model() is mock_model

    def test_lazy_loading_on_first_get_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Model is loaded lazily only when get_model() is first called."""
        init_calls: list[dict[str, Any]] = []

        class FakeSentenceTransformer:
            def __init__(self, model_name: str, **kwargs: Any) -> None:
                init_calls.append({"model_name": model_name, "kwargs": kwargs})

            def encode(self, sentences: list[str], **kwargs: Any) -> list[list[float]]:
                return [[0.1] * 384 for _ in sentences]

        fake_module = type("FakeModule", (), {"SentenceTransformer": FakeSentenceTransformer})()
        monkeypatch.setattr(
            "app.modules.ingestion.embedder.import_module",
            lambda name: fake_module if name == "sentence_transformers" else sys.modules[name],
        )

        cfg = EmbeddingConfig(device="cpu")
        manager = EmbeddingModelManager(config=cfg)
        assert manager.is_loaded is False
        assert len(init_calls) == 0

        # First call loads the model
        model = manager.get_model()
        assert manager.is_loaded is True
        assert len(init_calls) == 1
        assert init_calls[0]["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert init_calls[0]["kwargs"]["device"] == "cpu"
        assert "revision" not in init_calls[0]["kwargs"]
        assert isinstance(model, FakeSentenceTransformer)

    def test_repeated_get_model_calls_return_cached_instance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Subsequent get_model() calls return the identical cached instance without reloading."""
        init_calls: list[dict[str, Any]] = []

        class FakeSentenceTransformer:
            def __init__(self, model_name: str, **kwargs: Any) -> None:
                init_calls.append({"model_name": model_name, "kwargs": kwargs})

        fake_module = type("FakeModule", (), {"SentenceTransformer": FakeSentenceTransformer})()
        monkeypatch.setattr(
            "app.modules.ingestion.embedder.import_module",
            lambda name: fake_module if name == "sentence_transformers" else sys.modules[name],
        )

        manager = EmbeddingModelManager()
        model1 = manager.get_model()
        model2 = manager.get_model()
        assert model1 is model2
        assert len(init_calls) == 1

    def test_unload_model_clears_cached_instance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """unload_model() clears the cached model and resets is_loaded."""
        class FakeSentenceTransformer:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

        fake_module = type("FakeModule", (), {"SentenceTransformer": FakeSentenceTransformer})()
        monkeypatch.setattr(
            "app.modules.ingestion.embedder.import_module",
            lambda name: fake_module if name == "sentence_transformers" else sys.modules[name],
        )

        manager = EmbeddingModelManager()
        manager.get_model()
        assert manager.is_loaded is True

        manager.unload_model()
        assert manager.is_loaded is False
        assert manager._model is None

    def test_get_model_after_unload_loads_new_instance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Calling get_model() after unloading initializes a new model instance."""
        init_count = 0

        class FakeSentenceTransformer:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                nonlocal init_count
                init_count += 1

        fake_module = type("FakeModule", (), {"SentenceTransformer": FakeSentenceTransformer})()
        monkeypatch.setattr(
            "app.modules.ingestion.embedder.import_module",
            lambda name: fake_module if name == "sentence_transformers" else sys.modules[name],
        )

        manager = EmbeddingModelManager()
        m1 = manager.get_model()
        assert init_count == 1

        manager.unload_model()
        m2 = manager.get_model()
        assert init_count == 2
        assert m1 is not m2

    def test_configured_device_passed_to_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The configured device string is passed directly to the model constructor."""
        captured_kwargs: dict[str, Any] = {}

        class FakeSentenceTransformer:
            def __init__(self, model_name: str, **kwargs: Any) -> None:
                captured_kwargs.update(kwargs)

        fake_module = type("FakeModule", (), {"SentenceTransformer": FakeSentenceTransformer})()
        monkeypatch.setattr(
            "app.modules.ingestion.embedder.import_module",
            lambda name: fake_module if name == "sentence_transformers" else sys.modules[name],
        )

        cfg = EmbeddingConfig(device="cuda:1")
        manager = EmbeddingModelManager(config=cfg)
        manager.get_model()
        assert captured_kwargs["device"] == "cuda:1"

    def test_configured_revision_passed_only_when_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Model revision is included in constructor kwargs only when non-None."""
        captured_kwargs: list[dict[str, Any]] = []

        class FakeSentenceTransformer:
            def __init__(self, model_name: str, **kwargs: Any) -> None:
                captured_kwargs.append(kwargs)

        fake_module = type("FakeModule", (), {"SentenceTransformer": FakeSentenceTransformer})()
        monkeypatch.setattr(
            "app.modules.ingestion.embedder.import_module",
            lambda name: fake_module if name == "sentence_transformers" else sys.modules[name],
        )

        # Revision is None: kwargs should NOT have 'revision'
        manager_no_rev = EmbeddingModelManager(config=EmbeddingConfig(model_revision=None))
        manager_no_rev.get_model()
        assert "revision" not in captured_kwargs[0]

        # Revision is present: kwargs MUST have 'revision'
        manager_with_rev = EmbeddingModelManager(config=EmbeddingConfig(model_revision="v1.0.0"))
        manager_with_rev.get_model()
        assert captured_kwargs[1]["revision"] == "v1.0.0"

    def test_missing_dependency_raises_load_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ModuleNotFoundError when importing sentence_transformers raises EmbeddingModelLoadError."""
        def mock_import(name: str) -> Any:
            if name == "sentence_transformers":
                raise ModuleNotFoundError("No module named 'sentence_transformers'")
            return sys.modules[name]

        monkeypatch.setattr("app.modules.ingestion.embedder.import_module", mock_import)

        manager = EmbeddingModelManager()
        with pytest.raises(EmbeddingModelLoadError, match="Failed to import 'sentence_transformers'") as exc_info:
            manager.get_model()

        assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)
        assert manager.is_loaded is False

    def test_initialization_error_raises_load_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RuntimeError during model instantiation raises EmbeddingModelLoadError with cause."""
        class FailingSentenceTransformer:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("Corrupted model checkpoint")

        fake_module = type("FakeModule", (), {"SentenceTransformer": FailingSentenceTransformer})()
        monkeypatch.setattr(
            "app.modules.ingestion.embedder.import_module",
            lambda name: fake_module if name == "sentence_transformers" else sys.modules[name],
        )

        manager = EmbeddingModelManager()
        with pytest.raises(EmbeddingModelLoadError, match="Failed to initialize SentenceTransformer model") as exc_info:
            manager.get_model()

        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert manager.is_loaded is False
        assert manager._model is None

    def test_failed_initialization_does_not_cache_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Failed initialization leaves the manager in an unloaded state."""
        class FailingSentenceTransformer:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise ValueError("Bad model name")

        fake_module = type("FakeModule", (), {"SentenceTransformer": FailingSentenceTransformer})()
        monkeypatch.setattr(
            "app.modules.ingestion.embedder.import_module",
            lambda name: fake_module if name == "sentence_transformers" else sys.modules[name],
        )

        manager = EmbeddingModelManager()
        with pytest.raises(EmbeddingModelLoadError):
            manager.get_model()

        assert manager.is_loaded is False
        assert manager._model is None


# -------------------------------------------------------------------------
# Test Suite: ResearchEmbeddingGenerator (Stage 2 Inference & Validation)
# -------------------------------------------------------------------------

class TestResearchEmbeddingGenerator:
    """Tests for batch embedding generation, vector validation, association, and error handling."""

    def test_generator_and_manager_config_consistency(self) -> None:
        """ResearchEmbeddingGenerator enforces config consistency with its model manager."""
        custom_cfg = EmbeddingConfig(batch_size=32, dimensions=384)
        mismatched_cfg = EmbeddingConfig(batch_size=64, dimensions=384)
        mock_model = MockEmbeddingModel()

        # Case 1: No config provided, model_manager injected -> adopts manager's config
        manager1 = EmbeddingModelManager(config=custom_cfg, model=mock_model)
        gen1 = ResearchEmbeddingGenerator(config=None, model_manager=manager1)
        assert gen1.config is custom_cfg
        assert gen1.model_manager is manager1

        # Case 2: Matching config and model_manager provided -> succeeds
        manager2 = EmbeddingModelManager(config=custom_cfg, model=mock_model)
        gen2 = ResearchEmbeddingGenerator(config=custom_cfg, model_manager=manager2)
        assert gen2.config is custom_cfg
        assert gen2.model_manager is manager2

        # Case 3: Differing config and model_manager provided -> raises EmbeddingValidationError
        manager3 = EmbeddingModelManager(config=mismatched_cfg, model=mock_model)
        with pytest.raises(EmbeddingValidationError, match="Configuration mismatch"):
            ResearchEmbeddingGenerator(config=custom_cfg, model_manager=manager3)

        # Case 4: Neither provided -> uses default EmbeddingConfig for both
        gen4 = ResearchEmbeddingGenerator(config=None, model_manager=None)
        assert gen4.config == EmbeddingConfig()
        assert gen4.model_manager.config == EmbeddingConfig()

        # Case 5: Config provided without manager -> constructs manager with that config
        gen5 = ResearchEmbeddingGenerator(config=custom_cfg, model_manager=None)
        assert gen5.config is custom_cfg
        assert gen5.model_manager.config == custom_cfg

    def test_empty_and_whitespace_chunk_validation(self) -> None:
        """embed_chunks rejects chunks with empty or whitespace-only content with offending index."""
        generator = ResearchEmbeddingGenerator()

        # Empty string content at index 0
        chunk_empty = make_dummy_chunk(chunk_index=0, content="")
        with pytest.raises(EmbeddingValidationError, match="Chunk at index 0 has empty or whitespace-only content"):
            generator.embed_chunks([chunk_empty])

        # Whitespace-only content at index 1
        valid_chunk = make_dummy_chunk(chunk_index=0, content="Valid text content.")
        chunk_whitespace = make_dummy_chunk(chunk_index=1, content="   \t\n  ")
        with pytest.raises(EmbeddingValidationError, match="Chunk at index 1 has empty or whitespace-only content"):
            generator.embed_chunks([valid_chunk, chunk_whitespace])

    def test_empty_input_returns_empty_result_without_loading_model(self) -> None:
        """Empty chunks list returns empty EmbeddingResult without invoking model loader."""
        manager = EmbeddingModelManager()
        generator = ResearchEmbeddingGenerator(model_manager=manager)

        res = generator.embed_chunks([])

        assert isinstance(res, EmbeddingResult)
        assert res.embeddings == []
        assert res.model_name == generator.config.model_name
        assert res.provider == generator.config.provider
        assert res.dimensions == generator.config.dimensions
        assert res.normalized == generator.config.normalize_embeddings
        assert res.diagnostics.input_chunks == 0
        assert res.diagnostics.embedded_chunks == 0
        assert res.diagnostics.batch_count == 0
        assert res.diagnostics.elapsed_seconds >= 0.0
        assert res.diagnostics.warnings == []
        assert manager.is_loaded is False

    def test_single_chunk_produces_one_embedding(self) -> None:
        """A single chunk produces exactly one ChunkEmbedding with matching fields."""
        mock_model = MockEmbeddingModel(dimensions=384)
        generator = ResearchEmbeddingGenerator(model_manager=EmbeddingModelManager(model=mock_model))

        chunk = make_dummy_chunk(chunk_index=42, content="Single chunk text.", content_hash="hash_42")
        result = generator.embed_chunks([chunk])

        assert len(result.embeddings) == 1
        emb = result.embeddings[0]
        assert emb.chunk_index == 42
        assert emb.content_hash == "hash_42"
        assert len(emb.vector) == 384
        assert result.diagnostics.input_chunks == 1
        assert result.diagnostics.embedded_chunks == 1
        assert result.diagnostics.batch_count == 1

    def test_multiple_chunks_preserve_original_order_and_provenance(self) -> None:
        """Multiple chunks preserve original list ordering and respective provenance metadata."""
        mock_model = MockEmbeddingModel(dimensions=384)
        generator = ResearchEmbeddingGenerator(model_manager=EmbeddingModelManager(model=mock_model))

        chunks = [
            make_dummy_chunk(chunk_index=10, content="First chunk", content_hash="h10"),
            make_dummy_chunk(chunk_index=20, content="Second chunk", content_hash="h20"),
            make_dummy_chunk(chunk_index=30, content="Third chunk", content_hash="h30"),
        ]
        result = generator.embed_chunks(chunks)

        assert len(result.embeddings) == 3
        for i, (chunk, emb) in enumerate(zip(chunks, result.embeddings)):
            assert emb.chunk_index == chunk.chunk_index
            assert emb.content_hash == chunk.content_hash
            assert len(emb.vector) == 384

    def test_only_chunk_content_sent_to_model(self) -> None:
        """Only chunk.content is extracted and sent to model.encode without metadata."""
        mock_model = MockEmbeddingModel(dimensions=384)
        generator = ResearchEmbeddingGenerator(model_manager=EmbeddingModelManager(model=mock_model))

        c1 = make_dummy_chunk(chunk_index=1, content="Exact body text of chunk one.")
        c2 = make_dummy_chunk(chunk_index=2, content="Exact body text of chunk two.")
        generator.embed_chunks([c1, c2])

        assert len(mock_model.call_history) == 1
        sent_texts = mock_model.call_history[0]["sentences"]
        assert sent_texts == ["Exact body text of chunk one.", "Exact body text of chunk two."]

    def test_configured_batch_size_and_batch_count_respected(self) -> None:
        """Chunks are partitioned according to batch_size and accurate batch_count is reported."""
        cfg = EmbeddingConfig(batch_size=2)
        mock_model = MockEmbeddingModel(dimensions=384)
        manager = EmbeddingModelManager(config=cfg, model=mock_model)
        generator = ResearchEmbeddingGenerator(config=cfg, model_manager=manager)

        chunks = [make_dummy_chunk(i, f"Content {i}") for i in range(5)]
        result = generator.embed_chunks(chunks)

        assert len(mock_model.call_history) == 3
        assert len(mock_model.call_history[0]["sentences"]) == 2
        assert len(mock_model.call_history[1]["sentences"]) == 2
        assert len(mock_model.call_history[2]["sentences"]) == 1

        assert result.diagnostics.batch_count == 3
        assert result.diagnostics.input_chunks == 5
        assert result.diagnostics.embedded_chunks == 5
        assert len(result.embeddings) == 5

    def test_normalization_and_inference_parameters_passed(self) -> None:
        """normalize_embeddings, convert_to_numpy, and show_progress_bar flags are passed."""
        cfg_norm = EmbeddingConfig(normalize_embeddings=True)
        mock1 = MockEmbeddingModel()
        gen1 = ResearchEmbeddingGenerator(config=cfg_norm, model_manager=EmbeddingModelManager(cfg_norm, mock1))
        gen1.embed_chunks([make_dummy_chunk()])

        call1 = mock1.call_history[0]
        assert call1["normalize_embeddings"] is True
        assert call1["convert_to_numpy"] is True
        assert call1["show_progress_bar"] is False

        cfg_unnorm = EmbeddingConfig(normalize_embeddings=False)
        mock2 = MockEmbeddingModel()
        gen2 = ResearchEmbeddingGenerator(config=cfg_unnorm, model_manager=EmbeddingModelManager(cfg_unnorm, mock2))
        gen2.embed_chunks([make_dummy_chunk()])

        call2 = mock2.call_history[0]
        assert call2["normalize_embeddings"] is False

    def test_batch_inference_failure_raises_generation_error(self) -> None:
        """Model inference runtime failures raise EmbeddingGenerationError with original cause."""
        def failing_encode(sentences: list[str]) -> Any:
            raise RuntimeError("CUDA Out of Memory")

        mock_model = MockEmbeddingModel(return_func=failing_encode)
        generator = ResearchEmbeddingGenerator(model_manager=EmbeddingModelManager(model=mock_model))

        with pytest.raises(EmbeddingGenerationError, match="Model inference failed for batch 0") as exc_info:
            generator.embed_chunks([make_dummy_chunk()])

        assert isinstance(exc_info.value.__cause__, RuntimeError)

    def test_invalid_output_row_count_rejected(self) -> None:
        """Model output returning fewer or more vectors than input chunks raises EmbeddingGenerationError."""
        def bad_count_encode(sentences: list[str]) -> Any:
            # Returns 1 vector for 2 input sentences
            return [[0.1] * 384]

        mock_model = MockEmbeddingModel(return_func=bad_count_encode)
        generator = ResearchEmbeddingGenerator(model_manager=EmbeddingModelManager(model=mock_model))

        chunks = [make_dummy_chunk(0, "A"), make_dummy_chunk(1, "B")]
        with pytest.raises(EmbeddingGenerationError, match="output count mismatch: expected 2 vectors, got 1"):
            generator.embed_chunks(chunks)

    def test_incorrect_vector_dimensions_rejected(self) -> None:
        """Vectors with dimensional length different from config raise EmbeddingGenerationError."""
        def bad_dim_encode(sentences: list[str]) -> Any:
            # Returns 128 dimensions when config expects 384
            return [[0.1] * 128 for _ in sentences]

        mock_model = MockEmbeddingModel(return_func=bad_dim_encode)
        generator = ResearchEmbeddingGenerator(model_manager=EmbeddingModelManager(model=mock_model))

        with pytest.raises(EmbeddingGenerationError, match="dimension mismatch: expected 384, got 128"):
            generator.embed_chunks([make_dummy_chunk()])

    def test_non_numeric_vector_values_rejected(self) -> None:
        """Vectors containing non-numeric strings or objects raise EmbeddingGenerationError."""
        def non_numeric_encode(sentences: list[str]) -> Any:
            vec = [0.1] * 384
            vec[5] = "not_a_number"  # type: ignore[call-overload]
            return [vec]

        mock_model = MockEmbeddingModel(return_func=non_numeric_encode)
        generator = ResearchEmbeddingGenerator(model_manager=EmbeddingModelManager(model=mock_model))

        with pytest.raises(EmbeddingGenerationError, match="value 5 is not a valid real numeric value"):
            generator.embed_chunks([make_dummy_chunk()])

    @pytest.mark.parametrize(
        "invalid_val,desc",
        [
            ("0.123", "numeric_string"),
            (b"0.123", "bytes"),
            (bytearray(b"0.123"), "bytearray"),
            (1 + 2j, "complex"),
        ],
    )
    def test_strict_rejection_of_non_real_types(self, invalid_val: Any, desc: str) -> None:
        """Reject non-real numeric types such as strings, bytes, bytearray, and complex numbers."""
        def bad_val_encode(sentences: list[str]) -> Any:
            vec = [0.1] * 384
            vec[7] = invalid_val
            return [vec]

        mock_model = MockEmbeddingModel(return_func=bad_val_encode)
        generator = ResearchEmbeddingGenerator(model_manager=EmbeddingModelManager(model=mock_model))

        with pytest.raises(EmbeddingGenerationError, match="value 7 is not a valid real numeric value"):
            generator.embed_chunks([make_dummy_chunk()])

    def test_arbitrary_object_implementing_float_is_rejected(self) -> None:
        """Arbitrary objects defining __float__ are rejected if not genuine numbers.Real instances."""
        class CustomFloatLike:
            def __float__(self) -> float:
                return 0.123

        def float_like_encode(sentences: list[str]) -> Any:
            vec = [0.1] * 384
            vec[4] = CustomFloatLike()
            return [vec]

        mock_model = MockEmbeddingModel(return_func=float_like_encode)
        generator = ResearchEmbeddingGenerator(model_manager=EmbeddingModelManager(model=mock_model))

        with pytest.raises(EmbeddingGenerationError, match="value 4 is not a valid real numeric value"):
            generator.embed_chunks([make_dummy_chunk()])

    def test_valid_integers_and_floats_accepted(self) -> None:
        """Legitimate Python int and float numbers are accepted and converted to float."""
        def valid_mixed_encode(sentences: list[str]) -> Any:
            vec = [0.5] * 384
            vec[0] = 10     # integer
            vec[1] = -5     # negative integer
            vec[2] = 0      # zero
            vec[3] = 3.14   # float
            return [vec]

        mock_model = MockEmbeddingModel(return_func=valid_mixed_encode)
        generator = ResearchEmbeddingGenerator(model_manager=EmbeddingModelManager(model=mock_model))

        res = generator.embed_chunks([make_dummy_chunk()])
        v = res.embeddings[0].vector
        assert isinstance(v[0], float) and v[0] == 10.0
        assert isinstance(v[1], float) and v[1] == -5.0
        assert isinstance(v[2], float) and v[2] == 0.0
        assert isinstance(v[3], float) and v[3] == 3.14

    def test_valid_numpy_scalars_accepted(self) -> None:
        """Valid NumPy floating-point and integer scalars are accepted if numpy is available."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("NumPy not available in test environment")

        def numpy_scalar_encode(sentences: list[str]) -> Any:
            vec = [np.float32(0.1)] * 384
            vec[0] = np.float64(0.25)
            vec[1] = np.float32(0.75)
            vec[2] = np.int64(42)
            return [vec]

        mock_model = MockEmbeddingModel(return_func=numpy_scalar_encode)
        generator = ResearchEmbeddingGenerator(model_manager=EmbeddingModelManager(model=mock_model))

        res = generator.embed_chunks([make_dummy_chunk()])
        v = res.embeddings[0].vector
        assert isinstance(v[0], float) and math.isclose(v[0], 0.25)
        assert isinstance(v[1], float) and math.isclose(v[1], 0.75)
        assert isinstance(v[2], float) and v[2] == 42.0

    def test_boolean_vector_values_rejected(self) -> None:
        """Vectors containing boolean elements raise EmbeddingGenerationError."""
        def boolean_encode(sentences: list[str]) -> Any:
            vec = [0.1] * 384
            vec[3] = True
            return [vec]

        mock_model = MockEmbeddingModel(return_func=boolean_encode)
        generator = ResearchEmbeddingGenerator(model_manager=EmbeddingModelManager(model=mock_model))

        with pytest.raises(EmbeddingGenerationError, match="value 3 is boolean"):
            generator.embed_chunks([make_dummy_chunk()])

    @pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
    def test_nan_and_infinity_values_rejected(self, bad_val: float) -> None:
        """Vectors containing NaN or +/- infinity raise EmbeddingGenerationError."""
        def non_finite_encode(sentences: list[str]) -> Any:
            vec = [0.1] * 384
            vec[10] = bad_val
            return [vec]

        mock_model = MockEmbeddingModel(return_func=non_finite_encode)
        generator = ResearchEmbeddingGenerator(model_manager=EmbeddingModelManager(model=mock_model))

        with pytest.raises(EmbeddingGenerationError, match="value 10 is non-finite"):
            generator.embed_chunks([make_dummy_chunk()])

    def test_numpy_outputs_converted_to_native_python_float_lists(self) -> None:
        """NumPy array outputs are converted to native Python float lists without leaking NumPy types."""
        # Simulate numpy array without module-level dependency
        class FakeNumpy1DArray:
            def __init__(self, values: list[float]) -> None:
                self.values = values
                self.ndim = 1

            def __len__(self) -> int:
                return len(self.values)

            def __iter__(self) -> Any:
                return iter(self.values)

        class FakeNumpy2DArray:
            def __init__(self, rows: list[list[float]]) -> None:
                self.rows = [FakeNumpy1DArray(r) for r in rows]
                self.ndim = 2

            def __len__(self) -> int:
                return len(self.rows)

            def __iter__(self) -> Any:
                return iter(self.rows)

        def numpy_like_encode(sentences: list[str]) -> Any:
            return FakeNumpy2DArray([[0.123] * 384 for _ in sentences])

        mock_model = MockEmbeddingModel(return_func=numpy_like_encode)
        generator = ResearchEmbeddingGenerator(model_manager=EmbeddingModelManager(model=mock_model))

        result = generator.embed_chunks([make_dummy_chunk()])
        emb = result.embeddings[0]

        assert isinstance(emb.vector, list)
        assert len(emb.vector) == 384
        assert isinstance(emb.vector[0], float)
        assert math.isclose(emb.vector[0], 0.123, rel_tol=1e-5)

    def test_import_does_not_load_sentence_transformers(self) -> None:
        """Importing embedder does not load sentence_transformers into sys.modules."""
        check_script = (
            "import sys\n"
            "import app.modules.ingestion.embedder\n"
            "loaded = 'sentence_transformers' in sys.modules\n"
            "print('LOADED' if loaded else 'CLEAN')\n"
        )
        res = subprocess.run(
            [sys.executable, "-c", check_script],
            capture_output=True,
            text=True,
            check=True,
        )
        assert res.stdout.strip() == "CLEAN"
