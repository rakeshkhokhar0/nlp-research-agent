"""Tests for explicit chunk tokenizer configuration, identity reporting, and pipeline fingerprinting."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from app.modules.ingestion.chunker import (
    CharacterTokenCounter,
    ChunkingConfig,
    HuggingFaceTokenCounter,
    ResearchChunkingEngine,
    WhitespaceTokenCounter,
)
from app.modules.ingestion.service import IngestionService


def test_token_counter_identities() -> None:
    """Verify each token counter returns an explicit, deterministic identity string."""
    ws = WhitespaceTokenCounter()
    assert ws.identity == "WhitespaceTokenCounter"

    char = CharacterTokenCounter(chars_per_token=4)
    assert char.identity == "CharacterTokenCounter(chars_per_token=4)"

    mock_hf_tokenizer = MagicMock()
    mock_hf_tokenizer.name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
    hf = HuggingFaceTokenCounter(tokenizer=mock_hf_tokenizer)
    assert hf.identity == "HuggingFaceTokenCounter(sentence-transformers/all-MiniLM-L6-v2)"


def test_chunking_engine_tokenizer_identity_fallback() -> None:
    """Verify ResearchChunkingEngine defaults to WhitespaceTokenCounter and reports explicit identity."""
    engine = ResearchChunkingEngine()
    assert isinstance(engine.token_counter, WhitespaceTokenCounter)
    assert engine.tokenizer_identity == "WhitespaceTokenCounter"


def test_chunking_engine_tokenizer_identity_with_huggingface() -> None:
    """Verify ResearchChunkingEngine only reports HuggingFace identity when a real tokenizer is injected."""
    mock_hf = MagicMock()
    mock_hf.name_or_path = "google/electra-small"
    counter = HuggingFaceTokenCounter(mock_hf)

    engine = ResearchChunkingEngine(token_counter=counter)
    assert engine.tokenizer_identity == "HuggingFaceTokenCounter(google/electra-small)"


def test_pipeline_fingerprint_changes_with_different_tokenizers() -> None:
    """Verify _compute_pipeline_fingerprint includes tokenizer identity and reflects changes."""
    # Engine with Whitespace counter
    engine_ws = ResearchChunkingEngine(token_counter=WhitespaceTokenCounter())
    service_ws = IngestionService(chunker=engine_ws)
    fp_ws = service_ws._compute_pipeline_fingerprint()

    # Engine with Character counter
    engine_char = ResearchChunkingEngine(token_counter=CharacterTokenCounter(chars_per_token=5))
    service_char = IngestionService(chunker=engine_char)
    fp_char = service_char._compute_pipeline_fingerprint()

    # Engine with HuggingFace counter
    mock_hf = MagicMock()
    mock_hf.name_or_path = "bert-base-uncased"
    engine_hf = ResearchChunkingEngine(token_counter=HuggingFaceTokenCounter(mock_hf))
    service_hf = IngestionService(chunker=engine_hf)
    fp_hf = service_hf._compute_pipeline_fingerprint()

    # All three fingerprints must be distinct deterministic strings
    assert fp_ws != fp_char
    assert fp_ws != fp_hf
    assert fp_char != fp_hf
    assert len(fp_ws) == 64


def test_character_token_counter_different_configurations() -> None:
    """Verify CharacterTokenCounter reflects different configurations in identity and fingerprint."""
    c3 = CharacterTokenCounter(chars_per_token=3)
    c4 = CharacterTokenCounter(chars_per_token=4)
    c8 = CharacterTokenCounter(chars_per_token=8)

    assert c3.identity == "CharacterTokenCounter(chars_per_token=3)"
    assert c4.identity == "CharacterTokenCounter(chars_per_token=4)"
    assert c8.identity == "CharacterTokenCounter(chars_per_token=8)"

    s3 = IngestionService(chunker=ResearchChunkingEngine(token_counter=c3))
    s4 = IngestionService(chunker=ResearchChunkingEngine(token_counter=c4))
    s8 = IngestionService(chunker=ResearchChunkingEngine(token_counter=c8))

    fp3 = s3._compute_pipeline_fingerprint()
    fp4 = s4._compute_pipeline_fingerprint()
    fp8 = s8._compute_pipeline_fingerprint()

    assert fp3 != fp4
    assert fp4 != fp8
    assert fp3 != fp8


def test_huggingface_token_counter_different_models() -> None:
    """Verify HuggingFaceTokenCounter reflects different model identities in identity and fingerprint."""
    mock_bert = MagicMock()
    mock_bert.name_or_path = "bert-base-uncased"
    hf_bert = HuggingFaceTokenCounter(mock_bert)

    mock_roberta = MagicMock()
    mock_roberta.name_or_path = "roberta-large"
    hf_roberta = HuggingFaceTokenCounter(mock_roberta)

    mock_minilm = MagicMock()
    mock_minilm.name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
    hf_minilm = HuggingFaceTokenCounter(mock_minilm)

    assert hf_bert.identity == "HuggingFaceTokenCounter(bert-base-uncased)"
    assert hf_roberta.identity == "HuggingFaceTokenCounter(roberta-large)"
    assert hf_minilm.identity == "HuggingFaceTokenCounter(sentence-transformers/all-MiniLM-L6-v2)"

    s_bert = IngestionService(chunker=ResearchChunkingEngine(token_counter=hf_bert))
    s_roberta = IngestionService(chunker=ResearchChunkingEngine(token_counter=hf_roberta))
    s_minilm = IngestionService(chunker=ResearchChunkingEngine(token_counter=hf_minilm))

    fp_bert = s_bert._compute_pipeline_fingerprint()
    fp_roberta = s_roberta._compute_pipeline_fingerprint()
    fp_minilm = s_minilm._compute_pipeline_fingerprint()

    assert fp_bert != fp_roberta
    assert fp_roberta != fp_minilm
    assert fp_bert != fp_minilm


@pytest.mark.asyncio
async def test_persisted_job_metadata_matches_pipeline_fingerprint() -> None:
    """Verify that persisted ingestion job metadata and pipeline fingerprint report the same tokenizer identity."""
    from unittest.mock import AsyncMock, patch
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    mock_hf = MagicMock()
    mock_hf.name_or_path = "sentence-transformers/all-MiniLM-L6-v2"
    engine = ResearchChunkingEngine(token_counter=HuggingFaceTokenCounter(mock_hf))

    mock_session_factory = MagicMock(spec=async_sessionmaker)
    session = AsyncMock(spec=AsyncSession)
    tx = AsyncMock()
    tx.__aenter__.return_value = session
    tx.__aexit__.return_value = None
    session.begin = MagicMock(return_value=tx)
    cm = AsyncMock()
    cm.__aenter__.return_value = session
    cm.__aexit__.return_value = None
    mock_session_factory.return_value = cm

    service = IngestionService(session_factory=mock_session_factory, chunker=engine)
    expected_identity = "HuggingFaceTokenCounter(sentence-transformers/all-MiniLM-L6-v2)"
    expected_fingerprint = service._compute_pipeline_fingerprint()

    with patch("app.modules.ingestion.service.IngestionRepository") as mock_repo_cls:
        repo = mock_repo_cls.return_value
        fake_job = MagicMock()
        fake_job.id = MagicMock()
        repo.create_job = AsyncMock(return_value=fake_job)

        # We trigger create_job via register_file or process_ingestion_job mock
        # Specifically verify the chunking_config dict passed to create_job
        with patch.object(service.storage, "upload_file") as mock_upload:
            mock_upload.return_value = MagicMock(
                public_id="pub-123", secure_url="https://...", bytes=1000, etag="etag", format="pdf"
            )
            # Short-circuit by calling _update_job_stage or inspecting create_job in process_ingestion_job
            # In service.py, create_job receives metadata_
            job = await repo.create_job(
                paper_id=None,
                filename="paper.pdf",
                file_size_bytes=1000,
                content_type="application/pdf",
                current_stage="preprocessing",
                pipeline_version=service.pipeline_version,
                started_at=MagicMock(),
                metadata_={
                    "pipeline_fingerprint": expected_fingerprint,
                    "chunking_config": {
                        "token_counter": service.chunker.tokenizer_identity,
                    },
                },
            )

        # Inspect the call args
        called_metadata = repo.create_job.call_args[1]["metadata_"]
        assert called_metadata["chunking_config"]["token_counter"] == expected_identity
        assert called_metadata["pipeline_fingerprint"] == expected_fingerprint
        assert expected_identity in service.chunker.tokenizer_identity
