"""Conversation service orchestrating research chat, retrieval, and LLM answer generation."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import re
import time
from typing import TYPE_CHECKING, Any, Literal
import uuid

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.logging_config import correlation_context, get_correlation_ids, log_event
from app.database.session import AsyncSessionLocal
from app.modules.conversation.context import BuiltContext, ContextManager
from app.modules.conversation.llm_client import (
    LLMAnswerResult,
    LLMClientError,
    OpenAIResponseClient,
)
from app.modules.conversation.models import Conversation
from app.modules.conversation.repository import ConversationRepository
from app.modules.conversation.schemas import (
    ChatResponse,
    CitationResponse,
    ConversationMessageResponse,
    ConversationResponse,
)

if TYPE_CHECKING:
    from app.modules.retrieval.service import RetrievalService, RetrievedChunk

logger = logging.getLogger(__name__)

NO_EVIDENCE_RESPONSE = (
    "I could not find relevant information in the ingested research papers to answer this question."
)

GREETING_RESPONSE = (
    "Hello! I am your NLP Research Agent. How can I help you analyze your indexed research papers today?"
)

UNSUPPORTED_SCOPE_RESPONSE = (
    "I'm designed specifically to assist with your available research papers. "
    "I can help explain their methodology, findings, contributions, and limitations."
)

_GREETING_REGEX = re.compile(
    r"^(hi|hello|hey|howdy|hiya|greetings|good\s+(morning|afternoon|evening|day))"
    r"([\s!.,?]+(there|agent|assistant|bot|nlp\s+agent)?)?[\s!.,?]*$",
    re.IGNORECASE,
)

_UNSUPPORTED_SCOPE_PATTERNS = [
    # General conversation / personal / chitchat
    re.compile(r"^(how\s+are\s+you|who\s+are\s+you|what\s+are\s+you|tell\s+me\s+a\s+(joke|story)|what('?s|\s+is)\s+your\s+name)[\s!.,?]*$", re.IGNORECASE),
    re.compile(r"^(what\s+is\s+the\s+meaning\s+of\s+life|who\s+(created|made)\s+you)[\s!.,?]*$", re.IGNORECASE),
    re.compile(r"^(what('?s|\s+is)\s+the\s+weather|what\s+time\s+is\s+it)[\s!.,?]*$", re.IGNORECASE),
    # General trivia / geographic / political
    re.compile(r"^(what('?s|\s+is)\s+the\s+capital\s+of|who\s+is\s+the\s+president\s+of|who\s+is\s+the\s+prime\s+minister\s+of)\b", re.IGNORECASE),
    re.compile(r"^(who\s+won\s+the\s+(world\s+cup|super\s+bowl|olympics|championship))\b", re.IGNORECASE),
    re.compile(r"^(who\s+painted\s+the\s+mona\s+lisa|what\s+is\s+the\s+tallest\s+mountain)\b", re.IGNORECASE),
    # General coding requests unrelated to research papers
    re.compile(r"^(write|create|implement|give\s+me)\s+(a\s+)?(python|javascript|typescript|java|c\+\+|c#|html|css|bash|sql|rust|go)?\s*(script|function|code|program|class|algorithm)\s+(to|that|for)\b", re.IGNORECASE),
    re.compile(r"^(how\s+to|how\s+do\s+i)\s+(write|code|implement|program)\s+(a\s+)?(quicksort|binary\s+search|linked\s+list|bubble\s+sort|fibonacci|merge\s+sort)\b", re.IGNORECASE),
    # Cooking / recipes
    re.compile(r"^(recipe\s+for|how\s+to\s+(bake|cook|make))\s+([a-z\s]+)$", re.IGNORECASE),
]


def is_greeting(text: str) -> bool:
    """Return True if text is a common conversational greeting."""
    return bool(_GREETING_REGEX.match(text.strip()))


def is_unsupported_scope(text: str) -> bool:
    """Return True if text is clearly a non-research query (trivia, general coding, casual chat)."""
    return any(pattern.search(text.strip()) for pattern in _UNSUPPORTED_SCOPE_PATTERNS)


class ConversationServiceError(Exception):
    """Base exception for conversation domain operations."""


class ConversationNotFoundError(ConversationServiceError):
    """Raised when a requested conversation is not found."""


class ConversationInactiveError(ConversationServiceError):
    """Raised when attempting to add messages to an archived or inactive conversation."""


class RetrievalUnavailableError(ConversationServiceError):
    """Raised when the retrieval service is unconfigured, uninitialized, or unavailable."""


class LLMGenerationError(ConversationServiceError):
    """Raised when the LLM service fails to generate an answer."""


class ConversationService:
    """Coordinates conversational and research question answering between PostgreSQL history,

    custom retrieval engine, context management, and the OpenAI Responses API.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        retrieval_service: RetrievalService | None = None,
        llm_client: OpenAIResponseClient | None = None,
        context_manager: ContextManager | None = None,
    ) -> None:
        """Initialize the conversation service.

        Args:
            session_factory: SQLAlchemy async_sessionmaker for short-lived sessions.
            retrieval_service: Application-scoped in-memory RetrievalService.
            llm_client: Application-scoped OpenAIResponseClient.
            context_manager: ContextManager for history assembly and summarization.
        """
        self.session_factory: async_sessionmaker[AsyncSession] = (
            session_factory if session_factory is not None else AsyncSessionLocal
        )
        self.retrieval_service: RetrievalService | None = retrieval_service
        self.llm_client: OpenAIResponseClient = (
            llm_client if llm_client is not None else OpenAIResponseClient()
        )
        self.context_manager: ContextManager = (
            context_manager
            if context_manager is not None
            else ContextManager(llm_client=self.llm_client)
        )

    async def create_conversation(
        self,
        title: str | None = None,
        chat_type: Literal["general", "paper"] | None = None,
        metadata_: dict[str, Any] | None = None,
    ) -> ConversationResponse:
        """Create a new research or general conversation session."""
        meta = dict(metadata_) if metadata_ is not None else None
        if chat_type is not None:
            if meta is None:
                meta = {}
            meta["chat_type"] = chat_type

        async with self.session_factory() as session:
            async with session.begin():
                repo = ConversationRepository(session)
                conv = await repo.create_conversation(title=title, metadata_=meta)

            effective_chat_type = (meta.get("chat_type") if meta else None) or "paper"
            log_event(
                logger,
                logging.INFO,
                "conversation_created",
                conversation_id=str(conv.id),
                title=conv.title,
                chat_type=effective_chat_type,
            )

            return ConversationResponse(
                id=conv.id,
                title=conv.title,
                status=conv.status,
                chat_type=effective_chat_type,
                context_summary=conv.context_summary,
                created_at=conv.created_at,
                updated_at=conv.updated_at,
                last_message_at=conv.last_message_at,
            )

    async def get_conversation(self, conversation_id: uuid.UUID) -> ConversationResponse:
        """Retrieve conversation metadata by ID."""
        async with self.session_factory() as session:
            repo = ConversationRepository(session)
            conv = await repo.get_conversation(conversation_id)
            if conv is None:
                raise ConversationNotFoundError(f"Conversation '{conversation_id}' not found.")

            conv_chat_type = (conv.metadata_ or {}).get("chat_type", "paper")
            return ConversationResponse(
                id=conv.id,
                title=conv.title,
                status=conv.status,
                chat_type=conv_chat_type,
                context_summary=conv.context_summary,
                created_at=conv.created_at,
                updated_at=conv.updated_at,
                last_message_at=conv.last_message_at,
            )

    async def get_conversation_history(
        self,
        conversation_id: uuid.UUID,
    ) -> list[ConversationMessageResponse]:
        """Retrieve the ordered message history with citations for a conversation."""
        async with self.session_factory() as session:
            repo = ConversationRepository(session)
            conv = await repo.get_conversation(conversation_id)
            if conv is None:
                raise ConversationNotFoundError(f"Conversation '{conversation_id}' not found.")

            messages = await repo.get_conversation_messages(conversation_id, include_citations=True)
            responses: list[ConversationMessageResponse] = []
            for msg in messages:
                citations_resp: list[CitationResponse] = []
                for cit in msg.citations:
                    paper_title = (
                        cit.chunk.paper.title
                        if cit.chunk and cit.chunk.paper
                        else "Research Paper"
                    )
                    paper_id = cit.chunk.paper_id if cit.chunk else uuid.UUID(int=0)
                    citations_resp.append(
                        CitationResponse(
                            chunk_id=cit.chunk_id,
                            paper_id=paper_id,
                            paper_title=paper_title,
                            citation_order=cit.citation_order,
                            page_number=cit.page_number,
                            relevance_score=cit.relevance_score,
                            quote_text=cit.cited_text,
                        )
                    )

                msg_mode = None
                if hasattr(msg, "metadata_") and isinstance(msg.metadata_, dict):
                    msg_mode = msg.metadata_.get("mode")

                responses.append(
                    ConversationMessageResponse(
                        id=msg.id,
                        conversation_id=msg.conversation_id,
                        sequence_number=msg.sequence_number,
                        role=msg.role,
                        content=msg.content,
                        status=msg.status,
                        model_name=msg.model_name,
                        token_count=msg.token_count,
                        mode=msg_mode,
                        citations=citations_resp,
                        created_at=msg.created_at,
                    )
                )

            return responses

    async def ask_question(
        self,
        conversation_id: uuid.UUID,
        question: str,
        top_k: int = 5,
        paper_ids: list[uuid.UUID] | None = None,
        mode: Literal["normal", "advanced"] = "normal",
        chat_type: Literal["general", "paper"] | None = None,
    ) -> ChatResponse:
        """Process a user question against context management, retrieval (if paper chat), and LLM.

        Supports two chat types:
        1. 'general': Context-aware general conversation without retrieval.
        2. 'paper': Grounded research paper question answering with custom lexical retrieval.
        """
        if not question or not question.strip():
            raise ValueError("question must not be empty or whitespace.")
        clean_question = question.strip()
        selected_mode: Literal["normal", "advanced"] = "advanced" if mode == "advanced" else "normal"

        with correlation_context(conversation_id=str(conversation_id)):
            current_req_id = get_correlation_ids().get("request_id") or str(uuid.uuid4())

            # 1. Validate conversation exists and is active, and build context
            async with self.session_factory() as session:
                repo = ConversationRepository(session)
                conv = await repo.get_conversation(conversation_id)
                if conv is None:
                    raise ConversationNotFoundError(f"Conversation '{conversation_id}' not found.")
                if conv.status != "active":
                    raise ConversationInactiveError(
                        f"Conversation '{conversation_id}' is not active (status='{conv.status}')."
                    )

                # Determine effective chat type and enforce conversation consistency
                conv_metadata = dict(conv.metadata_ or {})
                locked_type = conv_metadata.get("chat_type")

                if locked_type in {"general", "paper"}:
                    if chat_type is not None and chat_type != locked_type:
                        raise ValueError(
                            f"Conversation '{conversation_id}' is locked to chat_type '{locked_type}'. "
                            f"Cannot send message with conflicting chat_type '{chat_type}'."
                        )
                    effective_chat_type: Literal["general", "paper"] = locked_type
                else:
                    effective_chat_type = chat_type if chat_type in {"general", "paper"} else "paper"
                    conv_metadata["chat_type"] = effective_chat_type
                    conv.metadata_ = conv_metadata
                    await session.flush()

                # Summarize older messages if history exceeds threshold
                await self.context_manager.maybe_summarize_history(conversation=conv, repo=repo)

                # Build context (recent history + summary + current question token budgeting)
                context = await self.context_manager.build_context(
                    conversation_id=conversation_id,
                    repo=repo,
                    current_question=clean_question,
                    summary=conv.context_summary,
                    chat_type=effective_chat_type,
                )

                await session.commit()

            # 2. General Chat Workflow (no retrieval invoked)
            if effective_chat_type == "general":
                return await self._handle_general_chat(
                    conversation_id=conversation_id,
                    clean_question=clean_question,
                    context=context,
                    selected_mode=selected_mode,
                    current_req_id=current_req_id,
                )

            # 3. Chat with Paper Workflow (grounded retrieval + citations)
            return await self._handle_paper_chat(
                conversation_id=conversation_id,
                clean_question=clean_question,
                context=context,
                top_k=top_k,
                paper_ids=paper_ids,
                selected_mode=selected_mode,
                current_req_id=current_req_id,
            )

    async def _handle_general_chat(
        self,
        conversation_id: uuid.UUID,
        clean_question: str,
        context: BuiltContext,
        selected_mode: Literal["normal", "advanced"],
        current_req_id: str,
    ) -> ChatResponse:
        """Handle context-aware General Chat without invoking research retrieval."""
        log_event(
            logger,
            logging.INFO,
            "general_chat_started",
            request_id=current_req_id,
            conversation_id=str(conversation_id),
            has_summary=bool(context.summary),
            history_length=len(context.history_messages),
        )

        # 1. Save user message in short transaction
        async with self.session_factory() as session:
            async with session.begin():
                repo = ConversationRepository(session)
                await repo.get_conversation(conversation_id, for_update=True)
                seq_user = await repo.get_next_sequence_number(conversation_id)
                user_msg = await repo.create_message(
                    conversation_id=conversation_id,
                    sequence_number=seq_user,
                    role="user",
                    content=clean_question,
                    status="completed",
                    metadata_={"chat_type": "general", "mode": selected_mode},
                )
                await repo.update_conversation_last_message_at(conversation_id)

        user_message_id = user_msg.id

        # 2. LLM general answer generation (no retrieval)
        llm_start = time.perf_counter()
        try:
            llm_result = await self.llm_client.generate_general_answer(
                question=clean_question,
                conversation_history=context.history_messages,
                conversation_summary=context.summary,
            )
        except Exception as exc:
            logger.error("OpenAI general chat generation failed: %s", type(exc).__name__)
            raise LLMGenerationError(
                "Failed to generate answer from language model. Please try again."
            ) from exc

        llm_duration_ms = (time.perf_counter() - llm_start) * 1000.0

        # 3. Save assistant message
        async with self.session_factory() as session:
            async with session.begin():
                repo = ConversationRepository(session)
                await repo.get_conversation(conversation_id, for_update=True)
                seq_asst = await repo.get_next_sequence_number(conversation_id)
                asst_msg = await repo.create_message(
                    conversation_id=conversation_id,
                    sequence_number=seq_asst,
                    role="assistant",
                    content=llm_result.answer,
                    model_name=llm_result.model_name,
                    token_count=llm_result.token_count,
                    status="completed",
                    metadata_={"chat_type": "general", "mode": selected_mode},
                )
                await repo.update_conversation_last_message_at(conversation_id)

        log_event(
            logger,
            logging.INFO,
            "general_chat_completed",
            request_id=current_req_id,
            conversation_id=str(conversation_id),
            user_message_id=str(user_message_id),
            assistant_message_id=str(asst_msg.id),
            model_name=llm_result.model_name,
            token_count=llm_result.token_count,
            llm_duration_ms=round(llm_duration_ms, 2),
        )

        return ChatResponse(
            conversation_id=conversation_id,
            user_message_id=user_message_id,
            assistant_message_id=asst_msg.id,
            answer=llm_result.answer,
            citations=[],
            model_name=llm_result.model_name,
            mode=selected_mode,
            chat_type="general",
            created_at=asst_msg.created_at or datetime.now(timezone.utc),
        )

    async def _handle_paper_chat(
        self,
        conversation_id: uuid.UUID,
        clean_question: str,
        context: BuiltContext,
        top_k: int,
        paper_ids: list[uuid.UUID] | None,
        selected_mode: Literal["normal", "advanced"],
        current_req_id: str,
    ) -> ChatResponse:
        """Handle grounded research Chat with Paper using retrieval and citation validation."""
        # 1. Check for basic conversational greetings (no retrieval or LLM required)
        if is_greeting(clean_question):
            log_event(
                logger,
                logging.INFO,
                "greeting_detected",
                request_id=current_req_id,
                conversation_id=str(conversation_id),
                mode=selected_mode,
            )
            async with self.session_factory() as session:
                async with session.begin():
                    repo = ConversationRepository(session)
                    await repo.get_conversation(conversation_id, for_update=True)
                    seq_user = await repo.get_next_sequence_number(conversation_id)
                    user_msg = await repo.create_message(
                        conversation_id=conversation_id,
                        sequence_number=seq_user,
                        role="user",
                        content=clean_question,
                        status="completed",
                        metadata_={"mode": selected_mode},
                    )
                    seq_asst = seq_user + 1
                    asst_msg = await repo.create_message(
                        conversation_id=conversation_id,
                        sequence_number=seq_asst,
                        role="assistant",
                        content=GREETING_RESPONSE,
                        model_name=None,
                        status="completed",
                        metadata_={"mode": selected_mode},
                    )
                    await repo.update_conversation_last_message_at(conversation_id)

            return ChatResponse(
                conversation_id=conversation_id,
                user_message_id=user_msg.id,
                assistant_message_id=asst_msg.id,
                answer=GREETING_RESPONSE,
                citations=[],
                model_name=None,
                mode=selected_mode,
                chat_type="paper",
                created_at=asst_msg.created_at or datetime.now(timezone.utc),
            )

        # 2. Check for unsupported non-research questions (general knowledge, coding, chitchat)
        if is_unsupported_scope(clean_question):
            log_event(
                logger,
                logging.INFO,
                "unsupported_scope_declined",
                request_id=current_req_id,
                conversation_id=str(conversation_id),
                mode=selected_mode,
            )
            async with self.session_factory() as session:
                async with session.begin():
                    repo = ConversationRepository(session)
                    await repo.get_conversation(conversation_id, for_update=True)
                    seq_user = await repo.get_next_sequence_number(conversation_id)
                    user_msg = await repo.create_message(
                        conversation_id=conversation_id,
                        sequence_number=seq_user,
                        role="user",
                        content=clean_question,
                        status="completed",
                        metadata_={"mode": selected_mode},
                    )
                    seq_asst = seq_user + 1
                    asst_msg = await repo.create_message(
                        conversation_id=conversation_id,
                        sequence_number=seq_asst,
                        role="assistant",
                        content=UNSUPPORTED_SCOPE_RESPONSE,
                        model_name=None,
                        status="completed",
                        metadata_={"mode": selected_mode},
                    )
                    await repo.update_conversation_last_message_at(conversation_id)

            return ChatResponse(
                conversation_id=conversation_id,
                user_message_id=user_msg.id,
                assistant_message_id=asst_msg.id,
                answer=UNSUPPORTED_SCOPE_RESPONSE,
                citations=[],
                model_name=None,
                mode=selected_mode,
                chat_type="paper",
                created_at=asst_msg.created_at or datetime.now(timezone.utc),
            )

        # 3. Verify retrieval service availability and retrieve relevant chunks
        if self.retrieval_service is None or not self.retrieval_service.is_initialized:
            logger.error("RetrievalService is unavailable or uninitialized.")
            raise RetrievalUnavailableError(
                "Retrieval service is unavailable or not initialized. Search index must be refreshed before answering questions."
            )

        retrieval_start = time.perf_counter()
        fetch_k = top_k * 3 if paper_ids else top_k
        raw_results = self.retrieval_service.search(query=clean_question, top_k=fetch_k)
        if paper_ids:
            target_pids = set(paper_ids)
            retrieved_chunks = [c for c in raw_results if c.paper_id in target_pids][:top_k]
        else:
            retrieved_chunks = raw_results[:top_k]
        retrieval_duration_ms = (time.perf_counter() - retrieval_start) * 1000.0

        evidence_size_chars = sum(len(c.content) for c in retrieved_chunks)

        log_event(
            logger,
            logging.INFO,
            "retrieval_executed",
            request_id=current_req_id,
            conversation_id=str(conversation_id),
            mode=selected_mode,
            retrieved_chunk_count=len(retrieved_chunks),
            top_k=top_k,
            evidence_size_chars=evidence_size_chars,
            retrieval_duration_ms=round(retrieval_duration_ms, 2),
        )

        # 4. Handle empty retrieval results without calling OpenAI
        if not retrieved_chunks:
            log_event(
                logger,
                logging.INFO,
                "empty_retrieval_no_evidence",
                request_id=current_req_id,
                conversation_id=str(conversation_id),
                mode=selected_mode,
            )
            async with self.session_factory() as session:
                async with session.begin():
                    repo = ConversationRepository(session)
                    await repo.get_conversation(conversation_id, for_update=True)
                    seq_user = await repo.get_next_sequence_number(conversation_id)
                    user_msg = await repo.create_message(
                        conversation_id=conversation_id,
                        sequence_number=seq_user,
                        role="user",
                        content=clean_question,
                        status="completed",
                        metadata_={"mode": selected_mode},
                    )
                    seq_asst = seq_user + 1
                    asst_msg = await repo.create_message(
                        conversation_id=conversation_id,
                        sequence_number=seq_asst,
                        role="assistant",
                        content=NO_EVIDENCE_RESPONSE,
                        model_name=None,
                        status="completed",
                        metadata_={"mode": selected_mode},
                    )
                    await repo.update_conversation_last_message_at(conversation_id)

            return ChatResponse(
                conversation_id=conversation_id,
                user_message_id=user_msg.id,
                assistant_message_id=asst_msg.id,
                answer=NO_EVIDENCE_RESPONSE,
                citations=[],
                model_name=None,
                mode=selected_mode,
                chat_type="paper",
                created_at=asst_msg.created_at or datetime.now(timezone.utc),
            )

        # 5. Save user message in a short database transaction
        async with self.session_factory() as session:
            async with session.begin():
                repo = ConversationRepository(session)
                await repo.get_conversation(conversation_id, for_update=True)
                seq_user = await repo.get_next_sequence_number(conversation_id)
                user_msg = await repo.create_message(
                    conversation_id=conversation_id,
                    sequence_number=seq_user,
                    role="user",
                    content=clean_question,
                    status="completed",
                    metadata_={"mode": selected_mode},
                )
                await repo.update_conversation_last_message_at(conversation_id)

        user_message_id = user_msg.id

        # 6. LLM answer generation outside any database transaction
        # Evidence chunks are separate factual evidence; summary is for context only
        llm_start = time.perf_counter()
        try:
            llm_result = await self.llm_client.generate_grounded_answer(
                question=clean_question,
                evidence_chunks=retrieved_chunks,
                conversation_history=context.history_messages,
                conversation_summary=context.summary,
                mode=selected_mode,
            )
        except Exception as exc:
            logger.error("OpenAI answer generation failed: %s", type(exc).__name__)
            raise LLMGenerationError(
                "Failed to generate answer from language model. Please try again."
            ) from exc

        llm_duration_ms = (time.perf_counter() - llm_start) * 1000.0

        # 7. Validate cited chunk IDs and fetch page information
        retrieved_map = {c.chunk_id: c for c in retrieved_chunks}
        validated_chunk_ids: list[uuid.UUID] = []
        seen_ids: set[uuid.UUID] = set()

        for raw_id in llm_result.cited_chunk_ids:
            try:
                uid = uuid.UUID(raw_id) if not isinstance(raw_id, uuid.UUID) else raw_id
            except (ValueError, TypeError):
                logger.warning("Ignoring malformed citation chunk UUID: %s", raw_id)
                continue

            if uid in retrieved_map and uid not in seen_ids:
                seen_ids.add(uid)
                validated_chunk_ids.append(uid)

        log_event(
            logger,
            logging.INFO,
            "citations_validated",
            request_id=current_req_id,
            conversation_id=str(conversation_id),
            mode=selected_mode,
            raw_citations_count=len(llm_result.cited_chunk_ids),
            validated_citations_count=len(validated_chunk_ids),
        )

        # Look up page numbers from database for validated citations
        async with self.session_factory() as session:
            repo = ConversationRepository(session)
            page_info = await repo.get_chunk_page_information(validated_chunk_ids)

        # 8. Save assistant response and validated citations in a short transaction
        citations_response: list[CitationResponse] = []
        async with self.session_factory() as session:
            async with session.begin():
                repo = ConversationRepository(session)
                await repo.get_conversation(conversation_id, for_update=True)
                seq_asst = await repo.get_next_sequence_number(conversation_id)
                asst_msg = await repo.create_message(
                    conversation_id=conversation_id,
                    sequence_number=seq_asst,
                    role="assistant",
                    content=llm_result.answer,
                    model_name=llm_result.model_name,
                    token_count=llm_result.token_count,
                    status="completed",
                    metadata_={"mode": selected_mode},
                )

                citations_data: list[dict[str, Any]] = []
                for order, cid in enumerate(validated_chunk_ids, start=1):
                    chunk_obj = retrieved_map[cid]
                    page_num = page_info.get(cid)
                    snippet = chunk_obj.content[:300] if chunk_obj.content else None
                    citations_data.append({
                        "chunk_id": cid,
                        "citation_order": order,
                        "relevance_score": chunk_obj.score,
                        "cited_text": snippet,
                        "page_number": page_num,
                    })
                    citations_response.append(
                        CitationResponse(
                            chunk_id=cid,
                            paper_id=chunk_obj.paper_id,
                            paper_title=chunk_obj.paper_title,
                            citation_order=order,
                            page_number=page_num,
                            relevance_score=chunk_obj.score,
                            quote_text=snippet,
                        )
                    )

                if citations_data:
                    await repo.create_citations(asst_msg.id, citations_data)
                await repo.update_conversation_last_message_at(conversation_id)

        log_event(
            logger,
            logging.INFO,
            "message_persisted",
            request_id=current_req_id,
            conversation_id=str(conversation_id),
            user_message_id=str(user_message_id),
            assistant_message_id=str(asst_msg.id),
            mode=selected_mode,
            model_name=llm_result.model_name,
            token_count=llm_result.token_count,
            llm_duration_ms=round(llm_duration_ms, 2),
            persistence_outcome="success",
        )

        return ChatResponse(
            conversation_id=conversation_id,
            user_message_id=user_message_id,
            assistant_message_id=asst_msg.id,
            answer=llm_result.answer,
            citations=citations_response,
            model_name=llm_result.model_name,
            mode=selected_mode,
            chat_type="paper",
            created_at=asst_msg.created_at or datetime.now(timezone.utc),
        )
