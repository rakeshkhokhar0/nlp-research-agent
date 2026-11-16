"""Bi-directional mapping between UUID chunk identifiers and integer document IDs."""

from __future__ import annotations

import uuid


def _validate_uuid(chunk_id: uuid.UUID) -> None:
    """Validate that chunk_id is an instance of uuid.UUID."""
    if not isinstance(chunk_id, uuid.UUID):
        raise TypeError(
            f"chunk_id must be an instance of uuid.UUID, got {type(chunk_id).__name__}"
        )


def _validate_integer_id(integer_id: int) -> None:
    """Validate that integer_id is a positive integer and not a boolean."""
    if isinstance(integer_id, bool) or not isinstance(integer_id, int):
        raise TypeError(
            f"integer_id must be a positive int, got {type(integer_id).__name__}"
        )
    if integer_id <= 0:
        raise ValueError(
            f"integer_id must be a positive integer (> 0), got {integer_id}"
        )


class ChunkIDMapper:
    """Maintains a bi-directional O(1) mapping between PostgreSQL UUID chunk
    identifiers and sequential positive integer IDs required by the lexical engine.
    """

    def __init__(self) -> None:
        """Initialize empty bi-directional mappings with next integer ID starting at 1."""
        self._uuid_to_int: dict[uuid.UUID, int] = {}
        self._int_to_uuid: dict[int, uuid.UUID] = {}
        self._next_id: int = 1

    def get_or_create(self, chunk_id: uuid.UUID) -> int:
        """Return the existing integer ID for chunk_id or assign a new sequential positive integer ID.

        Args:
            chunk_id: Valid uuid.UUID instance.

        Returns:
            Positive integer ID associated with chunk_id.

        Raises:
            TypeError: If chunk_id is not an instance of uuid.UUID.
        """
        _validate_uuid(chunk_id)

        existing_id = self._uuid_to_int.get(chunk_id)
        if existing_id is not None:
            return existing_id

        assigned_id = self._next_id
        self._next_id += 1
        self._uuid_to_int[chunk_id] = assigned_id
        self._int_to_uuid[assigned_id] = chunk_id
        return assigned_id

    def get_integer_id(self, chunk_id: uuid.UUID) -> int | None:
        """Return the integer ID associated with chunk_id, or None if unknown.

        Args:
            chunk_id: Valid uuid.UUID instance.

        Returns:
            Positive integer ID if registered, otherwise None.

        Raises:
            TypeError: If chunk_id is not an instance of uuid.UUID.
        """
        _validate_uuid(chunk_id)
        return self._uuid_to_int.get(chunk_id)

    def get_uuid(self, integer_id: int) -> uuid.UUID | None:
        """Return the uuid.UUID associated with integer_id, or None if unknown.

        Args:
            integer_id: Positive integer ID.

        Returns:
            uuid.UUID instance if registered, otherwise None.

        Raises:
            TypeError: If integer_id is not an integer or is a boolean.
            ValueError: If integer_id is not positive (> 0).
        """
        _validate_integer_id(integer_id)
        return self._int_to_uuid.get(integer_id)

    def clear(self) -> None:
        """Clear all bi-directional mappings and reset the next integer ID to 1."""
        self._uuid_to_int.clear()
        self._int_to_uuid.clear()
        self._next_id = 1

    def __len__(self) -> int:
        """Return the number of unique UUID-to-integer mappings currently registered."""
        return len(self._uuid_to_int)
