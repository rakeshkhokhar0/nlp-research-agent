"""Unit tests for ChunkIDMapper."""

from __future__ import annotations

import uuid

import pytest

from app.modules.retrieval import ChunkIDMapper


class TestChunkIDMapper:
    """Tests for ChunkIDMapper bi-directional mapping functionality."""

    def test_initialization_and_empty_length(self) -> None:
        """New mapper starts empty with length 0."""
        mapper = ChunkIDMapper()
        assert len(mapper) == 0

    def test_first_id_assignment(self) -> None:
        """First registered UUID receives integer ID 1."""
        mapper = ChunkIDMapper()
        uid = uuid.uuid4()
        int_id = mapper.get_or_create(uid)
        assert int_id == 1
        assert len(mapper) == 1

    def test_sequential_id_assignment(self) -> None:
        """Subsequent UUIDs receive strictly sequential integer IDs starting at 1."""
        mapper = ChunkIDMapper()
        u1 = uuid.uuid4()
        u2 = uuid.uuid4()
        u3 = uuid.uuid4()

        id1 = mapper.get_or_create(u1)
        id2 = mapper.get_or_create(u2)
        id3 = mapper.get_or_create(u3)

        assert id1 == 1
        assert id2 == 2
        assert id3 == 3
        assert len(mapper) == 3

    def test_repeated_uuid_lookup_returns_same_id(self) -> None:
        """Calling get_or_create repeatedly with the same UUID returns the identical integer ID."""
        mapper = ChunkIDMapper()
        uid = uuid.uuid4()

        first = mapper.get_or_create(uid)
        second = mapper.get_or_create(uid)
        third = mapper.get_or_create(uid)

        assert first == second == third == 1
        assert len(mapper) == 1

    def test_uuid_to_integer_lookup(self) -> None:
        """get_integer_id returns the registered integer ID without creating new mappings."""
        mapper = ChunkIDMapper()
        uid = uuid.uuid4()
        created_id = mapper.get_or_create(uid)

        assert mapper.get_integer_id(uid) == created_id
        assert len(mapper) == 1

    def test_integer_to_uuid_reverse_lookup(self) -> None:
        """get_uuid retrieves the exact uuid.UUID mapped to an integer ID."""
        mapper = ChunkIDMapper()
        uid = uuid.uuid4()
        int_id = mapper.get_or_create(uid)

        assert mapper.get_uuid(int_id) == uid

    def test_unknown_uuid_and_integer_lookups_return_none(self) -> None:
        """Querying unregistered UUIDs or integer IDs returns None without error."""
        mapper = ChunkIDMapper()
        unregistered_uuid = uuid.uuid4()

        assert mapper.get_integer_id(unregistered_uuid) is None
        assert mapper.get_uuid(999) is None
        assert len(mapper) == 0

    def test_uniqueness_and_bidirectional_consistency(self) -> None:
        """Different UUIDs receive distinct IDs and bi-directional lookups remain consistent."""
        mapper = ChunkIDMapper()
        uuids = [uuid.uuid4() for _ in range(50)]
        int_ids = [mapper.get_or_create(u) for u in uuids]

        # Distinct integer IDs
        assert len(set(int_ids)) == 50
        assert int_ids == list(range(1, 51))

        # Consistent bi-directional lookups
        for u, int_id in zip(uuids, int_ids):
            assert mapper.get_integer_id(u) == int_id
            assert mapper.get_uuid(int_id) == u

    def test_clear_and_reuse_after_clearing(self) -> None:
        """clear() empties mappings and resets next ID to 1 for subsequent registrations."""
        mapper = ChunkIDMapper()
        u1 = uuid.uuid4()
        u2 = uuid.uuid4()

        mapper.get_or_create(u1)
        mapper.get_or_create(u2)
        assert len(mapper) == 2

        mapper.clear()
        assert len(mapper) == 0
        assert mapper.get_integer_id(u1) is None
        assert mapper.get_integer_id(u2) is None
        assert mapper.get_uuid(1) is None
        assert mapper.get_uuid(2) is None

        # Re-registering after clear resets sequence to 1
        u3 = uuid.uuid4()
        new_id = mapper.get_or_create(u3)
        assert new_id == 1
        assert len(mapper) == 1
        assert mapper.get_uuid(1) == u3

    @pytest.mark.parametrize("invalid_uuid", [None, "str-uuid", 123, 4.5, True, [uuid.uuid4()]])
    def test_invalid_uuid_types_raise_type_error(self, invalid_uuid: object) -> None:
        """Non-uuid.UUID arguments to get_or_create and get_integer_id raise TypeError."""
        mapper = ChunkIDMapper()

        with pytest.raises(TypeError, match="chunk_id must be an instance of uuid.UUID"):
            mapper.get_or_create(invalid_uuid)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="chunk_id must be an instance of uuid.UUID"):
            mapper.get_integer_id(invalid_uuid)  # type: ignore[arg-type]

    @pytest.mark.parametrize("invalid_int_type", [True, False, 1.5, "1", None, [1], {"id": 1}])
    def test_invalid_integer_types_raise_type_error(self, invalid_int_type: object) -> None:
        """Non-int or boolean integer_id arguments to get_uuid raise TypeError."""
        mapper = ChunkIDMapper()

        with pytest.raises(TypeError, match="integer_id must be a positive int"):
            mapper.get_uuid(invalid_int_type)  # type: ignore[arg-type]

    @pytest.mark.parametrize("non_positive_int", [0, -1, -50, -9999])
    def test_non_positive_integer_ids_raise_value_error(self, non_positive_int: int) -> None:
        """Zero or negative integer_id arguments to get_uuid raise ValueError."""
        mapper = ChunkIDMapper()

        with pytest.raises(ValueError, match="integer_id must be a positive integer"):
            mapper.get_uuid(non_positive_int)

    def test_multiple_mapper_instances_remain_independent(self) -> None:
        """Distinct ChunkIDMapper instances maintain isolated state and sequences."""
        mapper_a = ChunkIDMapper()
        mapper_b = ChunkIDMapper()

        shared_uuid = uuid.uuid4()
        unique_uuid_a = uuid.uuid4()

        id_a_1 = mapper_a.get_or_create(unique_uuid_a)
        id_a_2 = mapper_a.get_or_create(shared_uuid)

        # In mapper_b, shared_uuid is registered first, so it gets ID 1
        id_b_1 = mapper_b.get_or_create(shared_uuid)

        assert id_a_1 == 1
        assert id_a_2 == 2
        assert id_b_1 == 1

        assert len(mapper_a) == 2
        assert len(mapper_b) == 1
        assert mapper_b.get_integer_id(unique_uuid_a) is None
