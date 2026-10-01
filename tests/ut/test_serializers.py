import pickle
from collections import namedtuple
from unittest import mock

import pytest

from aiocache.serializers import (
    BaseSerializer,
    CorruptedPayloadError,
    JsonSerializer,
    MigrationCycleError,
    MissingMigrationStep,
    MsgPackSerializer,
    NullSerializer,
    PickleSerializer,
    StringSerializer,
    UnknownVersionError,
    VersionedSerializer,
)

Dummy = namedtuple("Dummy", "a, b")

TYPES = [1, 2.0, "hi", True, ["1", 1], {"key": "value"}, Dummy(1, 2)]
JSON_TYPES = [1, 2.0, "hi", True, ["1", 1], {"key": "value"}]


class TestNullSerializer:
    def test_init(self):
        serializer = NullSerializer()
        assert isinstance(serializer, BaseSerializer)
        assert serializer.DEFAULT_ENCODING == "utf-8"
        assert serializer.encoding == "utf-8"

    def test_init_encoding(self):
        serializer = NullSerializer(encoding="whatever")
        assert serializer.DEFAULT_ENCODING == "utf-8"
        assert serializer.encoding == "whatever"

    @pytest.mark.parametrize("obj", TYPES)
    def test_set_types(self, obj):
        assert NullSerializer().dumps(obj) is obj

    def test_loads(self):
        assert NullSerializer().loads("hi") == "hi"


class TestStringSerializer:
    def test_init(self):
        serializer = StringSerializer()
        assert isinstance(serializer, BaseSerializer)
        assert serializer.DEFAULT_ENCODING == "utf-8"
        assert serializer.encoding == "utf-8"

    @pytest.mark.parametrize("obj", TYPES)
    def test_set_types(self, obj):
        assert StringSerializer().dumps(obj) == str(obj)

    def test_loads(self):
        assert StringSerializer().loads("hi") == "hi"


class TestPickleSerializer:
    @pytest.fixture
    def serializer(self):
        yield PickleSerializer(protocol=4)

    def test_init(self, serializer):
        assert isinstance(serializer, PickleSerializer)
        assert serializer.DEFAULT_ENCODING is None
        assert serializer.encoding is None
        assert serializer.protocol == 4

    def test_init_sets_default_protocol(self):
        serializer = PickleSerializer()
        assert serializer.protocol == pickle.DEFAULT_PROTOCOL

    @pytest.mark.parametrize("obj", TYPES)
    def test_set_types(self, obj, serializer):
        assert serializer.loads(serializer.dumps(obj)) == obj

    def test_dumps(self, serializer):
        expected = b"\x80\x04\x95\x06\x00\x00\x00\x00\x00\x00\x00\x8c\x02hi\x94."
        assert serializer.dumps("hi") == expected

    def test_dumps_with_none(self, serializer):
        assert isinstance(serializer.dumps(None), bytes)

    def test_loads(self, serializer):
        assert serializer.loads(b"\x80\x03X\x02\x00\x00\x00hiq\x00.") == "hi"

    def test_loads_with_none(self, serializer):
        assert serializer.loads(None) is None

    def test_dumps_and_loads(self, serializer):
        obj = Dummy(1, 2)
        assert serializer.loads(serializer.dumps(obj)) == obj


class TestJsonSerializer:
    def test_init(self):
        serializer = JsonSerializer()
        assert isinstance(serializer, BaseSerializer)
        assert serializer.DEFAULT_ENCODING == "utf-8"
        assert serializer.encoding == "utf-8"

    @pytest.mark.parametrize("obj", JSON_TYPES)
    def test_set_types(self, obj):
        serializer = JsonSerializer()
        assert serializer.loads(serializer.dumps(obj)) == obj

    def test_dumps(self):
        assert (
            JsonSerializer().dumps({"hi": 1}) == '{"hi": 1}'
            or JsonSerializer().dumps({"hi": 1}) == '{"hi":1}'  # json
        )  # ujson

    def test_dumps_with_none(self):
        assert JsonSerializer().dumps(None) == "null"

    def test_loads_with_null(self):
        assert JsonSerializer().loads("null") is None

    def test_loads_with_none(self):
        assert JsonSerializer().loads(None) is None

    def test_dumps_and_loads(self):
        obj = {"hi": 1}
        serializer = JsonSerializer()
        assert serializer.loads(serializer.dumps(obj)) == obj


class TestMsgPackSerializer:
    def test_init(self):
        serializer = MsgPackSerializer()
        assert isinstance(serializer, BaseSerializer)
        assert serializer.DEFAULT_ENCODING is None
        assert serializer.encoding is None

    def test_init_fails_if_msgpack_not_installed(self):
        with mock.patch("aiocache.serializers.serializers.msgpack", None):
            with pytest.raises(RuntimeError):
                MsgPackSerializer()
            assert JsonSerializer(), "Other serializers should still initialize"

    def test_init_use_list(self):
        serializer = MsgPackSerializer(use_list=True)
        assert serializer.use_list is True

    @pytest.mark.parametrize("obj", JSON_TYPES)
    def test_set_types(self, obj):
        serializer = MsgPackSerializer()
        assert serializer.loads(serializer.dumps(obj)) == obj

    def test_dumps(self):
        assert MsgPackSerializer().dumps("hi") == b"\xa2hi"

    def test_dumps_with_none(self):
        assert isinstance(MsgPackSerializer().dumps(None), bytes)

    def test_loads(self):
        assert MsgPackSerializer().loads(b"\xa2hi") == "hi"

    def test_loads_no_encoding(self):
        # encoding=None means the backend passes raw bytes; msgpack still decodes strings as str
        assert MsgPackSerializer(encoding=None).loads(b"\xa2hi") == "hi"

    def test_loads_with_none(self):
        assert MsgPackSerializer().loads(None) is None

    def test_dumps_and_loads_tuple(self):
        serializer = MsgPackSerializer()
        assert serializer.loads(serializer.dumps(Dummy(1, 2))) == [1, 2]

    def test_dumps_and_loads_dict(self):
        serializer = MsgPackSerializer()
        d = {"a": [1, 2, ("1", 2)], "b": {"b": 1, "c": [1, 2]}}
        assert serializer.loads(serializer.dumps(d)) == {
            "a": [1, 2, ["1", 2]],
            "b": {"b": 1, "c": [1, 2]},
        }


class TestVersionedSerializer:
    def test_wraps_current_value(self):
        serializer = VersionedSerializer(version=2)

        raw = serializer.dumps("value")

        assert raw == {
            "__aiocache_format__": "aiocache.versioned",
            "version": 2,
            "payload": "value",
        }
        assert serializer.loads(raw) == "value"

    def test_inner_serializer_is_used_for_envelope(self):
        serializer = VersionedSerializer(JsonSerializer(), version=2)

        raw = serializer.dumps("value")

        assert isinstance(raw, str)
        assert serializer.loads(raw) == "value"

    def test_reads_legacy_unwrapped_payload(self):
        serializer = VersionedSerializer(
            version=2,
            migrations={1: lambda payload: f"{payload}-v2"},
        )

        assert serializer.loads("legacy") == "legacy-v2"

    def test_migrates_multiple_versions(self):
        serializer = VersionedSerializer(
            version=3,
            migrations={
                1: lambda payload: payload + 1,
                2: lambda payload: payload * 2,
            },
        )
        raw = {
            "__aiocache_format__": "aiocache.versioned",
            "version": 1,
            "payload": 1,
        }

        assert serializer.loads(raw) == 4

    def test_explicit_migration_registry_skips_version(self):
        serializer = VersionedSerializer(
            version=3,
            migrations={(1, 3): lambda payload: payload + 3},
        )

        assert serializer.loads(serializer._make_envelope(1, 1)) == 4
        assert serializer.loads(serializer._make_envelope(3, 1)) == 1

    async def test_supports_async_migration_functions(self):
        async def migrate(payload):
            return payload + 1

        serializer = VersionedSerializer(version=2, migrations={1: migrate})
        raw = serializer._make_envelope(1, 1)

        assert await serializer.aloads(raw) == 2

    def test_same_instance_reads_different_versions(self):
        serializer = VersionedSerializer(
            version=3,
            migrations={
                1: lambda payload: payload + 1,
                2: lambda payload: payload + 2,
            },
        )

        assert serializer.loads(serializer._make_envelope(1, 0)) == 3
        assert serializer.loads(serializer._make_envelope(2, 0)) == 2
        assert serializer.loads(serializer._make_envelope(3, 0)) == 0

    def test_incomplete_registry_is_rejected(self):
        with pytest.raises(MissingMigrationStep):
            VersionedSerializer(
                version=3,
                migrations={1: lambda payload: payload},
            )

    def test_missing_migration_step(self):
        serializer = VersionedSerializer(
            version=3,
            migrations={2: lambda payload: payload},
        )

        with pytest.raises(MissingMigrationStep):
            serializer.loads(serializer._make_envelope(1, None))

    def test_unknown_newer_version(self):
        serializer = VersionedSerializer(version=2)

        with pytest.raises(UnknownVersionError):
            serializer.loads(serializer._make_envelope(3, None))

    def test_registry_cycle_is_rejected(self):
        with pytest.raises(MigrationCycleError):
            VersionedSerializer(
                version=3,
                migrations={
                    1: (2, lambda payload: payload),
                    2: (1, lambda payload: payload),
                },
            )

    def test_indirect_registry_cycle_is_rejected(self):
        with pytest.raises(MigrationCycleError):
            VersionedSerializer(
                version=4,
                migrations={
                    1: (2, lambda payload: payload),
                    2: (3, lambda payload: payload),
                    3: (1, lambda payload: payload),
                },
            )

    def test_corrupted_envelope(self):
        serializer = VersionedSerializer(JsonSerializer(), version=2)

        with pytest.raises(CorruptedPayloadError):
            serializer.loads("not-json")

    def test_missing_payload_is_corrupted(self):
        serializer = VersionedSerializer(version=2)

        with pytest.raises(CorruptedPayloadError):
            serializer.loads(
                {
                    "__aiocache_format__": "aiocache.versioned",
                    "version": 2,
                }
            )

    def test_invalid_version_is_corrupted(self):
        serializer = VersionedSerializer(version=2)

        with pytest.raises(CorruptedPayloadError):
            serializer.loads(
                {
                    "__aiocache_format__": "aiocache.versioned",
                    "version": "2",
                    "payload": None,
                }
            )

    def test_failed_migration_is_corrupted_payload(self):
        def migrate(payload):
            raise ValueError("bad payload")

        serializer = VersionedSerializer(version=2, migrations={1: migrate})

        with pytest.raises(CorruptedPayloadError):
            serializer.loads(serializer._make_envelope(1, None))

    def test_migration_receives_a_copy(self):
        payload = {"values": [1]}

        def migrate(value):
            value["values"].append(2)
            return value

        serializer = VersionedSerializer(version=2, migrations={1: migrate})
        raw = serializer._make_envelope(1, payload)

        assert serializer.loads(raw) == {"values": [1, 2]}
        assert payload == {"values": [1]}
