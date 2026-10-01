import json

import pytest

from aiocache.serializers import (
    CorruptedPayloadError,
    InvalidMigrationError,
    MigrationCycleError,
    MissingMigrationError,
    UnknownVersionError,
    VersionedSerializer,
)


def envelope(version, payload):
    return json.dumps({"version": version, "payload": payload})


class TestVersionedSerializer:
    def test_dumps_wraps_current_version(self):
        serializer = VersionedSerializer()

        assert json.loads(serializer.dumps({"name": "cache"})) == {
            "version": 1,
            "payload": {"name": "cache"},
        }

    def test_loads_current_version_returns_payload(self):
        serializer = VersionedSerializer(current_version=2)

        assert serializer.loads(envelope(2, {"name": "cache"})) == {"name": "cache"}

    def test_multi_step_migration(self):
        serializer = VersionedSerializer(
            current_version=3,
            migrations={
                (1, 2): lambda payload: {"name": payload["old_name"]},
                (2, 3): lambda payload: {"label": payload["name"].upper()},
            },
        )

        assert serializer.loads(envelope(1, {"old_name": "cache"})) == {"label": "CACHE"}

    def test_adjacent_version_registry_and_version_alias(self):
        serializer = VersionedSerializer(
            version=2,
            migrations={1: lambda payload: payload + 1},
        )

        assert serializer.current_version == 2
        assert serializer.loads(envelope(1, 1)) == 2

    def test_direct_migration_skips_intermediate_versions(self):
        serializer = VersionedSerializer(
            current_version=3,
            migrations={(1, 3): lambda payload: payload + 2, (2, 3): lambda p: p},
        )

        assert serializer.loads(envelope(1, 1)) == 3

    async def test_async_migration(self):
        async def migrate(payload):
            return payload + 1

        serializer = VersionedSerializer(
            current_version=2,
            migrations={(1, 2): migrate},
        )

        assert await serializer.aloads(envelope(1, 1)) == 2

    async def test_async_callable_object_migration(self):
        class Migration:
            async def __call__(self, payload):
                return payload + 1

        serializer = VersionedSerializer(
            current_version=2,
            migrations={(1, 2): Migration()},
        )

        assert await serializer.aloads(envelope(1, 1)) == 2

    async def test_loads_returns_awaitable_for_async_migration(self):
        async def migrate(payload):
            return payload + 1

        serializer = VersionedSerializer(
            current_version=2,
            migrations={(1, 2): migrate},
        )

        assert await serializer.loads(envelope(1, 1)) == 2

    def test_same_serializer_can_read_different_versions(self):
        serializer = VersionedSerializer(
            current_version=3,
            migrations={
                (1, 2): lambda payload: payload * 2,
                (2, 3): lambda payload: payload + 1,
            },
        )

        assert serializer.loads(envelope(1, 2)) == 5
        assert serializer.loads(envelope(2, 4)) == 5
        assert serializer.loads(envelope(3, 5)) == 5

    def test_missing_migration_step(self):
        with pytest.raises(MissingMigrationError):
            VersionedSerializer(
                current_version=3,
                migrations={(1, 2): lambda payload: payload},
            )

    def test_unknown_new_version(self):
        serializer = VersionedSerializer(current_version=2)

        with pytest.raises(UnknownVersionError):
            serializer.loads(envelope(3, "future"))

    def test_corrupt_payload(self):
        serializer = VersionedSerializer()

        with pytest.raises(CorruptedPayloadError):
            serializer.loads("not-json")

    @pytest.mark.parametrize(
        "raw",
        [
            json.dumps({"payload": "value"}),
            json.dumps({"version": 1, "other": "value"}),
            json.dumps({"version": "1", "payload": "value"}),
            json.dumps([]),
            json.dumps("plain"),
        ],
    )
    def test_corrupt_envelope_shapes(self, raw):
        serializer = VersionedSerializer()

        with pytest.raises(CorruptedPayloadError):
            serializer.loads(raw)

    def test_failing_migration_raises_corruption(self):
        def migrate(payload):
            return payload["missing"]

        serializer = VersionedSerializer(
            current_version=2,
            migrations={(1, 2): migrate},
        )

        with pytest.raises(CorruptedPayloadError):
            serializer.loads(envelope(1, {}))

    def test_self_cycle_is_rejected_at_construction(self):
        with pytest.raises(MigrationCycleError):
            VersionedSerializer(
                current_version=3,
                migrations={
                    (1, 2): lambda payload: payload,
                    (2, 2): lambda payload: payload,
                },
            )

    def test_two_node_cycle_is_rejected_at_construction(self):
        with pytest.raises(MigrationCycleError):
            VersionedSerializer(
                current_version=3,
                migrations={
                    (1, 2): lambda payload: payload,
                    (2, 1): lambda payload: payload,
                },
            )

    def test_backwards_migration_is_invalid(self):
        with pytest.raises(InvalidMigrationError):
            VersionedSerializer(
                current_version=3,
                migrations={
                    (2, 1): lambda payload: payload,
                },
            )

    def test_unknown_target_is_rejected(self):
        with pytest.raises(UnknownVersionError):
            VersionedSerializer(
                current_version=2,
                migrations={(1, 3): lambda payload: payload},
            )

    def test_registry_values_are_not_imported(self):
        with pytest.raises(InvalidMigrationError):
            VersionedSerializer(current_version=2, migrations={(1, 2): "module:function"})

    def test_loads_none(self):
        assert VersionedSerializer().loads(None) is None
