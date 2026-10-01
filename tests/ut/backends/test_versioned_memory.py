import asyncio
import json

import pytest

from aiocache.backends.memory import SimpleMemoryCache
from aiocache.serializers import (
    CorruptedPayloadError,
    ReadRepairUnsupportedError,
    VersionedSerializer,
)
from ...utils import ConcreteBaseCache


def make_raw(version, payload):
    return json.dumps({"version": version, "payload": payload})


def load_raw(cache, key="key"):
    return json.loads(cache._cache[cache.build_key(key)])


class TestVersionedMemoryCache:
    async def test_read_repair_preserves_expiry_deadline(self):
        cache = SimpleMemoryCache(
            serializer=VersionedSerializer(
                current_version=2,
                migrations={(1, 2): lambda payload: payload.upper()},
                read_repair=True,
            )
        )
        key = cache.build_key("key")
        await cache._set(key, make_raw(1, "old"), ttl=100)
        deadline = cache._deadlines[key]

        assert await cache.get("key") == "OLD"
        assert cache._cache[key] == make_raw(2, "OLD")

        assert load_raw(cache) == {"version": 2, "payload": "OLD"}
        assert cache._entry_versions[key] == 1
        assert cache._deadlines[key] == deadline
        remaining = cache._remaining_ttl(key)
        assert 0 < remaining < 100

    async def test_read_repair_uses_unchanged_cache_key(self):
        cache = SimpleMemoryCache(
            namespace="ns:",
            serializer=VersionedSerializer(
                current_version=2,
                migrations={(1, 2): lambda payload: payload.upper()},
                read_repair=True,
            )
        )
        key = cache.build_key("key", namespace="ns:")
        await cache._set(key, make_raw(1, "old"))

        assert await cache.get("key", namespace="ns:") == "OLD"
        assert key in cache._cache
        assert "key" not in cache._cache

    async def test_concurrent_set_between_migration_and_conditional_write(self):
        migration_started = asyncio.Event()
        release_migration = asyncio.Event()
        release_mutation = asyncio.Event()

        async def migrate(payload):
            migration_started.set()
            await release_migration.wait()
            return payload.upper()

        cache = SimpleMemoryCache(
            serializer=VersionedSerializer(
                current_version=2,
                migrations={(1, 2): migrate},
                read_repair=True,
            )
        )
        key = cache.build_key("key")
        await cache._set(key, make_raw(1, "old"), ttl=100)

        async def concurrent_set():
            await release_mutation.wait()
            await cache.set("key", "new-value", ttl=5)

        get_task = asyncio.create_task(cache.get("key"))
        mutation_task = asyncio.create_task(concurrent_set())
        await migration_started.wait()

        release_migration.set()
        release_mutation.set()

        assert await get_task == "OLD"
        await mutation_task

        assert load_raw(cache) == {"version": 2, "payload": "new-value"}
        assert 0 < cache._remaining_ttl(key) <= 5

    async def test_concurrent_delete_between_migration_and_conditional_write(self):
        migration_started = asyncio.Event()
        release_migration = asyncio.Event()
        release_mutation = asyncio.Event()

        async def migrate(payload):
            migration_started.set()
            await release_migration.wait()
            return payload.upper()

        cache = SimpleMemoryCache(
            serializer=VersionedSerializer(
                current_version=2,
                migrations={(1, 2): migrate},
                read_repair=True,
            )
        )
        key = cache.build_key("key")
        await cache._set(key, make_raw(1, "old"), ttl=100)

        async def concurrent_delete():
            await release_mutation.wait()
            return await cache.delete("key")

        get_task = asyncio.create_task(cache.get("key"))
        mutation_task = asyncio.create_task(concurrent_delete())
        await migration_started.wait()

        release_migration.set()
        release_mutation.set()

        assert await get_task == "OLD"
        assert await mutation_task == 1
        assert key not in cache._cache
        assert key not in cache._deadlines

    async def test_multi_step_read_repair(self):
        async def first_step(payload):
            await asyncio.sleep(0)
            return {"name": payload["old_name"]}

        cache = SimpleMemoryCache(
            serializer=VersionedSerializer(
                current_version=3,
                migrations={
                    (1, 2): first_step,
                    (2, 3): lambda payload: {"label": payload["name"].upper()},
                },
                read_repair=True,
            )
        )
        key = cache.build_key("key")
        await cache._set(key, make_raw(1, {"old_name": "cache"}), ttl=100)
        deadline = cache._deadlines[key]

        assert await cache.get("key") == {"label": "CACHE"}
        assert load_raw(cache) == {"version": 3, "payload": {"label": "CACHE"}}
        assert cache._deadlines[key] == deadline

    async def test_zero_remaining_ttl_does_not_revive_entry(self):
        release_migration = asyncio.Event()

        async def migrate(payload):
            await release_migration.wait()
            return payload.upper()

        cache = SimpleMemoryCache(
            serializer=VersionedSerializer(
                current_version=2,
                migrations={(1, 2): migrate},
                read_repair=True,
            )
        )
        key = cache.build_key("key")
        await cache._set(key, make_raw(1, "old"), ttl=0.02)

        get_task = asyncio.create_task(cache.get("key"))
        await asyncio.sleep(0.01)
        release_migration.set()
        await asyncio.sleep(0.03)

        assert await get_task == "OLD"
        assert key not in cache._cache
        assert cache._remaining_ttl(key) is None

    async def test_expired_entry_is_not_repaired_before_timer_runs(self):
        cache = SimpleMemoryCache(
            serializer=VersionedSerializer(
                current_version=2,
                migrations={(1, 2): lambda payload: payload.upper()},
                read_repair=True,
            )
        )
        key = cache.build_key("expired")
        old_raw = make_raw(1, "old")
        await cache._set(key, old_raw, ttl=100)
        cache._deadlines[key] = 0

        result = await cache._read_repair(
            key,
            (old_raw, cache._entry_versions[key]),
            make_raw(2, "OLD"),
        )

        assert result == 0
        assert cache._cache[key] == old_raw
        assert cache._remaining_ttl(key) == 0

    async def test_failed_migration_does_not_modify_entry(self):
        def migrate(payload):
            raise ValueError("bad payload")

        cache = SimpleMemoryCache(
            serializer=VersionedSerializer(
                current_version=2,
                migrations={(1, 2): migrate},
                read_repair=True,
            )
        )
        key = cache.build_key("key")
        old_raw = make_raw(1, "old")
        await cache._set(key, old_raw, ttl=100)
        deadline = cache._deadlines[key]

        with pytest.raises(CorruptedPayloadError):
            await cache.get("key")

        assert cache._cache[key] == old_raw
        assert cache._deadlines[key] == deadline

    async def test_disabled_read_repair_only_converts_for_this_call(self):
        cache = SimpleMemoryCache(
            serializer=VersionedSerializer(
                current_version=2,
                migrations={(1, 2): lambda payload: payload.upper()},
            )
        )
        key = cache.build_key("key")
        old_raw = make_raw(1, "old")
        await cache._set(key, old_raw)

        assert await cache.get("key") == "OLD"
        assert cache._cache[key] == old_raw

    async def test_multi_get_reads_multiple_versions_with_same_serializer(self):
        cache = SimpleMemoryCache(
            serializer=VersionedSerializer(
                current_version=3,
                migrations={
                    (1, 2): lambda value: value * 2,
                    (2, 3): lambda value: value + 1,
                },
            )
        )
        await cache._set(cache.build_key("v1"), make_raw(1, 2))
        await cache._set(cache.build_key("v2"), make_raw(2, 4))
        await cache._set(cache.build_key("v3"), make_raw(3, 5))

        assert await cache.multi_get(["v1", "v2", "v3"]) == [5, 5, 5]

    def test_backend_without_repair_capability_is_rejected(self):
        with pytest.raises(ReadRepairUnsupportedError):
            ConcreteBaseCache(
                serializer=VersionedSerializer(
                    current_version=2,
                    migrations={(1, 2): lambda payload: payload},
                    read_repair=True,
                )
            )
