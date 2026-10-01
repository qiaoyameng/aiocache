import asyncio
import pickle
from unittest.mock import ANY, AsyncMock, MagicMock, create_autospec, patch

import pytest

from aiocache.backends.memory import SimpleMemoryCache
from aiocache.base import BaseCache
from aiocache.serializers import (
    NullSerializer,
    PickleSerializer,
    ReadRepairUnsupportedError,
    VersionedSerializer,
)
from ...utils import ConcreteBaseCache, Keys


@pytest.fixture
def memory(mocker):
    memory = SimpleMemoryCache()
    mocker.spy(memory, "_cache")
    return memory


class TestSimpleMemoryCache:
    async def test_get(self, memory):
        await memory._get(Keys.KEY)
        memory._cache.get.assert_called_with(Keys.KEY)

    async def test_gets(self, mocker, memory):
        mocker.spy(memory, "_get")
        await memory._gets(Keys.KEY)
        memory._get.assert_called_with(Keys.KEY, encoding="utf-8", _conn=ANY)

    async def test_set(self, memory):
        await memory._set(Keys.KEY, "value")
        memory._cache.__setitem__.assert_called_with(Keys.KEY, "value")

    async def test_set_no_ttl_no_handle(self, memory):
        await memory._set(Keys.KEY, "value", ttl=0)
        assert Keys.KEY not in memory._handlers

        await memory._set(Keys.KEY, "value")
        assert Keys.KEY not in memory._handlers

    async def test_set_cancel_previous_ttl_handle(self, memory):
        with patch("asyncio.get_running_loop", autospec=True):
            await memory._set(Keys.KEY, "value", ttl=0.1)
            memory._handlers[Keys.KEY].cancel.assert_not_called()

            await memory._set(Keys.KEY, "new_value", ttl=0.1)
            memory._handlers[Keys.KEY].cancel.assert_called_once_with()

    async def test_set_ttl_handle(self, memory):
        await memory._set(Keys.KEY, "value", ttl=100)
        assert Keys.KEY in memory._handlers
        assert isinstance(memory._handlers[Keys.KEY], asyncio.Handle)

    async def test_set_cas_token(self, memory):
        memory._cache.get.return_value = "old_value"
        assert await memory._set(Keys.KEY, "value", _cas_token="old_value") == 1
        memory._cache.__setitem__.assert_called_with(Keys.KEY, "value")

    async def test_set_cas_fail(self, memory):
        memory._cache.get.return_value = "value"
        assert await memory._set(Keys.KEY, "value", _cas_token="old_value") == 0
        assert memory._cache.__setitem__.call_count == 0

    async def test_multi_get(self, memory):
        await memory._multi_get([Keys.KEY, Keys.KEY_1])
        memory._cache.get.assert_any_call(Keys.KEY)
        memory._cache.get.assert_any_call(Keys.KEY_1)

    async def test_multi_set(self, memory):
        await memory._multi_set([(Keys.KEY, "value"), (Keys.KEY_1, "random")])
        memory._cache.__setitem__.assert_any_call(Keys.KEY, "value")
        memory._cache.__setitem__.assert_any_call(Keys.KEY_1, "random")

    async def test_add(self, memory, mocker):
        mocker.spy(memory, "_set")
        await memory._add(Keys.KEY, "value")
        memory._set.assert_called_with(Keys.KEY, "value", ttl=None)

    async def test_add_existing(self, memory):
        memory._cache.__contains__.return_value = True
        with pytest.raises(ValueError):
            await memory._add(Keys.KEY, "value")

    async def test_exists(self, memory):
        await memory._exists(Keys.KEY)
        memory._cache.__contains__.assert_called_with(Keys.KEY)

    async def test_increment(self, memory):
        await memory._increment(Keys.KEY, 2)
        memory._cache.__contains__.assert_called_with(Keys.KEY)
        memory._cache.__setitem__.assert_called_with(Keys.KEY, 2)

    async def test_increment_missing(self, memory):
        memory._cache.__contains__.return_value = True
        memory._cache.__getitem__.return_value = 2
        await memory._increment(Keys.KEY, 2)
        memory._cache.__getitem__.assert_called_with(Keys.KEY)
        memory._cache.__setitem__.assert_called_with(Keys.KEY, 4)

    async def test_increment_typerror(self, memory):
        memory._cache.__contains__.return_value = True
        memory._cache.__getitem__.return_value = "asd"
        with pytest.raises(TypeError):
            await memory._increment(Keys.KEY, 2)

    async def test_expire_no_handle_no_ttl(self, memory):
        memory._cache.__contains__.return_value = True
        await memory._expire(Keys.KEY, 0)
        assert memory._handlers.get(Keys.KEY) is None

    async def test_expire_no_handle_ttl(self, memory):
        memory._cache.__contains__.return_value = True
        await memory._expire(Keys.KEY, 1)
        assert isinstance(memory._handlers.get(Keys.KEY), asyncio.Handle)

    async def test_expire_handle_ttl(self, memory):
        fake = create_autospec(asyncio.TimerHandle, instance=True)
        memory._handlers[Keys.KEY] = fake
        memory._cache.__contains__.return_value = True
        await memory._expire(Keys.KEY, 1)
        assert fake.cancel.call_count == 1
        assert isinstance(memory._handlers.get(Keys.KEY), asyncio.Handle)

    async def test_expire_missing(self, memory):
        memory._cache.__contains__.return_value = False
        assert await memory._expire(Keys.KEY, 1) is False

    async def test_delete(self, memory):
        fake = create_autospec(asyncio.TimerHandle, instance=True)
        memory._handlers[Keys.KEY] = fake
        await memory._delete(Keys.KEY)
        assert fake.cancel.call_count == 1
        assert Keys.KEY not in memory._handlers
        memory._cache.pop.assert_called_with(Keys.KEY, None)

    async def test_delete_missing(self, memory):
        memory._cache.pop.return_value = None
        await memory._delete(Keys.KEY)
        memory._cache.pop.assert_called_with(Keys.KEY, None)

    async def test_delete_non_truthy(self, memory):
        non_truthy = MagicMock(spec_set=("__bool__",))
        non_truthy.__bool__.side_effect = ValueError("Does not implement truthiness")

        with pytest.raises(ValueError):
            bool(non_truthy)

        memory._cache.pop.return_value = non_truthy
        await memory._delete(Keys.KEY)

        assert non_truthy.__bool__.call_count == 1
        memory._cache.pop.assert_called_with(Keys.KEY, None)

    async def test_clear_namespace(self, memory):
        memory._cache.__iter__.return_value = iter(["nma", "nmb", "no"])
        await memory._clear("nm")
        assert memory._cache.pop.call_count == 2
        memory._cache.pop.assert_any_call("nma", None)
        memory._cache.pop.assert_any_call("nmb", None)

    async def test_clear_no_namespace(self, memory):
        memory._handlers = "asdad"
        memory._cache = "asdad"
        await memory._clear()
        memory._handlers = {}
        memory._cache = {}

    async def test_raw(self, memory):
        await memory._raw("get", Keys.KEY)
        memory._cache.get.assert_called_with(Keys.KEY)

        await memory._set(Keys.KEY, "value")
        memory._cache.__setitem__.assert_called_with(Keys.KEY, "value")

    async def test_redlock_release(self, memory):
        memory._cache.get.return_value = "lock"
        fake = create_autospec(asyncio.TimerHandle, instance=True)
        memory._handlers[Keys.KEY] = fake
        assert await memory._redlock_release(Keys.KEY, "lock") == 1
        memory._cache.get.assert_called_with(Keys.KEY)
        memory._cache.pop.assert_called_with(Keys.KEY, None)
        assert fake.cancel.call_count == 1
        assert Keys.KEY not in memory._handlers

    async def test_redlock_release_nokey(self, memory):
        memory._cache.get.return_value = None
        assert await memory._redlock_release(Keys.KEY, "lock") == 0
        memory._cache.get.assert_called_with(Keys.KEY)
        assert memory._cache.pop.call_count == 0

    def test_name(self):
        assert SimpleMemoryCache.NAME == "memory"

    def test_inheritance(self):
        assert isinstance(SimpleMemoryCache(), BaseCache)

    def test_default_serializer(self):
        assert isinstance(SimpleMemoryCache().serializer, NullSerializer)

    def test_parse_uri_path(self):
        assert SimpleMemoryCache().parse_uri_path("/1/2/3") == {}

    async def test_maxsize_enforcement(self):
        cache = SimpleMemoryCache(maxsize=2)

        # Fill cache to maxsize
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        assert await cache.exists("key1")
        assert await cache.exists("key2")
        assert len(cache._cache) == 2

        # Add third item - should evict key1 (LRU)
        await cache.set("key3", "value3")
        assert not await cache.exists("key1")
        assert await cache.exists("key2")
        assert await cache.exists("key3")
        assert len(cache._cache) == 2

    async def test_eviction_order(self):
        cache = SimpleMemoryCache(maxsize=3)

        # Add initial items
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        # Access key1 to make it recently used
        await cache.get("key1")

        # Add fourth item - should evict key2 (least recently used)
        await cache.set("key4", "value4")
        assert await cache.exists("key1")
        assert not await cache.exists("key2")
        assert await cache.exists("key3")
        assert await cache.exists("key4")

    async def test_get_updates_recency(self):
        cache = SimpleMemoryCache(maxsize=2)

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        # Access key1 to make it recently used
        await cache.get("key1")

        # Add third item - should evict key2
        await cache.set("key3", "value3")
        assert await cache.exists("key1")
        assert not await cache.exists("key2")
        assert await cache.exists("key3")

    async def test_add_eviction(self):
        cache = SimpleMemoryCache(maxsize=2)

        await cache.add("key1", "value1")
        await cache.add("key2", "value2")

        # Add third item - should evict key1
        await cache.add("key3", "value3")
        assert not await cache.exists("key1")
        assert await cache.exists("key2")
        assert await cache.exists("key3")

    async def test_multi_set_eviction(self):
        cache = SimpleMemoryCache(maxsize=3)

        # Initial items
        await cache.set("key1", "value1")

        # Add two more items via multi_set
        await cache.multi_set([("key2", "value2"), ("key3", "value3")])

        # Add two more - should evict two oldest items
        await cache.multi_set([("key4", "value4"), ("key5", "value5")])

        assert not await cache.exists("key1")
        assert not await cache.exists("key2")
        assert await cache.exists("key3")  # key3 was most recent before multi_set
        assert await cache.exists("key4")
        assert await cache.exists("key5")

    async def test_no_eviction_when_below_maxsize(self):
        cache = SimpleMemoryCache(maxsize=5)

        for i in range(4):
            await cache.set(f"key{i}", f"value{i}")

        assert len(cache._cache) == 4
        for i in range(4):
            assert await cache.exists(f"key{i}")

    async def test_ttl_doesnt_affect_eviction_order(self):
        cache = SimpleMemoryCache(maxsize=2)

        await cache.set("key1", "value1", ttl=10)
        await cache.set("key2", "value2")

        # Access key1 to make it recently used
        await cache.get("key1")

        # Add third item - should evict key2 despite key1 having TTL
        await cache.set("key3", "value3")
        assert await cache.exists("key1")
        assert not await cache.exists("key2")
        assert await cache.exists("key3")

    async def test_clear_resets_size(self):
        cache = SimpleMemoryCache(maxsize=2)

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        assert len(cache._cache) == 2

        await cache.clear()
        assert len(cache._cache) == 0

        # Should be able to add new items without immediate eviction
        await cache.set("key3", "value3")
        await cache.set("key4", "value4")
        assert await cache.exists("key3")
        assert await cache.exists("key4")
        assert len(cache._cache) == 2

    async def test_delete_updates_size(self):
        cache = SimpleMemoryCache(maxsize=2)

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        assert len(cache._cache) == 2

        await cache.delete("key1")
        assert len(cache._cache) == 1

        # Should be able to add new item without eviction
        await cache.set("key3", "value3")
        assert await cache.exists("key2")
        assert await cache.exists("key3")
        assert len(cache._cache) == 2

    async def test_increment_updates_recency(self):
        cache = SimpleMemoryCache(maxsize=2)

        await cache.set("key1", 1)
        await cache.set("key2", 2)

        # Increment key1 to make it recently used
        await cache.increment("key1", 1)

        # Add third item - should evict key2
        await cache.set("key3", 3)
        assert await cache.exists("key1")
        assert not await cache.exists("key2")
        assert await cache.exists("key3")
        assert await cache.get("key1") == 2

    async def test_lru_eviction_with_handlers(self):
        cache = SimpleMemoryCache(maxsize=2)

        # Add two items with TTL
        await cache.set("key1", "value1", ttl=10)
        await cache.set("key2", "value2", ttl=10)

        key2_handler = cache._handlers["key2"]

        # Add a third item to trigger eviction
        await cache.set("key3", "value3")

        assert "key1" not in cache._cache
        assert "key1" not in cache._handlers
        assert "key2" in cache._cache
        assert cache._handlers.get("key2") is key2_handler  # Same handler instance
        assert "key3" in cache._cache
        assert "key3" not in cache._handlers  # No TTL

    async def test_custom_serializer(self):
        value = {"one": {"nested": "1"}, "two": 2}
        serialized = pickle.dumps(value)

        # use the main constructor instead of fixture for coverage
        async with SimpleMemoryCache(serializer=PickleSerializer()) as smc:
            assert isinstance(smc.serializer, PickleSerializer)
            await smc.set(Keys.KEY, value)
            assert await smc.get(Keys.KEY) == pickle.loads(serialized)


class BarrierVersionedSerializer(VersionedSerializer):
    def __init__(self, *args, after_read=None, before_write=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._after_read_callback = after_read
        self._before_write_callback = before_write

    async def _after_raw_read(self, key, raw):
        if self._after_read_callback is not None:
            await self._after_read_callback(key, raw)

    async def _before_conditional_write(self, key, value):
        if self._before_write_callback is not None:
            await self._before_write_callback(key, value)


class TestVersionedReadRepair:
    async def test_read_repairs_old_value(self):
        serializer = VersionedSerializer(
            version=2,
            read_repair=True,
            migrations={1: lambda payload: f"{payload}-v2"},
        )
        cache = SimpleMemoryCache(serializer=serializer)
        cache._cache[Keys.KEY] = serializer._make_envelope(1, "old")

        assert await cache.get(Keys.KEY) == "old-v2"

        raw = cache._cache[Keys.KEY]
        assert raw["version"] == 2
        assert raw["payload"] == "old-v2"

    async def test_read_repair_keeps_cache_key(self):
        serializer = VersionedSerializer(version=2, read_repair=True)
        cache = SimpleMemoryCache(serializer=serializer)
        cache._cache["prefix-key"] = serializer._make_envelope(2, "value")

        assert await cache.get("key", namespace="prefix-") == "value"
        assert set(cache._cache) == {"prefix-key"}

    async def test_read_only_without_repair(self):
        serializer = VersionedSerializer(
            version=2,
            migrations={1: lambda payload: f"{payload}-v2"},
        )
        cache = SimpleMemoryCache(serializer=serializer)
        old_value = serializer._make_envelope(1, "old")
        cache._cache[Keys.KEY] = old_value

        assert await cache.get(Keys.KEY) == "old-v2"
        assert cache._cache[Keys.KEY] is old_value

    async def test_read_repairs_multiple_steps(self):
        serializer = VersionedSerializer(
            version=3,
            read_repair=True,
            migrations={
                1: lambda payload: payload + 1,
                2: lambda payload: payload * 2,
            },
        )
        cache = SimpleMemoryCache(serializer=serializer)
        cache._cache[Keys.KEY] = serializer._make_envelope(1, 1)

        assert await cache.get(Keys.KEY) == 4
        assert cache._cache[Keys.KEY] == serializer._make_envelope(3, 4)

    async def test_failed_migration_leaves_old_entry(self):
        def migrate(payload):
            raise ValueError("bad payload")

        serializer = VersionedSerializer(
            version=2,
            read_repair=True,
            migrations={1: migrate},
        )
        cache = SimpleMemoryCache(serializer=serializer)
        old_value = serializer._make_envelope(1, "old")
        cache._cache[Keys.KEY] = old_value

        with pytest.raises(ValueError):
            await cache.get(Keys.KEY)

        assert cache._cache[Keys.KEY] is old_value

    async def test_migration_does_not_mutate_stored_old_entry(self):
        def migrate(payload):
            payload["value"] = 2
            return payload

        serializer = VersionedSerializer(
            version=2,
            read_repair=True,
            migrations={1: migrate},
        )
        cache = SimpleMemoryCache(serializer=serializer)
        old_value = serializer._make_envelope(1, {"value": 1})
        cache._cache[Keys.KEY] = old_value

        await cache.get(Keys.KEY)

        assert old_value["payload"] == {"value": 1}

    async def test_read_repair_preserves_remaining_ttl(self):
        started = asyncio.Event()
        finished = asyncio.Event()

        async def migrate(payload):
            started.set()
            await finished.wait()
            return f"{payload}-v2"

        serializer = BarrierVersionedSerializer(
            version=2,
            read_repair=True,
            migrations={1: migrate},
        )
        cache = SimpleMemoryCache(serializer=serializer)
        cache._cache[Keys.KEY] = serializer._make_envelope(1, "old")
        cache._store(Keys.KEY, cache._cache[Keys.KEY], ttl=100)

        get_task = asyncio.create_task(cache.get(Keys.KEY))
        await started.wait()
        old_handler = cache._handlers[Keys.KEY]
        await asyncio.sleep(0.02)
        finished.set()

        assert await get_task == "old-v2"
        assert cache._handlers[Keys.KEY] is old_handler
        loop = asyncio.get_running_loop()
        remaining = cache._expires_at[Keys.KEY] - loop.time()
        assert remaining == pytest.approx(99.98, abs=0.05)

    async def test_zero_remaining_time_during_migration(self):
        started = asyncio.Event()
        finished = asyncio.Event()

        async def migrate(payload):
            started.set()
            await finished.wait()
            return payload

        serializer = BarrierVersionedSerializer(
            version=2,
            read_repair=True,
            migrations={1: migrate},
        )
        cache = SimpleMemoryCache(serializer=serializer)
        cache._cache[Keys.KEY] = serializer._make_envelope(1, "old")
        cache._store(Keys.KEY, cache._cache[Keys.KEY], ttl=0.01)

        get_task = asyncio.create_task(cache.get(Keys.KEY))
        await started.wait()
        await asyncio.sleep(0.025)
        assert Keys.KEY not in cache._cache
        finished.set()

        assert await get_task == "old"
        assert Keys.KEY not in cache._cache

    @pytest.mark.parametrize("concurrent_operation", ("set", "delete"))
    async def test_concurrent_change_between_migration_and_write(self, concurrent_operation):
        reached_write = asyncio.Event()
        release_write = asyncio.Event()
        write_attempts = []

        async def before_write(key, value):
            reached_write.set()
            write_attempts.append(await release_write.wait())

        serializer = BarrierVersionedSerializer(
            version=2,
            read_repair=True,
            migrations={1: lambda payload: f"{payload}-v2"},
            before_write=before_write,
        )
        cache = SimpleMemoryCache(serializer=serializer)
        cache._cache[Keys.KEY] = serializer._make_envelope(1, "old")
        cache._store(Keys.KEY, cache._cache[Keys.KEY], ttl=100)
        old_handler = cache._handlers[Keys.KEY]
        repair_results = []
        original_read_repair_set = cache._read_repair_set

        async def record_read_repair_set(*args, **kwargs):
            result = await original_read_repair_set(*args, **kwargs)
            repair_results.append(result)
            return result

        cache._read_repair_set = record_read_repair_set

        get_task = asyncio.create_task(cache.get(Keys.KEY))
        await reached_write.wait()

        if concurrent_operation == "set":
            await cache.set(Keys.KEY, "newer", ttl=1)
            expected_handler = cache._handlers[Keys.KEY]
            assert expected_handler is not old_handler
        else:
            await cache.delete(Keys.KEY)
            expected_handler = None

        release_write.set()
        assert await get_task == "old-v2"
        assert write_attempts == [True]
        assert repair_results == [0]

        if concurrent_operation == "set":
            assert cache._cache[Keys.KEY] == serializer._make_envelope(2, "newer")
            assert cache._handlers[Keys.KEY] is expected_handler
            loop = asyncio.get_running_loop()
            remaining = cache._expires_at[Keys.KEY] - loop.time()
            assert remaining == pytest.approx(1, abs=0.05)
        else:
            assert Keys.KEY not in cache._cache
            assert Keys.KEY not in cache._handlers

    @pytest.mark.parametrize("concurrent_operation", ("set", "delete"))
    async def test_concurrent_change_between_read_and_migration(self, concurrent_operation):
        reached_read = asyncio.Event()
        release_read = asyncio.Event()
        repair_results = []

        async def after_read(key, raw):
            reached_read.set()
            await release_read.wait()

        serializer = BarrierVersionedSerializer(
            version=2,
            read_repair=True,
            migrations={1: lambda payload: f"{payload}-v2"},
            after_read=after_read,
        )
        cache = SimpleMemoryCache(serializer=serializer)
        cache._cache[Keys.KEY] = serializer._make_envelope(1, "old")
        original_read_repair_set = cache._read_repair_set

        async def record_read_repair_set(*args, **kwargs):
            result = await original_read_repair_set(*args, **kwargs)
            repair_results.append(result)
            return result

        cache._read_repair_set = record_read_repair_set

        get_task = asyncio.create_task(cache.get(Keys.KEY))
        await reached_read.wait()

        if concurrent_operation == "set":
            await cache.set(Keys.KEY, "newer", ttl=1)
        else:
            await cache.delete(Keys.KEY)

        release_read.set()

        assert await get_task == "old-v2"
        assert repair_results == [0]

        if concurrent_operation == "set":
            loop = asyncio.get_running_loop()
            remaining = cache._expires_at[Keys.KEY] - loop.time()
            assert cache._cache[Keys.KEY] == serializer._make_envelope(2, "newer")
            assert remaining == pytest.approx(1, abs=0.05)
        else:
            assert Keys.KEY not in cache._cache

    async def test_read_repair_explicitly_unsupported_for_other_backends(self):
        serializer = VersionedSerializer(
            version=2,
            read_repair=True,
            migrations={1: lambda payload: payload},
        )
        cache = ConcreteBaseCache(serializer=serializer)

        with pytest.raises(ReadRepairUnsupportedError):
            await cache.get(Keys.KEY)

    async def test_other_backends_still_read_without_repair(self):
        serializer = VersionedSerializer(
            version=2,
            migrations={1: lambda payload: f"{payload}-v2"},
        )
        raw = serializer._make_envelope(1, "old")
        cache = ConcreteBaseCache(serializer=serializer)
        cache._get = AsyncMock(return_value=raw)

        assert await cache.get(Keys.KEY) == "old-v2"
