import asyncio
from typing import Optional, OrderedDict

from aiocache.base import BaseCache
from aiocache.serializers import NullSerializer


class SimpleMemoryCache(BaseCache[str]):
    """
    Memory cache implementation with the following components as defaults:
        - serializer: :class:`aiocache.serializers.NullSerializer`
        - plugins: None
        - backend: dict

    Config options are:

    :param serializer: obj derived from :class:`aiocache.serializers.BaseSerializer`.
    :param plugins: list of :class:`aiocache.plugins.BasePlugin` derived classes.
    :param namespace: string to use as default prefix for the key used in all operations of
        the backend. Default is an empty string, "".
    :param timeout: int or float in seconds specifying maximum timeout for the operations to last.
        By default, its 5.
    :param maxsize: int maximum number of keys to store (None for unlimited)
    """

    NAME = "memory"

    # TODO(PY312): https://peps.python.org/pep-0692/
    def __init__(self, **kwargs):
        # Extract maxsize before passing kwargs to base class
        self.maxsize = kwargs.pop('maxsize', None)
        if "serializer" not in kwargs:
            kwargs["serializer"] = NullSerializer()
        super().__init__(**kwargs)

        self._cache: OrderedDict[str, object] = OrderedDict()
        self._handlers: dict[str, asyncio.TimerHandle] = {}

    def _mark_accessed(self, key: str) -> None:
        """Move key to end to mark as recently used."""
        if key in self._cache:
            self._cache.move_to_end(key)

    def _evict_if_needed(self) -> None:
        """Evict least recently used items if over maxsize."""
        if self.maxsize is None:
            return

        while len(self._cache) > self.maxsize:
            key, _ = self._cache.popitem(last=False)  # Remove LRU item
            self._cancel_handler(key)

    def _cancel_handler(self, key: str) -> None:
        handle = self._handlers.pop(key, None)
        if handle is not None:
            handle.cancel()

    def _schedule_expiry(self, key: str, ttl: float) -> asyncio.TimerHandle:
        loop = asyncio.get_running_loop()
        handle: asyncio.TimerHandle

        def _on_expiry() -> None:
            self.__expire(key, handle)

        handle = loop.call_later(ttl, _on_expiry)
        return handle

    async def _get(self, key, encoding="utf-8", _conn=None):
        self._mark_accessed(key)
        return self._cache.get(key)

    async def _gets(self, key, encoding="utf-8", _conn=None):
        return await self._get(key, encoding=encoding, _conn=_conn)

    async def _multi_get(self, keys, encoding="utf-8", _conn=None):
        return [await self._get(key, encoding=encoding, _conn=_conn) for key in keys]

    async def _set(self, key, value, ttl=None, _cas_token=None, _conn=None):
        if _cas_token is not None and self._cache.get(key) != _cas_token:
            return 0

        self._cancel_handler(key)

        self._cache[key] = value
        self._cache.move_to_end(key)

        if ttl:
            self._handlers[key] = self._schedule_expiry(key, ttl)

        # Evict the oldest items if over limit
        self._evict_if_needed()
        return True

    async def _multi_set(self, pairs, ttl=None, _conn=None):
        for key, value in pairs:
            await self._set(key, value, ttl=ttl)
        return True

    async def _add(self, key, value, ttl=None, _conn=None):
        if key in self._cache:
            raise ValueError(f"Key {key} already exists, use .set to update")
        return await self._set(key, value, ttl=ttl)

    async def _exists(self, key, _conn=None):
        return key in self._cache

    async def _increment(self, key, delta, _conn=None):
        if key not in self._cache:
            self._cache[key] = delta
        else:
            try:
                self._cache[key] = int(self._cache[key]) + delta
            except ValueError:
                raise TypeError("Value is not an integer") from None
        self._mark_accessed(key)
        return self._cache[key]

    async def _expire(self, key, ttl, _conn=None):
        if key not in self._cache:
            return False

        # Cancel existing timer
        self._cancel_handler(key)

        # Set new timer
        if ttl:
            self._handlers[key] = self._schedule_expiry(key, ttl)

        self._mark_accessed(key)
        return True

    async def _delete(self, key, _conn=None):
        return self.__delete(key)

    async def _clear(self, namespace=None, _conn=None):
        if namespace:
            for key in list(self._cache):
                if key.startswith(namespace):
                    self.__delete(key)
        else:
            for handle in self._handlers.values():
                handle.cancel()
            self._handlers.clear()
            self._cache.clear()
        return True

    async def _raw(self, command, *args, encoding="utf-8", _conn=None, **kwargs):
        return getattr(self._cache, command)(*args, **kwargs)

    async def _redlock_release(self, key, value):
        if self._cache.get(key) == value:
            return self.__delete(key)
        return 0

    def __delete(self, key):
        existed = key in self._cache
        value = self._cache.pop(key, None)
        if not existed and value is None:
            return 0

        handle = self._handlers.pop(key, None)
        if handle is not None:
            handle.cancel()
        return 1

    def __expire(self, key, handle):
        # A timer may only delete the entry generation that scheduled it. If the
        # handle is no longer registered for the key, this callback is stale
        # (the entry was cleared, overwritten, deleted or evicted meanwhile)
        # and must not touch the current value, even if the callback was
        # already queued in the event loop.
        if self._handlers.get(key) is not handle:
            return
        del self._handlers[key]
        self._cache.pop(key, None)

    def build_key(self, key: str, namespace: Optional[str] = None) -> str:
        return self._str_build_key(key, namespace)

    @classmethod
    def parse_uri_path(cls, path):
        return {}
