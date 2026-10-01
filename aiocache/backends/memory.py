import asyncio
from typing import Optional, OrderedDict

from aiocache.base import BaseCache, SENTINEL
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
    supports_tags = True

    # TODO(PY312): https://peps.python.org/pep-0692/
    def __init__(self, **kwargs):
        # Extract maxsize before passing kwargs to base class
        self.maxsize = kwargs.pop('maxsize', None)
        if "serializer" not in kwargs:
            kwargs["serializer"] = NullSerializer()
        super().__init__(**kwargs)

        self._cache: OrderedDict[str, object] = OrderedDict()
        self._handlers: dict[str, asyncio.TimerHandle] = {}
        self._tags: dict[str, set[str]] = {}
        self._key_tags: dict[str, set[str]] = {}
        self._tag_generation = 0

    def _remove_key_tags(self, key: str) -> None:
        tags = self._key_tags.pop(key, set())
        for tag in tags:
            tagged_keys = self._tags.get(tag)
            if tagged_keys is None:
                continue
            tagged_keys.discard(key)
            if not tagged_keys:
                del self._tags[tag]

    def _replace_key_tags(self, key: str, tags=SENTINEL) -> None:
        self._remove_key_tags(key)
        if tags is SENTINEL:
            return

        new_tags = set(tags)
        if new_tags:
            self._key_tags[key] = new_tags
            for tag in new_tags:
                self._tags.setdefault(tag, set()).add(key)

    def _evict_key(self, key: str) -> None:
        handle = self._handlers.pop(key, None)
        if handle is not None:
            handle.cancel()
        self._remove_key_tags(key)

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
            self._evict_key(key)

    async def _get(self, key, encoding="utf-8", _conn=None):
        self._mark_accessed(key)
        return self._cache.get(key)

    async def _gets(self, key, encoding="utf-8", _conn=None):
        return await self._get(key, encoding=encoding, _conn=_conn)

    async def _multi_get(self, keys, encoding="utf-8", _conn=None):
        return [await self._get(key, encoding=encoding, _conn=_conn) for key in keys]

    async def _set(
        self, key, value, ttl=None, tags=SENTINEL, _tag_generation=None,
        _cas_token=None, _conn=None
    ):
        if _tag_generation is not None and self._tag_generation > _tag_generation:
            return 0

        if _cas_token is not None and self._cache.get(key) != _cas_token:
            return 0

        if key in self._handlers:
            self._handlers[key].cancel()

        self._cache[key] = value
        self._cache.move_to_end(key)
        self._replace_key_tags(key, tags)

        if ttl:
            loop = asyncio.get_running_loop()
            self._handlers[key] = loop.call_later(ttl, self.__delete, key)

        # Evict the oldest items if over limit
        self._evict_if_needed()
        return True

    async def _multi_set(self, pairs, ttl=None, _conn=None):
        for key, value in pairs:
            await self._set(key, value, ttl=ttl, tags=())
        return True

    async def _add(self, key, value, ttl=None, tags=SENTINEL, _tag_generation=None, _conn=None):
        if _tag_generation is not None and self._tag_generation > _tag_generation:
            return 0

        if key in self._cache:
            raise ValueError(f"Key {key} already exists, use .set to update")
        if tags is SENTINEL and _tag_generation is None:
            return await self._set(key, value, ttl=ttl)
        return await self._set(
            key, value, ttl=ttl, tags=tags, _tag_generation=_tag_generation, _conn=_conn
        )

    async def _current_tag_generation(self, _conn=None):
        return self._tag_generation

    async def _invalidate(self, tag, _conn=None):
        self._tag_generation += 1
        keys = list(self._tags.pop(tag, set()))
        for key in keys:
            self.__delete(key)
        return len(keys)

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
        if key in self._handlers:
            self._handlers[key].cancel()

        # Set new timer
        if ttl:
            loop = asyncio.get_running_loop()
            self._handlers[key] = loop.call_later(ttl, self.__delete, key)

        self._mark_accessed(key)
        return True

    async def _delete(self, key, _conn=None):
        return self.__delete(key)

    async def _clear(self, namespace=None, _conn=None):
        if namespace:
            for key in list(self._cache):
                if key.startswith(namespace):
                    self.__delete(key)
            self._tag_generation += 1
        else:
            handlers = self._handlers
            self._cache = OrderedDict()
            self._handlers = {}
            self._tags = {}
            self._key_tags = {}
            self._tag_generation += 1
            if isinstance(handlers, dict):
                for handle in handlers.values():
                    handle.cancel()
        return True

    async def _raw(self, command, *args, encoding="utf-8", _conn=None, **kwargs):
        return getattr(self._cache, command)(*args, **kwargs)

    async def _redlock_release(self, key, value):
        if self._cache.get(key) == value:
            return self.__delete(key)
        return 0

    def __delete(self, key):
        if self._cache.pop(key, None) is not None:
            handle = self._handlers.pop(key, None)
            if handle:
                handle.cancel()
            self._remove_key_tags(key)
            return 1

        return 0

    def build_key(self, key: str, namespace: Optional[str] = None) -> str:
        return self._str_build_key(key, namespace)

    @classmethod
    def parse_uri_path(cls, path):
        return {}
