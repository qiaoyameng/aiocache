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
    SUPPORTS_TAGS = True

    # TODO(PY312): https://peps.python.org/pep-0692/
    def __init__(self, **kwargs):
        # Extract maxsize before passing kwargs to base class
        self.maxsize = kwargs.pop('maxsize', None)
        if "serializer" not in kwargs:
            kwargs["serializer"] = NullSerializer()
        super().__init__(**kwargs)

        self._cache: OrderedDict[str, object] = OrderedDict()
        self._handlers: dict[str, asyncio.TimerHandle] = {}
        self._tag_index: dict[str, set[str]] = {}
        self._key_tags: dict[str, set[str]] = {}
        self._tags_version = 0
        self._tag_invalidated_at: dict[str, int] = {}

    def get_tags_version(self) -> int:
        """Monotonic counter bumped on every tag invalidation.

        Used to detect that a computation started before an invalidation, so its
        result is not re-attached to the invalidated tags when written back.
        """
        return self._tags_version

    def _index_key(self, key: str, tags) -> None:
        tag_set = set(tags)
        if not tag_set:
            return
        self._key_tags[key] = tag_set
        for tag in tag_set:
            self._tag_index.setdefault(tag, set()).add(key)

    def _unindex_key(self, key: str) -> None:
        tags = self._key_tags.pop(key, None)
        if not tags:
            return
        for tag in tags:
            keys = self._tag_index.get(tag)
            if keys is not None:
                keys.discard(key)
                if not keys:
                    del self._tag_index[tag]

    async def invalidate_tag(self, tag):
        self._tags_version += 1
        self._tag_invalidated_at[tag] = self._tags_version
        removed = 0
        for key in list(self._tag_index.get(tag, ())):
            removed += self.__delete(key)
        return removed

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
            self._unindex_key(key)
            if key in self._handlers:
                self._handlers[key].cancel()
                del self._handlers[key]

    async def _get(self, key, encoding="utf-8", _conn=None):
        self._mark_accessed(key)
        return self._cache.get(key)

    async def _gets(self, key, encoding="utf-8", _conn=None):
        return await self._get(key, encoding=encoding, _conn=_conn)

    async def _multi_get(self, keys, encoding="utf-8", _conn=None):
        return [await self._get(key, encoding=encoding, _conn=_conn) for key in keys]

    async def _set(
        self, key, value, ttl=None, tags=None, _tags_version=None, _cas_token=None, _conn=None
    ):
        if _cas_token is not None and self._cache.get(key) != _cas_token:
            return 0

        if tags and _tags_version is not None:
            invalidated_at = self._tag_invalidated_at
            if any(invalidated_at.get(tag, 0) > _tags_version for tag in tags):
                # The computation that produced this value started before one of
                # its tags was invalidated; refuse to re-attach the stale result.
                return False

        if key in self._handlers:
            self._handlers[key].cancel()

        self._unindex_key(key)
        self._cache[key] = value
        self._cache.move_to_end(key)
        if tags:
            self._index_key(key, tags)

        if ttl:
            loop = asyncio.get_running_loop()
            self._handlers[key] = loop.call_later(ttl, self.__delete, key)

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
        else:
            self._cache = OrderedDict()
            self._handlers = {}
            self._tag_index = {}
            self._key_tags = {}
        return True

    async def _raw(self, command, *args, encoding="utf-8", _conn=None, **kwargs):
        return getattr(self._cache, command)(*args, **kwargs)

    async def _redlock_release(self, key, value):
        if self._cache.get(key) == value:
            return self.__delete(key)
        return 0

    def __delete(self, key):
        if self._cache.pop(key, None) is not None:
            self._unindex_key(key)
            handle = self._handlers.pop(key, None)
            if handle:
                handle.cancel()
            return 1

        return 0

    def build_key(self, key: str, namespace: Optional[str] = None) -> str:
        return self._str_build_key(key, namespace)

    @classmethod
    def parse_uri_path(cls, path):
        return {}
