import asyncio
import functools
import inspect
import logging

from aiocache.base import SENTINEL
from aiocache.lock import RedLock

logger = logging.getLogger(__name__)

# Unique sentinel marking "key is missing in the backend" when ``cache_none``
# is enabled. It is a private object compared by identity, so it can never be
# confused with a value stored by the user (unlike a plain string, which the
# user could legitimately cache).
_CACHE_MISS = object()


class cached:
    """
    Caches the functions return value into a key generated with module_name, function_name
    and args. The cache is available in the function object as ``<function_name>.cache``.

    :param cache: cache instance to use when calling the ``set``/``get`` operations.
    :param ttl: int seconds to store the function call. Default is None which means no expiration.
    :param key_builder: Callable that allows to build the function dynamically. It receives
        the function plus same args and kwargs passed to the function.
        This behavior is necessarily different than ``BaseCache.build_key()``
    :param skip_cache_func: Callable that receives the result after calling the
        wrapped function and should return `True` if the value should skip the
        cache (or `False` to store in the cache).
        e.g. to avoid caching `None` results: `lambda r: r is None`
    :param noself: bool if you are decorating a class function, by default self is also used to
        generate the key. This will result in same function calls done by different class instances
        to use different cache keys. Use noself=True if you want to ignore it.
    :param cache_none: bool to also cache ``None`` results. When enabled, a stored
        ``None`` is distinguished from a missing key (via an extra ``exists`` check),
        so calls whose result was ``None`` are served from the cache while the ttl
        is valid. Disabled by default, which keeps the historical behavior of
        re-calling the function on ``None`` results.
    """

    def __init__(
        self,
        cache,
        *,
        ttl=SENTINEL,
        key_builder=None,
        skip_cache_func=lambda x: False,
        noself=False,
        cache_none=False,
    ):
        self.ttl = ttl
        self.key_builder = key_builder
        self.skip_cache_func = skip_cache_func
        self.noself = noself
        self.cache_none = cache_none
        self.cache = cache

    def __call__(self, f):
        @functools.wraps(f)
        async def wrapper(*args, **kwargs):
            return await self.decorator(f, *args, **kwargs)

        wrapper.cache = self.cache
        return wrapper

    async def decorator(
        self, f, *args, cache_read=True, cache_write=True, aiocache_wait_for_write=True, **kwargs
    ):
        key = self.get_cache_key(f, args, kwargs)

        if cache_read:
            value = await self.get_from_cache(key)
            if self._is_cache_hit(value):
                return value

        result = await f(*args, **kwargs)

        if self.skip_cache_func(result):
            return result

        if cache_write:
            if aiocache_wait_for_write:
                await self.set_in_cache(key, result)
            else:
                # TODO: Use aiojobs to avoid warnings.
                asyncio.create_task(self.set_in_cache(key, result))

        return result

    def _is_cache_hit(self, value):
        if self.cache_none:
            return value is not _CACHE_MISS
        return value is not None

    def get_cache_key(self, f, args, kwargs):
        if self.key_builder:
            return self.key_builder(f, *args, **kwargs)

        return self._key_from_args(f, args, kwargs)

    def _key_from_args(self, func, args, kwargs):
        ordered_kwargs = sorted(kwargs.items())
        return (
            (func.__module__ or "")
            + func.__name__
            + str(args[1:] if self.noself else args)
            + str(ordered_kwargs)
        )

    async def get_from_cache(self, key):
        try:
            if self.cache_none:
                value = await self.cache.get(key, default=_CACHE_MISS)
                if value is _CACHE_MISS and await self.cache.exists(key):
                    # The key holds an actual None (either stored as is or
                    # obtained after deserializing), not a missing key.
                    return None
                return value
            return await self.cache.get(key)
        except Exception:
            logger.exception("Couldn't retrieve %s, unexpected error", key)
        return _CACHE_MISS if self.cache_none else None

    async def set_in_cache(self, key, value):
        try:
            await self.cache.set(key, value, ttl=self.ttl)
        except Exception:
            logger.exception("Couldn't set %s in key %s, unexpected error", value, key)


class cached_stampede(cached):
    """
    Caches the functions return value into a key generated with module_name, function_name and args
    while avoids for cache stampede effects.

    :param cache: cache instance to use when calling the ``set``/``get`` operations.
        Default is :class:`aiocache.SimpleMemoryCache`.
    :param lease: int seconds to lock function call to avoid cache stampede effects.
        If 0 or None, no locking happens (default is 2). valkey and memory backends support
        float ttls
    :param ttl: int seconds to store the function call. Default is None which means no expiration.
    :param key_from_attr: str arg or kwarg name from the function to use as a key.
    :param key_builder: Callable that allows to build the function dynamically. It receives
        the function plus same args and kwargs passed to the function.
        This behavior is necessarily different than ``BaseCache.build_key()``
    :param skip_cache_func: Callable that receives the result after calling the
        wrapped function and should return `True` if the value should skip the
        cache (or `False` to store in the cache).
        e.g. to avoid caching `None` results: `lambda r: r is None`
    :param noself: bool if you are decorating a class function, by default self is also used to
        generate the key. This will result in same function calls done by different class instances
        to use different cache keys. Use noself=True if you want to ignore it.
    :param cache_none: bool to also cache ``None`` results. See :class:`cached`.
    """

    def __init__(self, cache, lease=2, **kwargs):
        super().__init__(cache, **kwargs)
        self.lease = lease

    async def decorator(self, f, *args, **kwargs):
        key = self.get_cache_key(f, args, kwargs)

        value = await self.get_from_cache(key)
        if self._is_cache_hit(value):
            return value

        async with RedLock(self.cache, key, self.lease):
            value = await self.get_from_cache(key)
            if self._is_cache_hit(value):
                return value

            result = await f(*args, **kwargs)

            if self.skip_cache_func(result):
                return result

            await self.set_in_cache(key, result)

        return result


def _get_args_dict(func, args, kwargs):
    defaults = {
        arg_name: arg.default
        for arg_name, arg in inspect.signature(func).parameters.items()
        if arg.default is not inspect._empty  # TODO: bug prone..
    }
    args_names = func.__code__.co_varnames[: func.__code__.co_argcount]
    return {**defaults, **dict(zip(args_names, args)), **kwargs}


class multi_cached:
    """
    Only supports functions that return dict-like structures. This decorator caches each key/value
    of the dict-like object returned by the function. The dict keys of the returned data should
    match the set of keys that are passed to the decorated callable in an iterable object.
    The name of that argument is passed to this decorator via the parameter
    ``keys_from_attr``. ``keys_from_attr`` can be the name of a positional or keyword argument.

    If the argument specified by ``keys_from_attr`` is an empty list, the cache will be ignored
    and the function will be called. If only some of the keys in ``keys_from_attr``are cached
    (and ``cache_read`` is True) those values will be fetched from the cache, and only the
    uncached keys will be passed to the callable via the argument specified by ``keys_from_attr``.

    By default, the callable's name and call signature are not incorporated into the cache key,
    so if there is another cached function returning a dict with same keys, those keys will be
    overwritten. To avoid this, use a specific ``namespace`` in each cache decorator or pass a
    ``key_builder``.

    If ``key_builder`` is passed, then the values of ``keys_from_attr`` will be transformed
    before requesting them from the cache. Equivalently, the keys in the dict-like mapping
    returned by the decorated callable will be transformed before storing them in the cache.

    The cache is available in the function object as ``<function_name>.cache``.

    Extra args that are injected in the function that you can use to control the cache
    behavior are:

        - ``cache_read``: Controls whether the function call will try to read from cache first or
                          not. Enabled by default.
        - ``cache_write``: Controls whether the function call will try to write in the cache once
                           the result has been retrieved. Enabled by default.
        - ``aiocache_wait_for_write``: Controls whether the call of the function will wait for the
                                       value in the cache to be written. If set to False, the write
                                       happens in the background. Enabled by default

    :param cache: cache instance to use when calling the ``multi_set``/``multi_get`` operations.
    :param keys_from_attr: name of the arg or kwarg in the decorated callable that contains
        an iterable that yields the keys returned by the decorated callable.
    :param key_builder: Callable that enables mapping the decorated function's keys to the keys
        used by the cache. Receives a key from the iterable corresponding to
        ``keys_from_attr``, the decorated callable, and the positional and keyword arguments
        that were passed to the decorated callable. This behavior is necessarily different than
        ``BaseCache.build_key()`` and the call signature differs from ``cached.key_builder``.
    :param skip_cache_func: Callable that receives both key and value and returns True
        if that key-value pair should not be cached (or False to store in cache).
        The keys and values to be passed are taken from the wrapped function result.
    :param ttl: int seconds to store the keys. Default is 0 which means no expiration.
    :param cache_none: bool to also cache ``None`` values. When enabled, a stored
        ``None`` is distinguished from a missing key (via an extra ``exists`` check),
        so keys whose value was ``None`` are served from the cache while the ttl is
        valid and are kept in the returned mapping. Disabled by default, which keeps
        the historical behavior of treating ``None`` values as missing keys.
    """

    def __init__(
        self,
        cache=None,
        *,
        keys_from_attr,
        key_builder=None,
        skip_cache_func=lambda k, v: False,
        ttl=SENTINEL,
        cache_none=False,
    ):
        self.cache = cache
        self.keys_from_attr = keys_from_attr
        self.key_builder = key_builder or (lambda key, f, *args, **kwargs: key)
        self.skip_cache_func = skip_cache_func
        self.ttl = ttl
        self.cache_none = cache_none

    def __call__(self, f):
        @functools.wraps(f)
        async def wrapper(*args, **kwargs):
            return await self.decorator(f, *args, **kwargs)

        wrapper.cache = self.cache
        return wrapper

    async def decorator(
        self, f, *args, cache_read=True, cache_write=True, aiocache_wait_for_write=True, **kwargs
    ):
        missing_keys = []
        partial = {}
        orig_keys, cache_keys, new_args, args_index = self.get_cache_keys(f, args, kwargs)

        if cache_read:
            values = await self.get_from_cache(*cache_keys)
            for orig_key, value in zip(orig_keys, values):
                if self._is_cache_miss(value):
                    missing_keys.append(orig_key)
                else:
                    partial[orig_key] = value
            if values and not any(self._is_cache_miss(value) for value in values):
                return partial
        else:
            missing_keys = list(orig_keys)

        if args_index > -1:
            new_args[args_index] = missing_keys
        else:
            kwargs[self.keys_from_attr] = missing_keys

        result = await f(*new_args, **kwargs)
        result.update(partial)

        to_cache = {k: v for k, v in result.items() if not self.skip_cache_func(k, v)}

        if not to_cache:
            return result

        if cache_write:
            if aiocache_wait_for_write:
                await self.set_in_cache(to_cache, f, args, kwargs)
            else:
                # TODO: Use aiojobs to avoid warnings.
                asyncio.create_task(self.set_in_cache(to_cache, f, args, kwargs))

        return result

    def get_cache_keys(self, f, args, kwargs):
        args_dict = _get_args_dict(f, args, kwargs)
        orig_keys = args_dict.get(self.keys_from_attr, []) or []
        cache_keys = [self.key_builder(key, f, *args, **kwargs) for key in orig_keys]

        args_names = f.__code__.co_varnames[: f.__code__.co_argcount]
        new_args = list(args)
        keys_index = -1
        if self.keys_from_attr in args_names and self.keys_from_attr not in kwargs:
            keys_index = args_names.index(self.keys_from_attr)

        return orig_keys, cache_keys, new_args, keys_index

    def _is_cache_miss(self, value):
        if self.cache_none:
            return value is _CACHE_MISS
        return value is None

    async def get_from_cache(self, *keys):
        if not keys:
            return []
        try:
            values = await self.cache.multi_get(keys)
            if self.cache_none:
                values = await self._distinguish_missing(keys, values)
            return values
        except Exception:
            logger.exception("Couldn't retrieve %s, unexpected error", keys)
            return [_CACHE_MISS if self.cache_none else None] * len(keys)

    async def _distinguish_missing(self, keys, values):
        # multi_get reports None both for missing keys and for keys holding an
        # actual None (stored as is or obtained after deserializing). Resolve
        # the ambiguity with an exists check on the None-valued keys only.
        values = list(values)
        none_indexes = [index for index, value in enumerate(values) if value is None]
        if none_indexes:
            exists = await asyncio.gather(
                *(self.cache.exists(keys[index]) for index in none_indexes)
            )
            for index, key_exists in zip(none_indexes, exists):
                if not key_exists:
                    values[index] = _CACHE_MISS
        return values

    async def set_in_cache(self, result, fn, fn_args, fn_kwargs):
        try:
            await self.cache.multi_set(
                [(self.key_builder(k, fn, *fn_args, **fn_kwargs), v) for k, v in result.items()],
                ttl=self.ttl,
            )
        except Exception:
            logger.exception("Couldn't set %s, unexpected error", result)
