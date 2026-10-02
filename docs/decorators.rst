..  _decorators:

Decorators
==========

aiocache comes with a couple of decorators for caching results from asynchronous functions. Do not use the decorator in synchronous functions, it may lead to unexpected behavior.

..  _cached:

cached
------

.. automodule:: aiocache
  :members: cached

.. literalinclude:: ../examples/cached_decorator.py
  :language: python
  :linenos:

..  _cached_stale_while_revalidate:

cached_stale_while_revalidate
-----------------------------

.. automodule:: aiocache
  :members: cached_stale_while_revalidate

Use ``ttl`` for the fresh lifetime and ``stale_ttl`` for the additional
serve-stale lifetime. While an entry is stale, the first request starts a
single refresh for that cache instance and key and immediately receives the
old value. After the hard deadline, callers wait for the shared refresh. If
refresh fails, the stale value and SWR metadata are deleted, so the failure
is not hidden by expired data.

``cache_read=False`` bypasses both stored values and the shared refresh.
``cache_write=False`` prevents writes, deletes, and refresh starts; a value
found by the read can still be returned. ``aiocache_wait_for_write`` only
affects the direct write performed when ``cache_read=False``.

The cache serializer is used for the function result. SWR metadata is always
stored as internal JSON, so custom serializers only need to support the cached
result type.

The ``__aiocache_swr_metadata__:`` key prefix is reserved for SWR metadata.

..  _multi_cached:

multi_cached
------------

.. automodule:: aiocache
  :members: multi_cached

.. literalinclude:: ../examples/multicached_decorator.py
  :language: python
  :linenos:
