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

The optional ``tags`` argument can be an iterable or a callable. The callable receives the
wrapped function, its positional and keyword arguments and the computed result, so tags can
depend on both the request and the response:

.. code-block:: python

    @cached(
        cache=cache,
        tags=lambda f, args, kwargs, result: (
            f"product:{kwargs['product_id']}",
            f"taxes:{result['region']}",
        ),
    )
    async def price(product_id):
        ...

..  _multi_cached:

multi_cached
------------

.. automodule:: aiocache
  :members: multi_cached

.. literalinclude:: ../examples/multicached_decorator.py
  :language: python
  :linenos:
