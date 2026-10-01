..  _serializers:

Serializers
===========

Serializers can be attached to backends in order to serialize/deserialize data sent and retrieved from the backend. This allows to apply transformations to data in case you want it to be saved in a specific format in your cache backend. For example, imagine you have your ``Model`` and want to serialize it to something that Redis can understand (Redis can't store python objects). This is the task of a serializer.

To use a specific serializer::

    >>> from aiocache import SimpleMemoryCache
    >>> from aiocache.serializers import PickleSerializer
    cache = SimpleMemoryCache(serializer=PickleSerializer())

Currently the following are built in:


..  _nullserializer:

NullSerializer
--------------
.. autoclass:: aiocache.serializers.NullSerializer
  :members:


..  _stringserializer:

StringSerializer
----------------

.. autoclass:: aiocache.serializers.StringSerializer
  :members:

..  _pickleserializer:

PickleSerializer
----------------

.. autoclass:: aiocache.serializers.PickleSerializer
  :members:

..  _jsonserializer:

JsonSerializer
--------------

.. autoclass:: aiocache.serializers.JsonSerializer
  :members:

..  _msgpackserializer:

MsgPackSerializer
-----------------

.. autoclass:: aiocache.serializers.MsgPackSerializer
  :members:

..  _versionedserializer:

VersionedSerializer
-------------------

``VersionedSerializer`` wraps values with an explicit format version. Older
values are migrated through functions supplied in a registry; migration code
is never selected or imported from cache contents::

    >>> from aiocache import SimpleMemoryCache
    >>> from aiocache.serializers import VersionedSerializer
    >>> serializer = VersionedSerializer(
    ...     current_version=2,
    ...     migrations={(1, 2): lambda payload: payload.upper()},
    ...     read_repair=True,
    ... )
    >>> cache = SimpleMemoryCache(serializer=serializer)

``SimpleMemoryCache`` can perform read repair. Other backends read and convert
old values when ``read_repair`` is disabled, but reject enabling repair because
they do not provide the required atomic, TTL-preserving replacement.

.. autoclass:: aiocache.serializers.VersionedSerializer
  :members:

In case the current serializers are not covering your needs, you can always define your custom serializer as shown in ``examples/serializer_class.py``:

.. literalinclude:: ../examples/serializer_class.py
  :language: python
  :linenos:

You can also use marshmallow as your serializer (``examples/marshmallow_serializer_class.py``):

.. literalinclude:: ../examples/marshmallow_serializer_class.py
  :language: python
  :linenos:

By default cache backends assume they are working with ``str`` types. If your custom implementation transform data to bytes, you will need to set the class attribute ``encoding`` to ``None``.
