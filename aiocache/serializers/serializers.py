import asyncio
import inspect
import logging
import pickle  # noqa: S403
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

from aiocache.exceptions import (
    CorruptedPayloadError,
    InvalidMigrationError,
    MigrationCycleError,
    MissingMigrationError,
    UnknownVersionError,
)

logger = logging.getLogger(__name__)

try:
    import ujson as json  # noqa: I900
except ImportError:
    logger.debug("ujson module not found, using json")
    import json  # type: ignore[no-redef]

try:
    import msgpack
except ImportError:
    msgpack = None
    logger.debug("msgpack not installed, MsgPackSerializer unavailable")


_NOT_SET = object()

VERSION_KEY = "version"
PAYLOAD_KEY = "payload"


class MigrationResult(NamedTuple):
    payload: Any
    migrated: bool


Migration = Callable[[Any], Any]
MigrationRegistry = Mapping[Tuple[int, int], Migration]


class BaseSerializer(ABC):

    DEFAULT_ENCODING: Optional[str] = "utf-8"

    def __init__(self, *args, encoding=_NOT_SET, **kwargs):
        self.encoding = self.DEFAULT_ENCODING if encoding is _NOT_SET else encoding
        super().__init__(*args, **kwargs)

    @abstractmethod
    def dumps(self, value: Any, /) -> Any:
        """Serialise the value to be stored in the backend."""

    @abstractmethod
    def loads(self, value: Any, /) -> Any:
        """Decode the value retrieved from the backend."""


class NullSerializer(BaseSerializer):
    """
    This serializer does nothing. Its only recommended to be used by
    :class:`aiocache.SimpleMemoryCache` because for other backends it will
    produce incompatible data unless you work only with str types because it
    store data as is.

    DISCLAIMER: Be careful with mutable types and memory storage. The following
    behavior is considered normal (same as ``functools.lru_cache``)::

        cache = Cache()
        my_list = [1]
        await cache.set("key", my_list)
        my_list.append(2)
        await cache.get("key")  # Will return [1, 2]
    """

    def dumps(self, value):
        """
        Returns the same value
        """
        return value

    def loads(self, value):
        """
        Returns the same value
        """
        return value


class StringSerializer(BaseSerializer):
    """
    Converts all input values to str. All return values are also str. Be
    careful because this means that if you store an ``int(1)``, you will get
    back '1'.

    The transformation is done by just casting to str in the ``dumps`` method.

    If you want to keep python types, use ``PickleSerializer``. ``JsonSerializer``
    may also be useful to keep type of simple python types.
    """

    def dumps(self, value):
        """
        Serialize the received value casting it to str.

        :param value: obj Anything support cast to str
        :returns: str
        """
        return str(value)

    def loads(self, value):
        """
        Returns value back without transformations
        """
        return value


class PickleSerializer(BaseSerializer):
    """
    Transform data to bytes using pickle.dumps and pickle.loads to retrieve it back.
    """

    DEFAULT_ENCODING = None

    def __init__(self, *args, protocol=pickle.DEFAULT_PROTOCOL, **kwargs):
        super().__init__(*args, **kwargs)
        self.protocol = protocol

    def dumps(self, value):
        """
        Serialize the received value using ``pickle.dumps``.

        :param value: obj
        :returns: bytes
        """
        return pickle.dumps(value, protocol=self.protocol)

    def loads(self, value):
        """
        Deserialize value using ``pickle.loads``.

        :param value: bytes
        :returns: obj
        """
        if value is None:
            return None
        return pickle.loads(value)  # noqa: S301


class JsonSerializer(BaseSerializer):
    """
    Transform data to json string with json.dumps and json.loads to retrieve it back. Check
    https://docs.python.org/3/library/json.html#py-to-json-table for how types are converted.

    ujson will be used by default if available. Be careful with differences between built in
    json module and ujson:
        - ujson dumps supports bytes while json doesn't
        - ujson and json outputs may differ sometimes
    """

    def dumps(self, value):
        """
        Serialize the received value using ``json.dumps``.

        :param value: dict
        :returns: str
        """
        return json.dumps(value)

    def loads(self, value):
        """
        Deserialize value using ``json.loads``.

        :param value: str
        :returns: output of ``json.loads``.
        """
        if value is None:
            return None
        return json.loads(value)


class MsgPackSerializer(BaseSerializer):
    """
    Transform data to bytes using msgpack.dumps and msgpack.loads to retrieve it back. You need
    to have ``msgpack`` installed in order to be able to use this serializer.

    :param encoding: str. Can be used to change encoding param for ``msg.loads`` method.
        Default is None (backends pass raw bytes directly to loads without decoding).
    :param use_list: bool. Can be used to change use_list param for ``msgpack.loads`` method.
        Default is True.
    """

    DEFAULT_ENCODING = None

    def __init__(self, *args, use_list=True, **kwargs):
        if not msgpack:
            raise RuntimeError("msgpack not installed, MsgPackSerializer unavailable")
        self.use_list = use_list
        super().__init__(*args, **kwargs)

    def dumps(self, value):
        """
        Serialize the received value using ``msgpack.dumps``.

        :param value: obj
        :returns: bytes
        """
        return msgpack.dumps(value)

    def loads(self, value):
        """
        Deserialize value using ``msgpack.loads``.

        :param value: bytes
        :returns: obj
        """
        raw = self.encoding not in (None, "utf-8")
        if value is None:
            return None
        return msgpack.loads(value, raw=raw, use_list=self.use_list)


class VersionedSerializer(BaseSerializer):
    """
    Store a format version with a value and migrate old payloads using an
    explicit migration registry.

    ``migrations`` maps ``(source_version, target_version)`` to a callable.
    Callables may be coroutine functions. No code is imported based on cache
    contents; every migration step must already be registered here.
    """

    def __init__(
        self,
        *args,
        encoding=_NOT_SET,
        current_version: Optional[int] = None,
        version: Optional[int] = None,
        format_version: Optional[int] = None,
        migrations: Optional[Mapping[Any, Any]] = None,
        serializer: Optional[BaseSerializer] = None,
        read_repair: bool = False,
        repair: Optional[bool] = None,
        **kwargs,
    ):
        if current_version is None:
            if format_version is not None:
                current_version = format_version
            elif version is not None:
                current_version = version
            else:
                current_version = 1
        aliases = (version, format_version)
        if any(alias is not None and alias != current_version for alias in aliases):
            raise InvalidMigrationError("current_version and its aliases must match")

        if not isinstance(current_version, int) or isinstance(current_version, bool):
            raise InvalidMigrationError("current_version must be an integer")
        if current_version < 1:
            raise InvalidMigrationError("current_version must be greater than zero")

        if serializer is None:
            serializer = JsonSerializer(
                encoding=self.DEFAULT_ENCODING if encoding is _NOT_SET else encoding
            )

        if repair is not None and repair != read_repair:
            raise InvalidMigrationError("read_repair and repair must match")

        self.current_version = current_version
        self.read_repair = read_repair if repair is None else repair
        self.serializer = serializer
        self.migrations = self._normalize_migrations(migrations or {})
        self._validate_migrations()
        super().__init__(
            *args,
            encoding=self.serializer.encoding if encoding is _NOT_SET else encoding,
            **kwargs,
        )

    def dumps(self, value: Any, /) -> Any:
        envelope = {VERSION_KEY: self.current_version, PAYLOAD_KEY: value}
        return self.serializer.dumps(envelope)

    def loads(self, value: Any, /) -> Any:
        result = self.loads_versioned(value)
        if inspect.isawaitable(result):

            async def _load() -> Any:
                return (await result).payload

            return _load()
        return result.payload

    async def aloads(self, value: Any, /) -> Any:
        result = await self._await_migration_result(self.loads_versioned(value))
        return result.payload

    def loads_versioned(self, value: Any, /) -> MigrationResult:
        if value is None:
            return MigrationResult(None, False)

        envelope = self._load_envelope(value)
        version = envelope[VERSION_KEY]
        payload = envelope[PAYLOAD_KEY]

        if version == self.current_version:
            return MigrationResult(payload, False)

        path = self._migration_path(version)
        if any(self._is_async_migration(migration) for migration in path):
            return self._migrate(payload, path)

        for migration in path:
            try:
                payload = migration(payload)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise CorruptedPayloadError("Cached payload is corrupted") from exc
        return MigrationResult(payload, True)

    async def _migrate(self, payload: Any, path: Tuple[Migration, ...]) -> MigrationResult:
        for migration in path:
            try:
                migrated = migration(payload)
                if inspect.isawaitable(migrated):
                    migrated = await migrated
                payload = migrated
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise CorruptedPayloadError("Cached payload is corrupted") from exc
        return MigrationResult(payload, True)

    def _load_envelope(self, value: Any) -> Dict[str, Any]:
        try:
            envelope = self.serializer.loads(value)
        except CorruptedPayloadError:
            raise
        except Exception as exc:
            raise CorruptedPayloadError("Cached value is not a versioned payload") from exc

        if not isinstance(envelope, Mapping):
            raise CorruptedPayloadError("Versioned payload must be a mapping")
        if VERSION_KEY not in envelope or PAYLOAD_KEY not in envelope:
            raise CorruptedPayloadError("Versioned payload must contain version and payload")

        version = envelope[VERSION_KEY]
        if not isinstance(version, int) or isinstance(version, bool):
            raise CorruptedPayloadError("Payload version must be an integer")
        if version < 1:
            raise CorruptedPayloadError("Payload version must be greater than zero")
        if version > self.current_version:
            raise UnknownVersionError(
                f"Cached payload version {version} is newer than {self.current_version}"
            )
        return dict(envelope)

    def _migration_path(self, version: int) -> Tuple[Migration, ...]:
        path = []
        seen = set()
        while version != self.current_version:
            if version in seen:
                raise MigrationCycleError(f"Migration cycle detected at version {version}")
            seen.add(version)
            edge = next(
                (
                    (source, target, migration)
                    for (source, target), migration in self.migrations.items()
                    if source == version
                ),
                None,
            )
            if edge is None:
                raise MissingMigrationError(f"No migration registered from version {version}")
            source, target, migration = edge
            path.append(migration)
            version = target
        return tuple(path)

    @staticmethod
    async def _await_migration_result(result: Any) -> MigrationResult:
        if inspect.isawaitable(result):
            result = await result
        return result

    @staticmethod
    def _normalize_migrations(
        migrations: Mapping[Any, Any],
    ) -> Dict[Tuple[int, int], Migration]:
        normalized = {}
        for edge, migration in migrations.items():
            if isinstance(edge, tuple) and len(edge) == 2:
                source, target = edge
            elif (
                isinstance(edge, int)
                and isinstance(migration, tuple)
                and len(migration) == 2
            ):
                source = edge
                target, migration = migration
            elif isinstance(edge, int) and callable(migration):
                source, target = edge, edge + 1
            else:
                raise InvalidMigrationError("Migration registry keys must be (source, target)")

            if not callable(migration):
                raise InvalidMigrationError("Migration must be callable")
            normalized[(source, target)] = migration
        return normalized

    def _validate_migrations(self) -> None:
        graph: Dict[int, Tuple[int, Migration]] = {}
        for (source, target), migration in self.migrations.items():
            if not isinstance(source, int) or isinstance(source, bool):
                raise InvalidMigrationError("Migration source version must be an integer")
            if not isinstance(target, int) or isinstance(target, bool):
                raise InvalidMigrationError("Migration target version must be an integer")
            if source < 1 or target < 1:
                raise InvalidMigrationError("Migration versions must be greater than zero")
            if source >= self.current_version:
                raise InvalidMigrationError("Migrations cannot start from the current version")
            if target > self.current_version:
                raise UnknownVersionError(f"Unknown migration target version {target}")
            if source in graph:
                raise InvalidMigrationError(f"More than one migration registered from {source}")
            graph[source] = (target, migration)

        for start in graph:
            self._assert_no_cycle(start, graph)

        for source, (target, _) in graph.items():
            if target == source:
                raise MigrationCycleError("Migrations cannot use the same source and target")
            if target < source:
                raise InvalidMigrationError("Migrations must move to newer versions")

        for version in graph:
            self._assert_path_to_current(version, graph)

    def _assert_no_cycle(self, start: int, graph: Mapping[int, Tuple[int, Migration]]) -> None:
        seen = set()
        version = start
        while version != self.current_version and version in graph:
            if version in seen:
                raise MigrationCycleError(f"Migration cycle detected at version {version}")
            seen.add(version)
            version = graph[version][0]

    def _assert_path_to_current(
        self, start: int, graph: Mapping[int, Tuple[int, Migration]]
    ) -> None:
        seen = set()
        version = start
        while version != self.current_version:
            if version in seen:
                raise MigrationCycleError(f"Migration cycle detected at version {version}")
            seen.add(version)
            next_step = graph.get(version)
            if next_step is None:
                raise MissingMigrationError(f"No migration registered from version {version}")
            version = next_step[0]

    @staticmethod
    def _is_async_migration(migration: Migration) -> bool:
        return inspect.iscoroutinefunction(migration) or inspect.iscoroutinefunction(
            getattr(migration, "__call__", migration)
        )
