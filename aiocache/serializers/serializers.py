import copy
import logging
import pickle  # noqa: S403
from abc import ABC, abstractmethod
from inspect import isawaitable
from typing import Any, Callable, NamedTuple, Optional, Union

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
VERSIONED_FORMAT = "aiocache.versioned"


class MigrationRegistryError(ValueError):
    pass


class MissingMigrationStep(MigrationRegistryError):
    pass


class MigrationCycleError(MigrationRegistryError):
    pass


class UnknownVersionError(MigrationRegistryError):
    pass


class CorruptedPayloadError(ValueError):
    pass


class ReadRepairUnsupportedError(RuntimeError):
    pass


class MigrationTarget(NamedTuple):
    version: int
    function: Callable[[Any], Any]


MigrationResult = Any
MigrationFunction = Callable[[Any], MigrationResult]
MigrationRegistry = dict[int, MigrationTarget]
MigrationStep = Union[
    MigrationFunction,
    tuple[int, MigrationFunction],
    MigrationTarget,
]
MigrationInput = Union[
    dict[int, MigrationStep],
    dict[tuple[int, int], MigrationFunction],
]


def _normalize_migrations(migrations: MigrationInput) -> MigrationRegistry:
    normalized: MigrationRegistry = {}
    for source, target in tuple(migrations.items()):
        function: MigrationFunction
        if isinstance(source, tuple):
            if len(source) != 2:
                raise MigrationRegistryError("Migration edges must contain two versions")
            source, target_version = source
            function = target
            target = (target_version, function)

        if isinstance(source, bool) or not isinstance(source, int):
            raise MigrationRegistryError("Migration source versions must be integers")
        if isinstance(target, tuple):
            if len(target) != 2:
                raise MigrationRegistryError(
                    "Migration targets must contain a version and function"
                )
            target_version, function = target
        else:
            if callable(target):
                target_version, function = source + 1, target
            else:
                raise MigrationRegistryError("Invalid migration target")
        if isinstance(target_version, bool) or not isinstance(target_version, int):
            raise MigrationRegistryError("Migration target versions must be integers")
        if target_version <= source:
            raise MigrationCycleError("Migrations must move to a newer version")
        if not callable(function):
            raise MigrationRegistryError("Migration functions must be callable")
        normalized[source] = MigrationTarget(target_version, function)
    return normalized


def _validate_migration_chain(migrations: MigrationRegistry, current_version: int) -> None:
    visiting: set[int] = set()
    visited: set[int] = set()

    def visit(version: int) -> None:
        if version in visiting:
            raise MigrationCycleError(f"Migration cycle detected at version {version}")
        if version in visited:
            return
        visiting.add(version)
        target = migrations.get(version)
        if target is not None:
            visit(target.version)
        visiting.remove(version)
        visited.add(version)

    for version in migrations:
        visit(version)

    for version in migrations:
        target = migrations[version]
        if target.version > current_version:
            raise UnknownVersionError(
                f"Migration path reaches unknown version {target.version}"
            )
        path_version = target.version
        while path_version < current_version:
            next_target = migrations.get(path_version)
            if next_target is None:
                raise MissingMigrationStep(
                    f"No migration path from version {version} to {current_version}"
                )
            path_version = next_target.version


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

    async def aloads(self, value: Any, /) -> Any:
        """Asynchronously decode the value retrieved from the backend."""
        return self.loads(value)


class VersionedSerializer(BaseSerializer):
    """Wrap serialized values with a format version and explicit migrations."""

    _aiocache_versioned = True

    def __init__(
        self,
        inner: Optional[BaseSerializer] = None,
        *,
        version: int = 1,
        migrations: Optional[MigrationInput] = None,
        legacy_version: int = 1,
        read_repair: bool = False,
        **kwargs: Any,
    ) -> None:
        if isinstance(version, bool) or not isinstance(version, int):
            raise MigrationRegistryError("The current version must be an integer")
        if isinstance(legacy_version, bool) or not isinstance(legacy_version, int):
            raise MigrationRegistryError("The legacy version must be an integer")
        self._inner = inner or NullSerializer()
        super().__init__(**kwargs)
        self.version = version
        self.legacy_version = legacy_version
        self.read_repair = read_repair
        normalized_migrations = _normalize_migrations(migrations or {})
        _validate_migration_chain(normalized_migrations, version)
        self.migrations = normalized_migrations

    @property
    def encoding(self) -> Optional[str]:
        return self._inner.encoding

    @encoding.setter
    def encoding(self, value: Optional[str]) -> None:
        self._inner.encoding = value

    def dumps(self, value: Any) -> Any:
        return self._inner.dumps(self._make_envelope(self.version, value))

    def loads(self, value: Any) -> Any:
        version, payload = self._decode(value)
        return self._migrate(version, copy.deepcopy(payload))

    async def aloads(self, value: Any) -> Any:
        version, payload = self._decode(value)
        return await self._amigrate(version, copy.deepcopy(payload))

    async def _aiocache_read(
        self, cache: Any, key: str, raw: Any, token: Any, default: Any, _conn: Any
    ) -> Any:
        if raw is None:
            return default

        await self._after_raw_read(key, raw)
        version, payload = self._decode(raw)
        value = await self._amigrate(version, copy.deepcopy(payload))

        if self.read_repair:
            await self._before_conditional_write(key, value)
            repaired = self._inner.dumps(self._make_envelope(self.version, value))
            await cache._read_repair_set(key, repaired, token, _conn=_conn)

        return value

    async def _after_raw_read(self, key: str, raw: Any) -> None:
        pass

    async def _before_conditional_write(self, key: str, value: Any) -> None:
        pass

    def _make_envelope(self, version: int, payload: Any) -> dict[str, Any]:
        return {
            "__aiocache_format__": VERSIONED_FORMAT,
            "version": version,
            "payload": payload,
        }

    def _decode(self, raw: Any) -> tuple[int, Any]:
        if raw is None:
            return self.version, None

        try:
            decoded = self._inner.loads(raw)
        except CorruptedPayloadError:
            raise
        except Exception as exc:
            raise CorruptedPayloadError("Could not decode versioned cache payload") from exc

        if (
            isinstance(decoded, dict)
            and decoded.get("__aiocache_format__") == VERSIONED_FORMAT
        ):
            version = decoded.get("version")
            if isinstance(version, bool) or not isinstance(version, int):
                raise CorruptedPayloadError("Versioned cache payload has an invalid version")
            if "payload" not in decoded:
                raise CorruptedPayloadError("Versioned cache payload is missing its payload")
            return version, decoded["payload"]

        return self.legacy_version, decoded

    def _migration_steps(self, version: int) -> list[MigrationTarget]:
        if version > self.version:
            raise UnknownVersionError(
                f"Cannot read cache version {version}; supported version is {self.version}"
            )

        steps: list[MigrationTarget] = []
        seen = {version}
        while version < self.version:
            target = self.migrations.get(version)
            if target is None:
                raise MissingMigrationStep(
                    f"No migration registered from version {version} to {self.version}"
                )
            if target.version in seen:
                raise MigrationCycleError(
                    f"Migration cycle detected between versions {version} and {target.version}"
                )
            seen.add(target.version)
            steps.append(target)
            version = target.version

        if version > self.version:
            raise UnknownVersionError(
                f"Cannot read cache version {version}; supported version is {self.version}"
            )
        return steps

    def _migrate(self, version: int, payload: Any) -> Any:
        try:
            for target in self._migration_steps(version):
                result = target.function(payload)
                if isawaitable(result):
                    raise RuntimeError("Use aloads for asynchronously migrating cache values")
                payload = result
        except (MigrationRegistryError, RuntimeError):
            raise
        except Exception as exc:
            raise CorruptedPayloadError("Cache payload could not be migrated") from exc
        return payload

    async def _amigrate(self, version: int, payload: Any) -> Any:
        try:
            for target in self._migration_steps(version):
                result = target.function(payload)
                if isawaitable(result):
                    result = await result
                payload = result
        except MigrationRegistryError:
            raise
        except Exception as exc:
            raise CorruptedPayloadError("Cache payload could not be migrated") from exc
        return payload


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
