import logging

from aiocache.exceptions import (
    CorruptedPayloadError,
    InvalidMigrationError,
    MigrationCycleError,
    MigrationError,
    MissingMigrationError,
    ReadRepairUnsupportedError,
    UnknownVersionError,
)

from .serializers import (
    BaseSerializer,
    JsonSerializer,
    NullSerializer,
    PickleSerializer,
    StringSerializer,
    VersionedSerializer,
)

logger = logging.getLogger(__name__)


try:
    import msgpack
except ImportError:
    logger.debug("msgpack not installed, MsgPackSerializer unavailable")
else:
    from .serializers import MsgPackSerializer

    del msgpack


__all__ = [
    "BaseSerializer",
    "NullSerializer",
    "StringSerializer",
    "PickleSerializer",
    "JsonSerializer",
    "MsgPackSerializer",
    "VersionedSerializer",
    "MigrationError",
    "MissingMigrationError",
    "MigrationCycleError",
    "InvalidMigrationError",
    "UnknownVersionError",
    "CorruptedPayloadError",
    "ReadRepairUnsupportedError",
]
