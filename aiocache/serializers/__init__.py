import logging

from .serializers import (
    BaseSerializer,
    CorruptedPayloadError,
    JsonSerializer,
    MigrationCycleError,
    MigrationRegistryError,
    MigrationTarget,
    MissingMigrationStep,
    NullSerializer,
    PickleSerializer,
    ReadRepairUnsupportedError,
    StringSerializer,
    UnknownVersionError,
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
    "MigrationRegistryError",
    "MigrationCycleError",
    "MissingMigrationStep",
    "UnknownVersionError",
    "CorruptedPayloadError",
    "ReadRepairUnsupportedError",
    "MigrationTarget",
]
