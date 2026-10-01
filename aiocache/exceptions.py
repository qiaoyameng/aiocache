class InvalidCacheType(Exception):
    pass


class SerializationError(ValueError):
    pass


class MigrationError(SerializationError):
    pass


class MissingMigrationError(MigrationError):
    pass


class MigrationCycleError(MigrationError):
    pass


class InvalidMigrationError(MigrationError):
    pass


class UnknownVersionError(MigrationError):
    pass


class CorruptedPayloadError(SerializationError):
    pass


class ReadRepairUnsupportedError(ValueError):
    pass
