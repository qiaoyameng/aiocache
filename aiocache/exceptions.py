class InvalidCacheType(Exception):
    pass


class TagsNotSupportedError(NotImplementedError):
    """Raised when tag operations are used with a backend that does not support them."""
