import enum


class EngineCoreRequestType(enum.Enum):
    """Wire-level request types shared by CoreClient and EngineCore."""

    ADD = b"\x00"
    SHUTDOWN = b"\x05"
