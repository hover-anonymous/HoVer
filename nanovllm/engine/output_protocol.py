"""Strict, versioned EngineCore-to-frontend output protocol.

EngineCore owns the mutable :class:`Sequence` objects used by the scheduler
and model runner.  The API process consumes only a small immutable projection
after each round.  This module deliberately preserves *round-message cadence*:
an all-WAITING/PREFILLING round is represented by an empty v1 batch and is
still sent over the wire by the integration patch.

This protocol is output-direction only.  Multimodal request state such as
``pixel_values``, ``token_modalities``, ``images_seq_mask``, source image paths,
or crop metadata has already served its purpose inside the engine and must
never be projected back to the frontend.  The allowlist below is exactly the
five small fields consumed by ``_AsyncLLMEngine.step``.
"""

from __future__ import annotations

import operator
import time
from typing import Iterable, NamedTuple, Optional, Tuple, Union

from nanovllm.engine.sequence import SequenceStatus


ENGINE_OUTPUT_SCHEMA_VERSION = 1
ENGINE_OUTPUT_WIRE_PROTOCOL = "minimal-sequence-dto-v1"

_EMITTED_STATUSES = frozenset(
    {
        SequenceStatus.DECODING,
        SequenceStatus.FINISHED,
        SequenceStatus.ERROR,
    }
)
_INTERNAL_STATUSES = frozenset(
    {SequenceStatus.WAITING, SequenceStatus.PREFILLING}
)


class EngineOutputProtocolError(RuntimeError):
    """Raised when an output payload cannot be consumed without ambiguity."""


SequenceId = Union[str, int]


class EngineSequenceOutput(NamedTuple):
    """The exact state consumed by ``_AsyncLLMEngine.step``."""

    seq_id: SequenceId
    status: SequenceStatus
    last_token: int
    error_message: str

    @property
    def is_finished(self) -> bool:
        return self.status is SequenceStatus.FINISHED


class EngineOutputBatch(NamedTuple):
    """One wire message corresponding to one EngineCore output round."""

    schema_version: int
    emitted_monotonic_ns: int
    source_sequence_count: int
    internal_sequence_count: int
    outputs: Tuple[EngineSequenceOutput, ...]

    def validate(self) -> "EngineOutputBatch":
        if type(self.schema_version) is not int:
            raise EngineOutputProtocolError("schema version must be an int")
        if self.schema_version != ENGINE_OUTPUT_SCHEMA_VERSION:
            raise EngineOutputProtocolError(
                "unsupported engine output schema "
                f"{self.schema_version!r}; expected "
                f"{ENGINE_OUTPUT_SCHEMA_VERSION}"
            )
        _require_plain_int(
            self.emitted_monotonic_ns,
            "emitted_monotonic_ns",
            minimum=1,
        )
        source_count = _require_plain_int(
            self.source_sequence_count,
            "source_sequence_count",
            minimum=0,
        )
        internal_count = _require_plain_int(
            self.internal_sequence_count,
            "internal_sequence_count",
            minimum=0,
        )
        if not isinstance(self.outputs, tuple):
            raise EngineOutputProtocolError("outputs must be an immutable tuple")
        if len(self.outputs) + internal_count != source_count:
            raise EngineOutputProtocolError("engine output count mismatch")
        for output in self.outputs:
            if not isinstance(output, EngineSequenceOutput):
                raise EngineOutputProtocolError("invalid sequence output DTO")
            _validate_sequence_id(output.seq_id)
            _validate_status(output.status, allowed=_EMITTED_STATUSES)
            token = _validate_token(output.last_token)
            if output.status in (
                SequenceStatus.DECODING,
                SequenceStatus.FINISHED,
            ) and token < 0:
                raise EngineOutputProtocolError(
                    "visible token output must have a non-negative token id"
                )
            if output.status is SequenceStatus.ERROR and token != -1:
                raise EngineOutputProtocolError(
                    "ERROR output must use last_token=-1"
                )
            _validate_error_message(output.error_message)
        return self


def _require_plain_int(value, field: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise EngineOutputProtocolError(
            f"{field} must be an int >= {minimum}; got {value!r}"
        )
    return value


def _required_attribute(obj: object, field: str):
    try:
        return getattr(obj, field)
    except AttributeError as exc:
        raise EngineOutputProtocolError(
            f"sequence is missing required field {field!r}"
        ) from exc


def _validate_sequence_id(value) -> SequenceId:
    if isinstance(value, str):
        if not value:
            raise EngineOutputProtocolError("seq_id must not be empty")
        return value
    if isinstance(value, bool):
        raise EngineOutputProtocolError("bool is not a valid seq_id")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise EngineOutputProtocolError(
            f"seq_id must be a string or non-negative integer; got {value!r}"
        ) from exc
    if normalized < 0:
        raise EngineOutputProtocolError(
            f"negative seq_id {normalized} is reserved for control messages"
        )
    return int(normalized)


def _validate_status(status, *, allowed) -> SequenceStatus:
    if not isinstance(status, SequenceStatus):
        raise EngineOutputProtocolError(
            f"status must be SequenceStatus; got {status!r}"
        )
    if status not in allowed:
        raise EngineOutputProtocolError(
            f"status {status.name!r} is not valid in this payload position"
        )
    return status


def _validate_token(value) -> int:
    if isinstance(value, bool):
        raise EngineOutputProtocolError("bool is not a valid token id")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise EngineOutputProtocolError(
            f"last_token must be an integer; got {value!r}"
        ) from exc


def _validate_error_message(value) -> str:
    if not isinstance(value, str):
        raise EngineOutputProtocolError(
            f"error_message must be a string; got {value!r}"
        )
    return value


def _project_sequence(sequence: object):
    raw_seq_id = _required_attribute(sequence, "seq_id")
    if not isinstance(raw_seq_id, bool):
        try:
            if operator.index(raw_seq_id) == -1:
                raise EngineOutputProtocolError(
                    "shutdown sentinel must use the SHUTDOWN request type"
                )
        except TypeError:
            pass

    seq_id = _validate_sequence_id(raw_seq_id)
    status = _validate_status(
        _required_attribute(sequence, "status"),
        allowed=_EMITTED_STATUSES | _INTERNAL_STATUSES,
    )
    last_token = _validate_token(_required_attribute(sequence, "last_token"))
    error_message = _validate_error_message(
        _required_attribute(sequence, "error_message")
    )
    return seq_id, status, last_token, error_message


def build_engine_output_batch(
    sequences: Iterable[object],
    *,
    emitted_monotonic_ns: Optional[int] = None,
) -> EngineOutputBatch:
    """Project one EngineCore round onto an immutable v1 wire batch.

    WAITING/PREFILLING Sequence *objects* are omitted from ``outputs`` because
    the frontend discards them, but their round is not omitted: the caller must
    send the resulting empty batch when every source sequence is internal.
    """

    try:
        sequence_list = list(sequences)
    except TypeError as exc:
        raise EngineOutputProtocolError("sequences must be iterable") from exc

    outputs = []
    internal_count = 0
    for sequence in sequence_list:
        seq_id, status, last_token, error_message = _project_sequence(sequence)
        if status in _INTERNAL_STATUSES:
            internal_count += 1
            continue
        if status in (SequenceStatus.DECODING, SequenceStatus.FINISHED):
            if last_token < 0:
                raise EngineOutputProtocolError(
                    "visible token output must have a non-negative token id"
                )
        elif status is SequenceStatus.ERROR and last_token != -1:
            raise EngineOutputProtocolError("ERROR output must use last_token=-1")
        outputs.append(
            EngineSequenceOutput(
                seq_id=seq_id,
                status=status,
                last_token=last_token,
                error_message=error_message,
            )
        )

    if emitted_monotonic_ns is None:
        emitted_ns = time.monotonic_ns()
    else:
        emitted_ns = _require_plain_int(
            emitted_monotonic_ns,
            "emitted_monotonic_ns",
            minimum=1,
        )
    return EngineOutputBatch(
        schema_version=ENGINE_OUTPUT_SCHEMA_VERSION,
        emitted_monotonic_ns=emitted_ns,
        source_sequence_count=len(sequence_list),
        internal_sequence_count=internal_count,
        outputs=tuple(outputs),
    ).validate()


def _validate_legacy_sequence(sequence: object) -> None:
    """Validate the fields consumed by the current legacy frontend."""

    _, status, last_token, _ = _project_sequence(sequence)
    if status in (SequenceStatus.DECODING, SequenceStatus.FINISHED):
        if last_token < 0:
            raise EngineOutputProtocolError(
                "legacy visible token output has a negative token id"
            )
    elif status is SequenceStatus.ERROR and last_token != -1:
        raise EngineOutputProtocolError(
            "legacy ERROR output must use last_token=-1"
        )


def unwrap_engine_output_payload(payload, *, allow_legacy: bool = True):
    """Return frontend rows after strict validation.

    Compatibility is intentionally one-way: a new CoreClient may accept the
    legacy ``list[Sequence]`` payload when ``allow_legacy`` is true.  An old
    CoreClient cannot consume a v1 batch, so both processes still need an
    atomic restart.  Formal/diagnostic DTO runs should set
    ``NANOVLLM_REQUIRE_OUTPUT_DTO_V1=1`` in CoreClient.
    """

    if isinstance(payload, EngineOutputBatch):
        return list(payload.validate().outputs)
    if isinstance(payload, list):
        if not allow_legacy:
            raise EngineOutputProtocolError(
                "legacy Sequence-list output rejected: DTO v1 is required"
            )
        for sequence in payload:
            _validate_legacy_sequence(sequence)
        return payload
    raise EngineOutputProtocolError(
        f"unexpected engine output payload type {type(payload).__name__}"
    )
