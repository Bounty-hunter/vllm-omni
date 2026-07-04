# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextvars
import threading
from typing import ClassVar

import torch

_TLS = threading.local()
_COMM_STREAM_VAR: contextvars.ContextVar[torch.cuda.Stream | None] = contextvars.ContextVar(
    "omni_comm_stream", default=None
)


class CommStreamManager:
    """Per-device singleton comm stream for overlap collectives."""

    _streams: ClassVar[dict[int, torch.cuda.Stream]] = {}

    @classmethod
    def get(cls, device: torch.device | int) -> torch.cuda.Stream:
        if not torch.cuda.is_available():
            raise RuntimeError("CommStreamManager requires CUDA.")
        dev = torch.device(device)
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        if idx not in cls._streams:
            cls._streams[idx] = torch.cuda.Stream(device=idx)
        return cls._streams[idx]

    @classmethod
    def get_compute_stream(cls, device: torch.device | int) -> torch.cuda.Stream:
        if not torch.cuda.is_available():
            raise RuntimeError("CommStreamManager requires CUDA.")
        dev = torch.device(device)
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        return torch.cuda.current_stream(device=idx)


def get_current_comm_stream() -> torch.cuda.Stream | None:
    """Return the comm stream for the active overlap context, if any."""
    return _COMM_STREAM_VAR.get()


class CommStreamContext:
    """Route collectives to a dedicated comm stream without switching compute stream."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        device: torch.device | int | None = None,
        comm_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self._enabled = enabled and torch.cuda.is_available()
        self._device = device
        self._comm_stream = comm_stream
        self._token: contextvars.Token | None = None
        self._prev_tls: torch.cuda.Stream | None = None

    def __enter__(self) -> CommStreamContext:
        if not self._enabled:
            self._token = _COMM_STREAM_VAR.set(None)
            return self

        stream = self._comm_stream
        if stream is None:
            dev = self._device if self._device is not None else torch.cuda.current_device()
            stream = CommStreamManager.get(dev)

        self._prev_tls = getattr(_TLS, "comm_stream", None)
        _TLS.comm_stream = stream
        self._token = _COMM_STREAM_VAR.set(stream)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._token is not None:
            _COMM_STREAM_VAR.reset(self._token)
        if self._prev_tls is None:
            if hasattr(_TLS, "comm_stream"):
                delattr(_TLS, "comm_stream")
        else:
            _TLS.comm_stream = self._prev_tls
