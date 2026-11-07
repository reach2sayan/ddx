"""JIT configuration, with the ranges the C++ struct cannot check for a caller."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from . import _ddx
from ._ddx import Backend, OptLevel, VecLib

# The defaults depend on how the library was built, so they are read from it
# rather than restated here.
_DEFAULTS: _ddx._Options = _ddx._Options()


class Options(BaseModel):

    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: Backend = _DEFAULTS.backend
    """``COMPILE`` compiles each lane on the first call that needs it, and that
    call waits for the kernel."""

    points: int = Field(_DEFAULTS.points, ge=1)
    lanes: int | None = Field(_DEFAULTS.lanes, ge=1)
    """Points per loop iteration. ``None`` derives it from ``points`` and the
    host; ``1`` is scalar. Every width gives the same bits."""

    opt_level: OptLevel = _DEFAULTS.opt_level
    codegen_level: OptLevel = _DEFAULTS.codegen_level

    slp_vectorize: bool = _DEFAULTS.slp_vectorize
    veclib: VecLib = _DEFAULTS.veclib
    contract: bool = _DEFAULTS.contract

    def _native(self) -> _ddx._Options:
        native = _ddx._Options()
        for field in type(self).model_fields:
            setattr(native, field, getattr(self, field))
        return native

    @classmethod
    def _from_native(cls, native: _ddx._Options) -> Self:
        return cls(**{f: getattr(native, f) for f in cls.model_fields})
