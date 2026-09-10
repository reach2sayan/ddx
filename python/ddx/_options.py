"""JIT configuration, with the ranges the C++ struct cannot check for a caller."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from . import _ddx
from ._ddx import Backend, Level, VecLib

# What the build settled on: DDX_JIT_DEFAULT_OPT follows CMAKE_BUILD_TYPE and
# DDX_JIT_DEFAULT_CONTRACT follows DDX_FP_FLAGS, so the defaults are read off the
# library rather than restated here, where they would drift.
_DEFAULTS: _ddx._Options = _ddx._Options()


class Options(BaseModel):

    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: Backend = _DEFAULTS.backend
    """``COMPILE`` starts the compile there and then; calls before it lands are
    swept and switch over when it arrives.
        ``ADAPT`` waits until a lane has been
    asked for ``warm_points`` before compiling it at all.
        ``DEVICE`` builds the graph for the OpenCL device ``device`` names, and
    sweeps until it lands."""

    points: int = Field(_DEFAULTS.points, ge=1)
    lanes: int | None = Field(_DEFAULTS.lanes, ge=1)
    """Points per loop iteration. ``None`` derives it from ``points`` and the
    host; ``1`` is scalar. Every width gives the same bits."""

    opt_level: Level = _DEFAULTS.opt_level
    codegen_level: Level = _DEFAULTS.codegen_level

    warm_points: int = Field(_DEFAULTS.warm_points, ge=0)
    """Under ``ADAPT``, the batch points a lane must be asked for before it is
    compiled. 0 compiles on the first call."""

    hot_points: int = Field(_DEFAULTS.hot_points, ge=0)
    """Under ``ADAPT``, the further points the cheap kernel must run before the
    top one is compiled. A Python lane has one rung, so this is read only by
    equations compiled through the C++ facade."""

    slp: bool = _DEFAULTS.slp
    loop_vectorize: bool = _DEFAULTS.loop_vectorize
    veclib: VecLib = _DEFAULTS.veclib
    contract: bool = _DEFAULTS.contract

    time_passes: bool = _DEFAULTS.time_passes

    retain_object: bool = _DEFAULTS.retain_object
    cache_dir: str = _DEFAULTS.cache_dir # empty disables it

    device: str = _DEFAULTS.device
    """Under ``DEVICE``, which OpenCL device: a case-insensitive part of
    ``"<platform> / <device>"``. Empty takes the first GPU with double
    precision, else the first device of any kind that has it."""

    def _native(self) -> _ddx._Options:
        native = _ddx._Options()
        for field in type(self).model_fields:
            setattr(native, field, getattr(self, field))
        return native

    @classmethod
    def _from_native(cls, native: _ddx._Options) -> Self:
        return cls(**{f: getattr(native, f) for f in cls.model_fields})
