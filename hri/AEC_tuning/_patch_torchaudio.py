"""
Compatibility shim for deepfilternet + torchaudio >= 2.2.

torchaudio >= 2.2 removed `torchaudio.backend.common.AudioMetaData`.
deepfilternet 0.5.6 still imports it. This module patches the import
path so `df` can load without downgrading torchaudio.

Import this module BEFORE importing `df`:
    import _patch_torchaudio  # noqa: F401
    import df as DF_MODULE
"""

import types
import sys


def _apply():
    try:
        from torchaudio.backend.common import AudioMetaData  # noqa: F401
        return  # Already works, no patch needed
    except (ImportError, ModuleNotFoundError):
        pass

    # Create a minimal AudioMetaData namedtuple matching the old API
    from typing import NamedTuple

    class AudioMetaData(NamedTuple):
        sample_rate: int
        num_frames: int
        num_channels: int
        bits_per_sample: int = 0
        encoding: str = ""

    # Inject the shim module into sys.modules
    import torchaudio

    backend_mod = types.ModuleType("torchaudio.backend")
    common_mod = types.ModuleType("torchaudio.backend.common")
    common_mod.AudioMetaData = AudioMetaData

    backend_mod.common = common_mod

    sys.modules["torchaudio.backend"] = backend_mod
    sys.modules["torchaudio.backend.common"] = common_mod

    # Also attach to the torchaudio package so attribute access works
    if not hasattr(torchaudio, "backend"):
        torchaudio.backend = backend_mod


_apply()
