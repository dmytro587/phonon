"""The vendored Phonon decode runtime, and the importer that isolates it.

WHAT IS IN HERE
---------------
Five pure-Python modules, ~112 KB, vendored from the Phonon engine's release
tree:

    quint5_codec.py          base-5 storage codec        (byte-identical)
    quint_fold_codec.py      plane-folding codec         (byte-identical)
    package_release_bps.py   byte-plane archive unpacker (byte-identical)
    v18_runtime.py           the model + weight loader   (+ q8 back-port)
    optimized_v18.py         the decode wrapper          (+ q8 back-port)

WHAT IS NOT IN HERE, AND MUST NEVER BE
---------------------------------------
Two module names (`CLOSED_MODULES` below) are deliberately not shipped.
`v18_runtime.py` refers to both, but ONLY as function-local imports inside
branches no shipping profile takes (`trit3_codes` is false for every
published artifact, and `use_fused_qmv` is never enabled by this CLI), so
the files' absence is correct and inert rather than a latent crash.
`fermion/_speech/selftest.py` asserts, on every run, that neither file exists
anywhere under `fermion/` and that neither is imported at module scope.

THE TWO BACK-PORTED HUNKS, AND WHY THEY WERE NECESSARY
-------------------------------------------------------
The vendored release tree predates `PHONON_ENC=q8` — the 8-bit audio tower
the Mac app ships, gated 2026-08-26. So the unmodified runtime cannot run
the shipping configuration: `load_optimized_v18()` there has no
`audio_tower_bits` parameter at all. Defaulting the CLI to the app's fast
path therefore REQUIRED back-porting it.

What was back-ported is exactly the additive q8 plumbing, and nothing else:

  * `v18_runtime.quantize_audio_tower()` — copied byte-identically from
    the engine's source (asserted by the selftest), plus its two
    `load_v18()` parameters and the four-line call site;
  * `optimized_v18` — the same two parameters threaded through
    `OptimizedV18.__init__` and `load_optimized_v18`.

Nothing else was taken: the vendored copies keep the release tree's lazy,
guarded import form.

WHY A CUSTOM IMPORTER
---------------------
These modules import each other by BARE name (`from quint5_codec import …`),
because in their own repo they are five files in one flat directory. Rewriting
those imports would make the vendored copies non-identical to the reviewed
originals, which is exactly the property worth keeping. So instead they are
loaded from this package's directory and registered under their flat names —
and if a flat name is ALREADY taken by something that is not ours, this raises
rather than silently binding a stranger's module into our decode path.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: Every vendored module.
MODULES = ("quint5_codec", "quint_fold_codec", "package_release_bps",
           "v18_runtime", "optimized_v18")

#: Which siblings each one imports by bare name, in the order they must be
#: registered. Declared rather than inferred so that loading the UNPACKER does
#: not drag in the decoder: `package_release_bps` needs only numpy, while
#: `quint5_codec` imports `mlx.core` — and unpacking must stay possible without
#: paying for the Metal runtime.
DEPS = {
    "quint5_codec": (),
    "quint_fold_codec": (),
    "package_release_bps": (),
    "v18_runtime": ("quint5_codec", "quint_fold_codec"),
    "optimized_v18": ("quint5_codec", "quint_fold_codec", "v18_runtime"),
}

#: Modules that must NEVER appear in a shipped wheel.
CLOSED_MODULES = ("fused_trit_kernel", "trit3_codec")

_PREFIX = "fermion._speech._engine."


class EngineImportError(ImportError):
    pass


def _install(name: str):
    """Import one vendored module under both its flat and namespaced name."""
    full = _PREFIX + name
    if full in sys.modules:
        return sys.modules[full]
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, "__file__", None) \
            and Path(existing.__file__).resolve().parent != HERE:
        raise EngineImportError(
            f"the name {name!r} is already bound to {existing.__file__}, "
            f"which is not part of {__name__}. The Phonon runtime imports its "
            f"siblings by bare name; binding a foreign module there would put "
            f"unknown code in the decode path. Rename or unload it and retry.")
    path = HERE / f"{name}.py"
    if not path.is_file():
        raise EngineImportError(f"vendored runtime module missing: {path}")
    spec = importlib.util.spec_from_file_location(full, path)
    module = importlib.util.module_from_spec(spec)
    # Register BEFORE exec so a circular bare-name import resolves, and under
    # the flat name so `from quint5_codec import …` inside the next module
    # finds this one rather than searching sys.path.
    sys.modules[full] = module
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(full, None)
        if sys.modules.get(name) is module:
            sys.modules.pop(name, None)
        raise
    return module


def load(name: str):
    """Import a vendored engine module by flat name, installing its deps."""
    if name not in MODULES:
        raise EngineImportError(f"{name!r} is not a vendored engine module")
    for dep in DEPS[name]:
        _install(dep)
    return _install(name)


def vendored_files() -> list[Path]:
    return [HERE / f"{n}.py" for n in MODULES]
