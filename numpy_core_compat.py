"""
Compatibility helper for pickles that reference the historical ``numpy._core`` package.

Newer versions of NumPy (>=1.26) expose ``numpy.core`` but not ``numpy._core``,
which means that objects pickled on older versions fail to import.  Trying to load
``numpy._core`` modules directly can crash the interpreter because it attempts to
re-initialize NumPy's C extensions.  Instead, we alias the already-imported
``numpy.core`` modules to the legacy ``numpy._core`` names so unpickling simply
reuses the existing module objects.
"""

from __future__ import annotations

import importlib
import sys
from typing import Iterable

_ALIAS_INITIALIZED = False


def _ensure_modules_loaded(module_names: Iterable[str]) -> None:
    """Import the provided NumPy submodules so we can alias them safely."""
    for module_name in module_names:
        importlib.import_module(module_name)


def ensure_numpy_core_compatibility() -> None:
    """
    Populate ``sys.modules`` entries for ``numpy._core`` and its submodules so
    that pickles created on old versions of NumPy keep working.
    """
    global _ALIAS_INITIALIZED
    if _ALIAS_INITIALIZED:
        return

    import numpy as np

    core_prefix = "numpy.core"
    legacy_prefix = "numpy._core"
    base_module = importlib.import_module(core_prefix)
    sys.modules.setdefault(legacy_prefix, base_module)

    # Ensure the critical NumPy core modules are imported once under their canonical names.
    required_core_modules = [
        f"{core_prefix}.{suffix}"
        for suffix in [
            "_add_newdocs",
            "_add_newdocs_scalars",
            "_asarray",
            "_dtype",
            "_dtype_ctypes",
            "_exceptions",
            "_internal",
            "_machar",
            "_methods",
            "_multiarray_tests",
            "_multiarray_umath",
            "_string_helpers",
            "_type_aliases",
            "_ufunc_config",
            "arrayprint",
            "defchararray",
            "einsumfunc",
            "fromnumeric",
            "function_base",
            "getlimits",
            "memmap",
            "multiarray",
            "numeric",
            "numerictypes",
            "overrides",
            "records",
            "shape_base",
            "umath",
        ]
    ]
    _ensure_modules_loaded(required_core_modules)

    # Mirror every loaded numpy.core module to the legacy numpy._core namespace.
    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith(core_prefix):
            continue
        alias_name = legacy_prefix + module_name[len(core_prefix) :]
        sys.modules.setdefault(alias_name, module)

    # pandas pickles also import numpy._core.numeric directly when reconstructing ndarrays.
    sys.modules.setdefault(f"{legacy_prefix}.numeric", np.core.numeric)  # type: ignore[attr-defined]

    _ALIAS_INITIALIZED = True


__all__ = ["ensure_numpy_core_compatibility"]
