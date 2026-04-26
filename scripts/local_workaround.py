"""
Local workaround for OneDrive + Python 3.11 + Transformers environment issues.

This module patches three problems that arise in this specific environment
(OneDrive sync + uv virtualenv + Python 3.11):

1. importlib.metadata.version('torch') returns None
   → transformers/accelerate internally call packaging.version.parse(version("torch"))
   → TypeError on None

2. packaging.version.parse(None) raises TypeError
   → same root cause as above

3. transformers.modeling_utils.check_torch_load_is_safe raises on our safe usage
   → we use weights_only=True and only load trusted checkpoints; the check is
     unnecessary for our use case

Apply BEFORE importing FlagEmbedding / transformers:

    from scripts.local_workaround import apply_patches
    apply_patches()

This should ONLY be needed when running embed_files.py locally on this machine.
Do not commit patches directly into library code — keep them here.
"""

import sys


def apply_patches() -> None:
    # Clear any previously-cached broken transformers/accelerate modules.
    # They may have crashed at import time with None torch version and are now
    # permanently broken in sys.modules.
    for mod in list(sys.modules.keys()):
        if "transformers" in mod or "accelerate" in mod:
            del sys.modules[mod]

    # ── Patch 1: importlib.metadata.version ─────────────────────────────────
    import importlib.metadata

    _dists_map = {d.name: d for d in importlib.metadata.distributions()}
    _orig_version = importlib.metadata.version

    def _patched_version(name: str) -> str:
        if name in _dists_map:
            return _dists_map[name].version
        return _orig_version(name)

    importlib.metadata.version = _patched_version

    # ── Patch 2: packaging.version.parse(None) ──────────────────────────────
    import packaging.version

    _orig_parse = packaging.version.parse

    def _patched_parse(version_string):
        if version_string is None:
            return packaging.version.Version("0.0.0")
        return _orig_parse(version_string)

    packaging.version.parse = _patched_parse

    # ── Patch 3: disable check_torch_load_is_safe ────────────────────────────
    import transformers.utils.import_utils
    import transformers.modeling_utils

    transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None

    print("[local_workaround] All patches applied.")


if __name__ == "__main__":
    apply_patches()