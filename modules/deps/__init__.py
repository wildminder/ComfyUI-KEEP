"""Vendored third-party packages of ComfyUI-KEEP.

The packages below are **not** top-level packages: since Round 21 they are
imported only through ``src/keep_vendored.py`` under the pack-private name
``_keep_vendor`` - never as top-level ``basicsr`` / ``facelib`` and never
under the shared top-level ``modules`` name, so they never shadow (or get
shadowed by) an installed ``basicsr`` / ``facelib`` in ``sys.modules``
(P2.5).  The tree is instantiated once per process, anchored to the pack's
own files.
"""
