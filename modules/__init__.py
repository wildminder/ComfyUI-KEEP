"""The low-level shared modules of the KEEP pack.

Since P3.2 this package holds **only** the vendored third-party tree
(``modules/deps``): the engines moved to ``src/`` and the node classes to
``nodes/``.  The package marker exists so that the vendored tree is a real
subpackage of the node package (P2.5) - it must never be put on ``sys.path``.
"""
