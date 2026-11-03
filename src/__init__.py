"""The engines of the KEEP pack (P3.2).

``src/`` holds the *engines* - the model configs, the image conversions, the
model loader, the face processor, the path resolution and the logger.  The
node classes live in ``nodes/``; the vendored third-party tree lives in
``modules/deps/``.

The engines never define a node schema (the structure guard (b) enforces
that) and they never import ``nodes/``.
"""
