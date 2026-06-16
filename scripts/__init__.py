"""Marker so ``python -m scripts.<module>`` works (e.g. ``python -m scripts.calibrate_grader``).

Packaging-safe: ``[tool.setuptools.packages.find]`` includes only ``backend*`` /
``reprolab*``, so ``scripts`` is never built into the wheel. Existing scripts are
still runnable in path form (``python scripts/foo.py``) — this only *adds* the
module-form invocation the grader-fidelity handoff documents.
"""
