"""
Conclave package shim.

This package exposes existing top-level folders (core, agent, perception,
memory, api) under the `conclave` namespace so imports like
`from conclave.core.engine import ...` work without moving files.
"""
# This file intentionally left minimal; subpackage __init__.py files
# will extend __path__ to include the real implementation directories.

__all__ = ["core", "agent", "perception", "memory", "api"]
