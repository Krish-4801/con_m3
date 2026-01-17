import os
import sys

# Append the real 'core' folder (project_root/core) to this package's __path__
pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
real_core = os.path.join(pkg_root, 'core')
if os.path.isdir(real_core) and real_core not in __path__:
    __path__.append(real_core)

# Optionally expose version or metadata here
__all__ = []
