import os

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
real_memory = os.path.join(pkg_root, 'memory')
if os.path.isdir(real_memory) and real_memory not in __path__:
    __path__.append(real_memory)

__all__ = []
