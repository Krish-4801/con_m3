import os

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
real_api = os.path.join(pkg_root, 'api')
if os.path.isdir(real_api) and real_api not in __path__:
    __path__.append(real_api)

__all__ = []
