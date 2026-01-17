import os

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
real_agent = os.path.join(pkg_root, 'agent')
if os.path.isdir(real_agent) and real_agent not in __path__:
    __path__.append(real_agent)

__all__ = []
