"""Guard tests for the declared public API surface (audit #20).

A stale `__all__` entry is worse than none: it *claims* an export exists.
Every name declared by any package's `__all__` must resolve as an attribute
of that package.
"""

import importlib

import src
import src.core
import src.providers
import src.structured
import src.tools

DECLARED_PACKAGES = [src, src.core, src.providers, src.structured, src.tools]


def test_every_declared_public_name_resolves_on_its_package():
    for package in DECLARED_PACKAGES:
        for name in package.__all__:
            assert hasattr(package, name), f"{package.__name__}.{name} is declared in __all__ but does not exist"


def test_top_level_package_exports_nothing_by_design():
    # src/ is a namespace: the public surface is the subpackages' __all__
    # lists. Keeping the top level empty means one import of `src` never
    # drags in every module.
    assert src.__all__ == []
