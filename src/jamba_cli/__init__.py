"""Jamba PDF chat CLI package."""

from importlib.metadata import version, PackageNotFoundError

PACKAGE_NAME = "jambashrimp"

try:
    __version__ = version(PACKAGE_NAME)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]

