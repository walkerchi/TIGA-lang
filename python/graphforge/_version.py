"""Installed distribution version without importing build-time tooling."""

from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("graphforge-compiler")
except PackageNotFoundError:
    __version__ = "0.1.0a1+source"
