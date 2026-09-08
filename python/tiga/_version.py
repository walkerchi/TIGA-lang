"""Installed distribution version without importing build-time tooling."""

from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("tiga-lang")
except PackageNotFoundError:
    __version__ = "0.1.0+source"
