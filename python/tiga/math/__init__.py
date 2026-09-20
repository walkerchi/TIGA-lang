"""Tiga scalar/tensor intrinsics captured by compiler frontends."""

from __future__ import annotations


def exp(value):
    method = getattr(value, "exp", None)
    if method is not None:
        return method()
    raise TypeError("tg.math.exp expects a staged Tiga value")


def maximum(left, right):
    method = getattr(left, "maximum", None)
    if method is not None:
        return method(right)
    method = getattr(right, "maximum", None)
    if method is not None:
        return method(left)
    raise TypeError("tg.math.maximum expects at least one staged Tiga value")


__all__ = ["exp", "maximum"]
