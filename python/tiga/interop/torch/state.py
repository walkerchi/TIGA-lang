"""Conservative cache invalidation for versioned and inference tensors."""


def tensor_version(tensor):
    """Return a comparable version, or a fresh token when caching is unsafe.

    Inference tensors have no mutation counter. A constant sentinel would
    silently reuse stale topology/weights after an in-place update, so each
    observation must invalidate data-dependent caches instead.
    """
    if tensor.is_inference():
        return object()
    return tensor._version
