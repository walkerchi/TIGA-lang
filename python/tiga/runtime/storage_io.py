"""Bounded staging for spill and compiler-planned file transfers."""

CHUNK_BYTES = 1 << 20


def buffer_to_file(buffer, stream, size, *, offset=0):
    for start in range(0, size, CHUNK_BYTES):
        stream.write(buffer.read(offset=offset + start, bytes=min(CHUNK_BYTES, size - start)))


def file_to_buffer(stream, buffer, size):
    for start in range(0, size, CHUNK_BYTES):
        payload = stream.read(min(CHUNK_BYTES, size - start))
        if len(payload) != min(CHUNK_BYTES, size - start):
            raise RuntimeError("short storage transfer")
        buffer.write(payload, offset=start)
