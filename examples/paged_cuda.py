"""Stored graph -> host-staged CUDA pages -> checked recurrent forward steps.

Requires the native CUDA provider and NumPy. Advanced storage uses tg.Tensor;
ordinary in-memory application code should continue using torch.Tensor.
"""
from pathlib import Path
import tempfile

import numpy as np
import tiga as tg


class Average(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        return src.x

    def node(self, dst, aggregate):
        return aggregate * (1.0 / 16)


def main():
    with tempfile.TemporaryDirectory(prefix='tiga-paged-cuda-') as directory:
        path = Path(directory) / 'ring.gfg'
        expected = (np.arange(4096) % 97).astype(np.float32)
        graph = tg.Graph.stencil((4096,), tuple((i,) for i in range(1,17)), periodic=True)
        tg.save(graph, path, fields={'src': {'x': tg.tensor(expected.tolist())}})
        del graph
        # --8<-- [start:core]
        # Ingest is complete before entering the constrained execution scope.
        with tg.execution(device='cuda', memory={'device': '128KiB'},
                          page_rows=128, prefetch_depth=1) as run:
            graph = tg.load(path, device='cuda')
            x = graph.fields('src')['x']  # lazy, no full-field device allocation
            for _ in range(3):
                x = Average()(graph=graph, src={'x': x}, dst={}, prefetch=False)
                expected = sum(np.roll(expected, -i) for i in range(1,17)) / 16
                np.testing.assert_allclose(x.to_numpy(), expected, rtol=3e-6, atol=3e-6)
            print(x.execution)
            print(run.memory_report())
            del x, graph
        # --8<-- [end:core]


if __name__ == '__main__':
    main()
