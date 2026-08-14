"""A causal reducer expressed as a triangular Graph, not a core NN operator."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import graphforge as gf


class CausalWeightedValue(gf.MessagePassing):
    reducer = gf.online_softmax()

    def edge(self, src, dst, edge, scale):
        del edge
        score = (src.key * dst.query).sum(dim=-1) * scale
        return self.reducer(score, src.value)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("this CUDA compiler example requires a CUDA device")
    heads, nodes, width = 2, 256, 64
    generator = torch.Generator(device="cuda").manual_seed(7)
    query = torch.randn(
        heads, nodes, width, device="cuda", dtype=torch.float16,
        generator=generator)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    # Torch supplies external storage in this example. GraphForge owns graph
    # semantics, UDF capture, Domain→Kernel lowering, TTIR and the launch.
    graph = gf.Graph.triangular(nodes, device="cuda")
    kernel = CausalWeightedValue()
    output = kernel(
        graph=graph,
        src={"key": key.permute(1, 0, 2),
             "value": value.permute(1, 0, 2)},
        dst={"query": query.permute(1, 0, 2)},
        scale=width**-0.5,
    ).permute(1, 0, 2)
    expected = F.scaled_dot_product_attention(
        query, key, value, is_causal=True)
    torch.testing.assert_close(output, expected, rtol=4e-3, atol=4e-3)

    print(graph.explain())
    print(kernel.explain())
    print(kernel.ir("gf.kernel.ttir").splitlines()[0])


if __name__ == "__main__":
    main()
