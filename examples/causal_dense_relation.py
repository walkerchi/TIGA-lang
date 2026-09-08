"""A causal reducer expressed as a triangular Graph, not a core NN operator."""

import torch

import tiga as gf

torch.manual_seed(7)
nodes, heads, width = 256, 2, 64
# the streaming kernel reads lane-major storage: (H, N, D) viewed as (N, H, D)
query = torch.randn(heads, nodes, width, device="cuda", dtype=torch.float16).permute(1, 0, 2)
key = torch.randn(heads, nodes, width, device="cuda", dtype=torch.float16).permute(1, 0, 2)
value = torch.randn(heads, nodes, width, device="cuda", dtype=torch.float16).permute(1, 0, 2)


# --8<-- [start:core]
class CausalWeightedValue(gf.MessagePassing):
    reducer = gf.online_softmax()

    def edge(self, src, dst, edge, scale):
        score = (src.key * dst.query).sum(dim=-1) * scale
        return self.reducer(score, src.value)


kernel = CausalWeightedValue()
output = kernel(  # (N, H, D)
    graph=gf.Graph.triangular(nodes, device="cuda"),
    src={"key": key, "value": value},
    dst={"query": query},
    scale=width**-0.5,
)
# --8<-- [end:core]

# Inspect: graph.explain()   relation summary (triangular, edges=N(N+1)/2)
#          kernel.explain()  provider/lowering/cache summary
#          kernel.ir("gf.kernel.ttir")   generated TTIR streaming kernel


def main() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not torch.cuda.is_available():
        raise SystemExit("this CUDA compiler example requires a CUDA device")
    return output, query, key, value


if __name__ == "__main__":
    main()
