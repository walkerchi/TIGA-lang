import pytest
import torch

from benchmarks.graph_operations.torch_graph_baseline import knn_indices, radius_csr


@pytest.mark.parametrize('pair_budget',[37,111,10000])
def test_chunked_pure_torch_knn_matches_full_distance(pair_budget):
    p=torch.rand(37,3,generator=torch.Generator().manual_seed(7))
    d=(p[:,None,:]-p[None,:,:]).square().sum(-1)
    d.fill_diagonal_(float('inf'))
    expected=d.topk(5,largest=False).indices
    torch.testing.assert_close(knn_indices(p,5,pair_budget=pair_budget),expected)


@pytest.mark.parametrize('cutoff',[0.,.4,10.])
def test_chunked_pure_torch_radius_matches_full_distance(cutoff):
    p=torch.rand(37,3,generator=torch.Generator().manual_seed(8))
    d=(p[:,None,:]-p[None,:,:]).square().sum(-1)
    accept=d<cutoff**2
    accept.fill_diagonal_(False)
    expected=torch.where(accept,d.sqrt(),0)
    actual=radius_csr(p,cutoff,pair_budget=111)
    torch.testing.assert_close(actual.to_dense(),expected)
    assert actual.values().numel()==int(accept.sum())


def test_distance_budget_rejects_too_small_row():
    with pytest.raises(ValueError,match='complete candidate row'):
        knn_indices(torch.rand(10,3),2,pair_budget=9)
