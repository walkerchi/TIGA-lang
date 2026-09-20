"""Ranked executable reuse guards and live-coordinate regression coverage."""
from unittest.mock import Mock, patch

import pytest
import torch
import tiga as tg

from tiga.interop.torch.graph import Graph
from tiga.interop.torch.message_passing import (
    _EXECUTABLE_MISS, _RankedExecutable, _field_spec,
)


def executable(positions, source, *, k=3):
    graph = Graph.knn(positions, k)
    runner = Mock(return_value='ran')
    return _RankedExecutable(
        graph=graph, bindings=(('src','x'),), specs=(_field_spec(source),),
        variant=None, runner=runner,
        position_specs=(_field_spec(positions), _field_spec(positions)),
        relation_spec=_RankedExecutable.relation_signature(graph),
        field_specs={'src': {'x': _field_spec(source)}, 'dst': {}, 'edge': {}},
    )


def test_equivalent_new_graph_rebinds_live_coordinates():
    p, x = torch.rand(16,3), torch.rand(16)
    plan = executable(p,x)
    q = p.clone().add_(2)
    assert plan.try_run(Graph.knn(q,3), {'x':x}, {}, {}) == 'ran'
    assert plan.runner.call_args.args[0] is q
    assert plan.runner.call_args.args[1] is q


@pytest.mark.parametrize('change', ['k','self','bipartite','size','width','dtype',
    'stride','position_grad','source_grad','extra_field','missing_field','param'])
def test_incompatible_ranked_rebind_misses(change):
    p,x=torch.rand(16,3),torch.rand(16)
    plan=executable(p,x)
    kw={'k':3}; src={'x':x}; params={}
    if change=='k': kw['k']=4
    if change=='self': kw['exclude_self']=False
    if change=='bipartite': kw['candidates']=p.clone()
    if change=='size': p=torch.rand(17,3)
    if change=='width': p=torch.rand(16,2)
    if change=='dtype': p=p.double()
    if change=='stride': p=torch.rand(16,6)[:,::2]
    if change=='position_grad': p.requires_grad_()
    if change=='source_grad': x.requires_grad_()
    if change=='extra_field': src['extra']=x
    if change=='missing_field': src={}
    if change=='param': params={'scale':2}
    assert plan.try_run(Graph.knn(p,**kw),src,{}, {},params) is _EXECUTABLE_MISS
    plan.runner.assert_not_called()


class NeighborSum(tg.MessagePassing):
    reducer=tg.sum()
    def edge(self,src,dst,edge):
        return src.x


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_new_graph_calls_do_not_reenter_compilation_and_keep_old_outputs():
    rng=torch.Generator(device='cuda').manual_seed(73)
    x=torch.rand(129,device='cuda',generator=rng)
    kernel=NeighborSum()
    saved=[]
    for step in range(3):
        p=torch.rand(129,3,device='cuda',generator=rng)
        if step:
            # A cache hit must bypass capture as well as provider preparation.
            with patch('tiga.compiler.domain_capture.capture_message_passing',
                       side_effect=AssertionError('unexpected recapture')):
                actual=kernel(graph=tg.Graph.knn(p,13),src={'x':x},dst={})
        else:
            actual=kernel(graph=tg.Graph.knn(p,13),src={'x':x},dst={})
        d=torch.cdist(p,p,compute_mode='donot_use_mm_for_euclid_dist')
        d.fill_diagonal_(float('inf'))
        expected=x[d.topk(13,largest=False).indices].sum(1)
        torch.testing.assert_close(actual,expected,rtol=3e-4,atol=3e-4)
        saved.append((actual,expected))
    for actual,expected in saved:
        torch.testing.assert_close(actual,expected,rtol=3e-4,atol=3e-4)
    assert kernel.cache_info['misses']==1
    assert kernel.cache_info['hits']==2


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('k',[1,13,16,63])
@pytest.mark.parametrize('order',['ascending','descending','ties'])
def test_pruned_ranked_tiles_preserve_exact_selection(k,order):
    # Several complete tiles plus a tail: ordered input exercises both pruning
    # and late replacement; ties exercise the source-index part of the key.
    p=torch.arange(1031,device='cuda',dtype=torch.float32)[:,None]
    if order=='descending': p=p.flip(0)
    if order=='ties': p=(p%3).contiguous()
    query=torch.tensor([[0.],[350.]],device='cuda')
    x=torch.arange(len(p),device='cuda',dtype=torch.float32)
    kernel=NeighborSum()
    actual=kernel(graph=tg.Graph.knn(query,k,candidates=p),src={'x':x},dst={})
    d=(query-p.T).square()
    selected=d.argsort(dim=1,stable=True)[:,:k]
    torch.testing.assert_close(actual,x[selected].sum(1))
    assert '%may_improve' in kernel.ir('gf.kernel.ttir')
