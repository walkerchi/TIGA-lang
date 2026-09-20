"""Protocol regression tests; hardware measurements remain separate artifacts."""
from unittest.mock import patch

import numpy as np
import pytest

from benchmarks.distributed.paper_scaling import case, mesh_case, MeasuredNCCL
from benchmarks.memory_hierarchy.profile_paging import ExclusiveTimer


def test_exclusive_timer_subtracts_nested_time():
    timer = ExclusiveTimer()
    timer.active = True
    inner = timer.wrap(lambda: 42, 'inner')
    outer = timer.wrap(inner, 'outer')
    with patch('benchmarks.memory_hierarchy.profile_paging.time.perf_counter', side_effect=[0.,1.,3.,5.]):
        assert outer() == 42
    assert dict(timer.seconds) == {'inner':2., 'outer':3.}
    assert not timer.stack


def test_exclusive_timer_restores_stack_on_failure():
    timer = ExclusiveTimer()
    timer.active = True
    def fail():
        raise ValueError('test')
    with pytest.raises(ValueError, match='test'):
        timer.wrap(fail, 'failure')()
    assert not timer.stack and timer.calls['failure'] == 1


@pytest.mark.parametrize('boundary_every', [0,4])
def test_distributed_fixtures_and_full_oracle(boundary_every):
    rows, columns, values, expected = case(64, 3, boundary_every)
    np.testing.assert_array_equal(expected, values[columns.reshape(64,16)].mean(axis=1))
    np.testing.assert_array_equal(rows, np.arange(65)*16)
    left = columns.reshape(64,16)[:32]
    ghosts = np.unique(left[left >= 32])
    assert len(ghosts) == (16 if boundary_every == 0 else 32)


def test_nccl_byte_counts_and_control_are_separate():
    from types import SimpleNamespace
    class Base:
        rank, world_size = 0, 2
        def exchange_device(self, sends, receives, *, stream):
            assert stream == 'stream'
            return 'event'
    control = object()
    wrapped = MeasuredNCCL(Base(), control)
    region = SimpleNamespace(byte_count=128)
    assert wrapped.exchange_device([(1,region)], [(1,region)], stream='stream') == 'event'
    assert wrapped.control is control
    assert [(e[0],e[1]) for e in wrapped.events] == [('send',128),('receive',128)]


@pytest.mark.parametrize('side',[4,6,8])
def test_spatial_mesh_surface_volume_and_exact_oracle(side):
    rows,cols,values,expected=mesh_case(side,3)
    n=side**3
    assert len(cols)==(3*side-2)**3-n
    assert int(np.diff(rows).max())==26
    destinations=np.repeat(np.arange(n),np.diff(rows))
    assert np.all(destinations!=cols)
    oracle=np.zeros_like(values)
    np.add.at(oracle,destinations,values[cols])
    np.testing.assert_array_equal(expected,oracle)
    keys=destinations*n+cols
    np.testing.assert_array_equal(np.sort(keys),np.sort(cols*n+destinations))
    for rank in (0,1):
        begin,end=rank*(n//2),(rank+1)*(n//2)
        local=cols[rows[begin]:rows[end]]
        ghosts=np.unique(local[(local<begin)|(local>=end)])
        assert len(ghosts)==side**2
        boundary=np.unique(destinations[(destinations>=begin)&(destinations<end)&((cols<begin)|(cols>=end))])
        assert len(boundary)==side**2
