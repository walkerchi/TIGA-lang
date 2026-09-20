"""Automatic whole-value paging: correctness before overlap/performance."""
import gc

import pytest
import tiga as tg


def test_lru_restores_and_releases_files(tmp_path):
    with tg.execution(memory={"ram": 16, "nvme": 64}, spill_dir=tmp_path,
                      eviction="lru") as run:
        x = tg.tensor([1., 2.])
        y = tg.tensor([3., 4.])
        assert x.tolist() == [1., 2.]  # y is now least recently used
        z = tg.tensor([5., 6.])
        assert not y.residency["resident"]
        assert x.residency["resident"]
        for value, expected in ((y, [3., 4.]), (x, [1., 2.]), (z, [5., 6.])):
            assert value.tolist() == expected
        report = run.memory_report()
        assert report["peak_bytes"]["ram"] <= 16
        assert report["evictions"] >= 3
        assert report["restores"] >= 3
        del value, x, y, z
        gc.collect()
        assert run.memory_report()["live_bytes"] == dict.fromkeys(report["live_bytes"], 0)
        assert not list(tmp_path.iterdir())


def test_inputs_are_pinned_and_oversized_working_set_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("TIGA_TENSOR_BACKEND", "python")
    with tg.execution(memory={"ram": 16}, spill_dir=tmp_path, eviction="lru"):
        x, y = tg.tensor([1., 2.]), tg.tensor([3., 4.])
        with pytest.raises(MemoryError, match="budget exceeded"):
            (x + y).realize()
        assert x.residency["resident"] and y.residency["resident"]


def test_automatic_spill_preserves_gradient(tmp_path, monkeypatch):
    monkeypatch.setenv("TIGA_TENSOR_BACKEND", "python")
    with tg.execution(memory={"ram": 32}, spill_dir=tmp_path, eviction="lru"):
        x = tg.tensor([2., 3.], requires_grad=True)
        unused = [tg.tensor([10., 20.]) for _ in range(4)]
        assert not x.residency["resident"]
        assert (x * x).tolist() == [4., 9.]
        assert tg.autograd.grad((x * x).sum(), x).tolist() == [4., 6.]
        assert unused[-1].tolist() == [10., 20.]


def test_disk_budget_failure_preserves_original(tmp_path):
    with tg.execution(memory={"ram": 8, "nvme": 0}, spill_dir=tmp_path, eviction="lru"):
        x = tg.tensor([1., 2.])
        with pytest.raises(MemoryError, match="nvme budget"):
            tg.tensor([3., 4.])
        assert x.tolist() == [1., 2.]
        assert not list(tmp_path.iterdir())


def test_alias_input_stays_pinned(tmp_path, monkeypatch):
    monkeypatch.setenv("TIGA_TENSOR_BACKEND", "python")
    with tg.execution(memory={"ram": 16}, spill_dir=tmp_path, eviction="lru"):
        x = tg.tensor([1., 2.])
        alias = x.reshape((2,))
        assert (alias + alias).tolist() == [2., 4.]
        assert x.tolist() == [1., 2.]


def test_policy_validation_and_prepared_storage(tmp_path):
    with pytest.raises(ValueError, match="spill_dir"):
        tg.execution(eviction="lru")
    with pytest.raises(ValueError, match="eviction"):
        tg.execution(eviction="unknown")
    with tg.execution(eviction="lru", spill_dir=tmp_path):
        with pytest.raises(RuntimeError, match="binds storage"):
            tg.tensor([1.]).prepare()


def test_cuda_lru_native_forward_backward(tmp_path, monkeypatch):
    try:
        tg.runtime.cuda_compute_capability()
    except RuntimeError:
        pytest.skip("CUDA unavailable")
    monkeypatch.setenv("TIGA_TENSOR_BACKEND", "native")
    with tg.execution(device="cuda", memory={"device": 128, "nvme": 1024},
                      spill_dir=tmp_path, eviction="lru") as run:
        x = tg.tensor([2., 3.], requires_grad=True)
        unused = [tg.tensor([10., 20.]) for _ in range(20)]
        assert not x.residency["resident"]
        output = x * x
        assert output.tolist() == [4., 9.]
        gradient = tg.autograd.grad(output.sum(), x)
        assert gradient.tolist() == [4., 6.]
        assert output.execution["backend"] == "cuda-ttir-triton"
        assert gradient.execution["backend"] == "cuda-ttir-triton"
        assert unused[-1].tolist() == [10., 20.]
        assert run.memory_report()["peak_bytes"]["device"] <= 128
        assert run.memory_report()["evictions"] > 0
        assert run.memory_report()["restores"] > 0
