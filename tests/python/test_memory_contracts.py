"""Residency must preserve logical values, device and gradient semantics."""
import gc
from threading import Event, Thread
import pytest
import tiga as tg
from tiga.runtime.memory import parse_bytes


@pytest.mark.parametrize("backend", ["python", "native"])
def test_spill_preserves_autograd(tmp_path, monkeypatch, backend):
    monkeypatch.setenv("TIGA_TENSOR_BACKEND", backend)
    with tg.execution(spill_dir=tmp_path):
        x = tg.tensor([2., 3.], requires_grad=True)
        y = (x * x).sum()
        y.spill()
        assert not y.is_leaf
        assert tg.autograd.grad(y, x).tolist() == [4., 6.]
        assert (y * 2).tolist() == 26.
        y.realize()
        assert not list(tmp_path.iterdir())


def test_prefetch_inherits_budget():
    from tiga.message_passing.paged import _stream_pages
    with tg.execution(memory={"host": 0}):
        with pytest.raises(MemoryError, match="managed ram"):
            list(_stream_pages(lambda begin, end: tg.empty(1), [(0, 1), (1, 2)],
                               prefetch=True, prefetch_depth=1))


def test_noncontiguous_snapshot(tmp_path):
    x = tg.tensor([[1., 2.], [3., 4.]]).transpose(0, 1)
    path = tmp_path / "view.tiga"
    x.save(path)
    assert tg.load(path).tolist() == [[1., 3.], [2., 4.]]


def test_views_retain_capacity(tmp_path):
    with tg.execution(memory={"host": 16, "nvme": 32}, spill_dir=tmp_path) as run:
        x = tg.tensor([1., 2., 3., 4.])
        view = x.reshape(2, 2)
        x.spill()
        assert run.memory_report()["live_bytes"]["ram"] == 16
        with pytest.raises(MemoryError):
            tg.empty(1)
        del view
        gc.collect()
        assert run.memory_report()["live_bytes"]["ram"] == 0
        x.realize()
        assert x.tolist() == [1., 2., 3., 4.]
        assert run.memory_report()["live_bytes"]["nvme"] == 0


def test_context_does_not_invalidate_outputs():
    with tg.execution(memory={"ram": 8}) as run:
        x = tg.tensor([1., 2.])
    assert x.tolist() == [1., 2.]
    assert run.memory_report()["live_bytes"]["ram"] == 8
    del x
    gc.collect()
    assert run.memory_report()["live_bytes"]["ram"] == 0


def test_snapshot_is_lazy_and_stable(tmp_path):
    path = tmp_path / "state.tiga"
    x = tg.tensor([1., 2.], requires_grad=True)
    tg.save(x, path)
    with tg.execution(memory={"host": 0}):
        attached = tg.load(path)
        assert attached._buffer is None
    with pytest.raises(FileExistsError):
        tg.save(tg.tensor([7., 8.]), path)
    tg.save(tg.tensor([7., 8.]), path, overwrite=True)
    assert attached.tolist() == [1., 2.]
    assert tg.load(path).tolist() == [7., 8.]
    assert not attached.requires_grad and x.requires_grad


def test_named_collision_and_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("TIGA_SPILL_DIR", str(tmp_path))
    first = tg.tensor([1.]).disk(name="state")
    with pytest.raises(FileExistsError):
        tg.tensor([2.]).disk(name="state")
    assert first.tolist() == [1.]
    for name in ("../state", "/state", "a\\b", ""):
        with pytest.raises(ValueError):
            tg.from_disk(name)


def test_cleanup_and_failed_budget(tmp_path):
    with tg.execution(memory={"nvme": 4}, spill_dir=tmp_path):
        x = tg.tensor([1., 2.])
        with pytest.raises(MemoryError):
            x.spill()
        assert x.tolist() == [1., 2.]
        assert not list(tmp_path.iterdir())
        y = tg.tensor([3.]).spill()
        del y
        gc.collect()
        assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("value", [-1, True, 1.5, "1GB", "-1MiB"])
def test_invalid_budget(value):
    with pytest.raises(ValueError):
        parse_bytes(value)


def test_tier_aliases_and_nested_context():
    assert parse_bytes("2MiB") == 2 << 20
    for alias, expected in (("hbm", "device"), ("host", "ram"), ("ssd", "nvme"), ("pinned_ram", "host-pinned")):
        assert tg.runtime.MemoryTier.parse(alias).value == expected
    with tg.execution():
        with pytest.raises(RuntimeError, match="nested"):
            with tg.execution():
                pass


def test_failed_transfer_still_releases_capacity(tmp_path, monkeypatch):
    with tg.execution(memory={"ram": 4, "nvme": 4}) as run:
        rt = tg.runtime.HierarchyRuntime(nvme_directory=tmp_path)
        source = rt.allocate("x", 0, tier="ram", capacity_bytes=4)
        destination = rt.allocate("x", 0, tier="nvme", capacity_bytes=4)
        def fail(*args):
            raise OSError("injected I/O failure")
        monkeypatch.setattr(rt, "_file_transfer", fail)
        rt.transfer(source, destination)
        with pytest.raises(OSError, match="injected"):
            rt.close()
        assert not list(tmp_path.iterdir())
        assert not any(run.memory_report()["live_bytes"].values())


def test_source_close_waits_for_transfer(tmp_path, monkeypatch):
    rt = tg.runtime.HierarchyRuntime(nvme_directory=tmp_path)
    source = rt.allocate("x", 0, tier="ram", capacity_bytes=4)
    destination = rt.allocate("x", 0, tier="nvme", capacity_bytes=4)
    source.buffer.write(b"data")
    started, proceed, closed = Event(), Event(), Event()
    original = rt._file_transfer
    def delayed(src, dst):
        started.set()
        assert proceed.wait(5)
        original(src, dst)
    monkeypatch.setattr(rt, "_file_transfer", delayed)
    completion = rt.transfer(source, destination)
    assert started.wait(5)
    def close_source():
        source.close()
        closed.set()
    thread = Thread(target=close_source)
    thread.start()
    try:
        assert not closed.wait(.02)
    finally:
        proceed.set()
        thread.join(5)
    completion.wait()
    assert destination.path.read_bytes() == b"data"
    rt.close()


@pytest.mark.parametrize("backend", ["native", "python"])
def test_cuda_copy_and_spill(tmp_path, monkeypatch, backend):
    monkeypatch.setenv("TIGA_TENSOR_BACKEND", backend)
    try:
        tg.runtime.cuda_compute_capability()
    except RuntimeError:
        pytest.skip("CUDA unavailable")
    with tg.execution(device="cuda", memory={"device": "1MiB"}, spill_dir=tmp_path):
        x = tg.tensor([2., 3.], requires_grad=True)
        assert str(x.device) == "cuda:0"
        cpu = x.cpu()
        assert str(cpu.device) == "cpu:0" and cpu is not x
        assert tg.autograd.grad((cpu * cpu).sum(), x).tolist() == [4., 6.]
        x.spill()
        assert str(x.device) == "cuda:0"
        assert x.tolist() == [2., 3.]
        assert str(x.device) == "cuda:0"
