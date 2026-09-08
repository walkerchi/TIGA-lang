from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import tiga as gf
import pytest


class WeightedSmoothing(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        return edge.w * src.x

    def node(self, dst, aggregate):
        return aggregate + dst.bias


class Mean(gf.Reducer):
    associative = True
    commutative = True

    def identity(self):
        return 0.0, 0.0

    def lift(self, value):
        return value, 1.0

    def combine(self, left, right):
        return left[0] + right[0], left[1] + right[1]

    def finalize(self, state):
        return state[0] / state[1]


class MeanAggregation(gf.MessagePassing):
    reducer = Mean()

    def edge(self, src, dst, edge):
        del dst
        return self.reducer(edge.w * src.x)


def _stencil_case(tmp_path):
    """60x40 five-point stencil on disk plus matching random fields."""
    graph = gf.Graph.stencil(
        (60, 40), ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)))
    rng = random.Random(20240817)
    weights = gf.tensor([rng.random() for _ in range(graph.num_edges)])
    nodes = graph.schema.num_dst
    x = gf.tensor([rng.random() for _ in range(nodes)])
    bias = gf.tensor([rng.random() for _ in range(nodes)])
    path = tmp_path / "stencil.gfg"
    gf.save(graph, path)
    paged = gf.load(path)
    assert paged.schema.realization == "paged_csr"
    return graph, paged, x, bias, weights


def test_paged_sum_matches_materialized(tmp_path):
    graph, paged, x, bias, weights = _stencil_case(tmp_path)
    kernel = WeightedSmoothing()

    reference = kernel(
        graph=graph, src={"x": x}, dst={"bias": bias}, edge={"w": weights})
    # 997 does not divide 2400 and rows have edges reaching across pages.
    output = kernel(
        graph=paged, src={"x": x}, dst={"bias": bias}, edge={"w": weights},
        page_rows=997)

    assert output.tolist() == pytest.approx(reference.tolist(), abs=1e-6)


def test_paged_custom_mean_matches_materialized(tmp_path):
    graph, paged, x, _bias, weights = _stencil_case(tmp_path)
    kernel = MeanAggregation()

    reference = kernel(graph=graph, src={"x": x}, dst={}, edge={"w": weights})
    output = kernel(
        graph=paged, src={"x": x}, dst={}, edge={"w": weights}, page_rows=313)

    assert output.tolist() == pytest.approx(reference.tolist(), abs=1e-6)


def test_paged_default_and_env_page_rows(tmp_path, monkeypatch):
    graph, paged, x, bias, weights = _stencil_case(tmp_path)
    kernel = WeightedSmoothing()
    bindings = {"src": {"x": x}, "dst": {"bias": bias}, "edge": {"w": weights}}
    reference = kernel(graph=graph, **bindings)

    single_page = kernel(graph=paged, **bindings)  # default: 100k rows
    monkeypatch.setenv("TIGA_PAGED_PAGE_ROWS", "100")
    env_paged = kernel(graph=paged, **bindings)

    assert single_page.tolist() == pytest.approx(reference.tolist(), abs=1e-6)
    assert env_paged.tolist() == pytest.approx(reference.tolist(), abs=1e-6)


def test_paged_prefetch_is_result_neutral(tmp_path):
    _graph, paged, x, bias, weights = _stencil_case(tmp_path)
    kernel = WeightedSmoothing()
    bindings = {
        "graph": paged, "src": {"x": x}, "dst": {"bias": bias},
        "edge": {"w": weights}, "page_rows": 97,
    }

    prefetched = kernel(**bindings, prefetch=True)
    serial = kernel(**bindings, prefetch=False)

    assert prefetched.tolist() == serial.tolist()


def test_paged_backward_matches_materialized(tmp_path):
    """Paged autograd: src/dst/edge adjoints match the unpaged reference."""
    graph, paged, x, bias, weights = _stencil_case(tmp_path)
    kernel = WeightedSmoothing()
    x = gf.tensor(x.tolist(), requires_grad=True)
    bias = gf.tensor(bias.tolist(), requires_grad=True)
    weights = gf.tensor(weights.tolist(), requires_grad=True)
    bindings = {"src": {"x": x}, "dst": {"bias": bias}, "edge": {"w": weights}}

    reference = kernel(graph=graph, **bindings)
    ref_grads = gf.autograd.grad(reference.sum(), (x, bias, weights))

    # 997 does not divide 2400 and rows have edges reaching across pages.
    output = kernel(graph=paged, **bindings, page_rows=997)
    assert output.tolist() == pytest.approx(reference.tolist(), abs=1e-6)
    grads = gf.autograd.grad(output.sum(), (x, bias, weights))

    for paged_grad, ref_grad in zip(grads, ref_grads):
        assert paged_grad.tolist() == pytest.approx(ref_grad.tolist(), abs=1e-5)


def test_paged_backward_custom_mean_matches_materialized(tmp_path):
    """The general-reducer path also differentiates page-locally."""
    graph, paged, x, _bias, weights = _stencil_case(tmp_path)
    kernel = MeanAggregation()
    x = gf.tensor(x.tolist(), requires_grad=True)
    weights = gf.tensor(weights.tolist(), requires_grad=True)
    bindings = {"src": {"x": x}, "dst": {}, "edge": {"w": weights}}

    reference = kernel(graph=graph, **bindings)
    ref_grads = gf.autograd.grad(reference.sum(), (x, weights))

    output = kernel(graph=paged, **bindings, page_rows=313)
    assert output.tolist() == pytest.approx(reference.tolist(), abs=1e-6)
    grads = gf.autograd.grad(output.sum(), (x, weights))

    for paged_grad, ref_grad in zip(grads, ref_grads):
        assert paged_grad.tolist() == pytest.approx(ref_grad.tolist(), abs=1e-5)


def test_paged_field_shape_validation(tmp_path):
    _graph, paged, x, bias, weights = _stencil_case(tmp_path)
    kernel = WeightedSmoothing()

    with pytest.raises(ValueError, match="edge.w leading dimension"):
        kernel(
            graph=paged, src={"x": x}, dst={"bias": bias},
            edge={"w": gf.tensor([1.0, 2.0])})
    with pytest.raises(ValueError, match="dst.bias leading dimension"):
        kernel(
            graph=paged, src={"x": x}, dst={"bias": gf.tensor([1.0])},
            edge={"w": weights})


class ScaledSmoothing2D(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge, scale):
        return edge.w * src.x * scale

    def node(self, dst, aggregate):
        return aggregate + dst.bias


def _flat(values):
    if not isinstance(values, list):
        return [values]
    flat = []
    for row in values:
        flat.extend(row if isinstance(row, list) else [row])
    return flat


def test_paged_disk_fields_match_materialized(tmp_path):
    """Fields persisted in the .gfg page through the same forward numbers."""
    graph, _paged, x, bias, weights = _stencil_case(tmp_path)
    kernel = WeightedSmoothing()
    path = tmp_path / "stencil-fields.gfg"
    gf.save(graph, path, fields={
        "src": {"x": x}, "dst": {"bias": bias}, "edge": {"w": weights}})

    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "tiga.gfg.csr.v2"
    assert {(entry["role"], entry["name"]) for entry in manifest["fields"]} == {
        ("src", "x"), ("dst", "bias"), ("edge", "w")}

    paged = gf.load(path)
    shell = paged.fields("src")["x"]
    assert shell._buffer is None  # the payload stays on disk
    # An explicit read materializes the whole field as an escape hatch.
    assert shell.tolist() == pytest.approx(x.tolist())

    reference = kernel(
        graph=graph, src={"x": x}, dst={"bias": bias}, edge={"w": weights})
    bindings = {
        "graph": paged,
        "src": paged.fields("src"),
        "dst": paged.fields("dst"),
        "edge": paged.fields("edge"),
    }
    output = kernel(**bindings, page_rows=997)
    serial = kernel(**bindings, page_rows=997, prefetch=False)
    assert output.tolist() == pytest.approx(reference.tolist(), abs=1e-6)
    assert serial.tolist() == output.tolist()

    # Re-saving a fielded graph goes through the source-store copy path and
    # keeps the persisted fields.
    copied = tmp_path / "stencil-fields-copy.gfg"
    gf.save(paged, copied)
    repaged = gf.load(copied)
    recopied = kernel(
        graph=repaged, src=repaged.fields("src"), dst=repaged.fields("dst"),
        edge=repaged.fields("edge"), page_rows=997)
    assert recopied.tolist() == pytest.approx(reference.tolist(), abs=1e-6)


def test_paged_disk_fields_backward_matches_materialized(tmp_path):
    """Paged autograd over disk fields: src/dst/edge/param adjoints match."""
    graph = gf.Graph.stencil(
        (30, 25), ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)))
    rng = random.Random(4242)
    nodes, edges, feat = graph.schema.num_dst, graph.num_edges, 3
    x = gf.tensor(
        [[rng.random() for _ in range(feat)] for _ in range(nodes)],
        requires_grad=True)
    bias = gf.tensor(
        [[rng.random() for _ in range(feat)] for _ in range(nodes)],
        requires_grad=True)
    weights = gf.tensor(
        [[rng.random() for _ in range(feat)] for _ in range(edges)],
        requires_grad=True)
    scale = gf.tensor(1.7, requires_grad=True)
    grad_output = gf.tensor(
        [[rng.random() * 0.01 for _ in range(feat)] for _ in range(nodes)])

    kernel = ScaledSmoothing2D()
    reference = kernel(
        graph=graph, src={"x": x}, dst={"bias": bias}, edge={"w": weights},
        scale=scale)
    ref_grads = gf.autograd.grad(
        reference, (x, bias, weights, scale), grad_output=grad_output)

    path = tmp_path / "fields2d.gfg"
    gf.save(graph, path, fields={
        "src": {"x": x}, "dst": {"bias": bias}, "edge": {"w": weights}})
    paged = gf.load(path)
    src = paged.fields("src", requires_grad=True)
    dst = paged.fields("dst", requires_grad=True)
    edge = paged.fields("edge", requires_grad=True)
    # 97 does not divide 750 and rows have edges reaching across pages.
    output = kernel(
        graph=paged, src=src, dst=dst, edge=edge, scale=scale, page_rows=97)
    assert _flat(output.tolist()) == pytest.approx(
        _flat(reference.tolist()), abs=1e-5)

    grads = gf.autograd.grad(
        output,
        (src["x"], dst["bias"], edge["w"], scale),
        grad_output=grad_output,
    )
    for paged_grad, ref_grad in zip(grads, ref_grads):
        assert _flat(paged_grad.tolist()) == pytest.approx(
            _flat(ref_grad.tolist()), rel=1e-4, abs=1e-4)


_RSS_WORKER = r'''
import array
import json
import os
import random
import resource
import subprocess
import sys
from pathlib import Path

# tiga is imported lazily inside the workload modes so the "driver" mode
# stays tiny: ru_maxrss survives fork+exec, so the workers inherit the
# driver's (small) watermark instead of the pytest parent's.

mode, directory = sys.argv[1], sys.argv[2]
directory = Path(directory)
NUM_NODES = 200_000
FEAT = 32
PAGE_ROWS = 10_000


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def field_payload(count):
    unit = array.array("f", [(index % 977) * 0.001 for index in range(1024)])
    values = unit * (count // len(unit) + 1)
    del values[count:]
    return values.tobytes()


def tensor_from_bytes(payload, shape, dtype):
    gf = __import__("tiga")
    result = gf.empty(shape, dtype=dtype)
    result._buffer.write(payload)
    return result


def load_into(path, shape, dtype):
    gf = __import__("tiga")
    value = gf.empty(shape, dtype=dtype)
    fd = os.open(path, os.O_RDONLY)
    size = os.fstat(fd).st_size
    for offset in range(0, size, 16 << 20):  # chunked: no giant bytes object
        value._buffer.write(
            os.pread(fd, min(16 << 20, size - offset), offset), offset=offset)
    os.close(fd)
    return value


def run_workload(kind):
    import tiga as gf

    class Smoothing(gf.MessagePassing):
        reducer = gf.sum()

        def edge(self, src, dst, edge):
            return edge.w * src.x

        def node(self, dst, aggregate):
            return aggregate + dst.bias

    if kind == "materialized":
        root = directory / "powerlaw.gfg"
        manifest = json.loads((root / "manifest.json").read_text())
        num_edges = manifest["num_edges"]
        graph = gf.Graph.from_csr(
            load_into(root / "row_ptr.bin", (NUM_NODES + 1,), gf.int64),
            load_into(root / "col_idx.bin", (num_edges,), gf.int64),
            num_src=NUM_NODES,
        )
        assert graph.schema.realization == "materialized_csr"
        fields = {}
        for entry in manifest["fields"]:
            value = load_into(
                root / entry["file"], tuple(entry["shape"]), gf.float32)
            fields.setdefault(entry["role"], {})[entry["name"]] = value
        output = Smoothing()(
            graph=graph, src=fields["src"], dst=fields["dst"],
            edge=fields["edge"])
        checksum = output.sum().tolist()
        return {"checksum": checksum, "peak_mb": peak_mb()}

    baseline = peak_mb()
    paged = gf.load(directory / "powerlaw.gfg")
    output = Smoothing()(
        graph=paged,
        src=paged.fields("src"),
        dst=paged.fields("dst"),
        edge=paged.fields("edge"),
        page_rows=PAGE_ROWS,
    )
    checksum = output.sum().tolist()
    return {"checksum": checksum, "peak_mb": peak_mb(), "baseline_mb": baseline}


if mode == "driver":
    results = {}
    for sub in ("setup", "materialized", "paged"):
        completed = subprocess.run(
            [sys.executable, str(Path(__file__)), sub, str(directory)],
            capture_output=True, text=True, check=True)
        results[sub] = json.loads(completed.stdout.strip().splitlines()[-1])
    print(json.dumps(results))
elif mode == "setup":
    import tiga as gf

    rng = random.Random(20260907)
    rows = array.array("q", [0])
    columns = array.array("q")
    for _dst in range(NUM_NODES):
        degree = 1 + int(15 * rng.random() ** 2)  # skewed degree distribution
        for _ in range(degree):
            # power-law source popularity: low ranks attract most edges
            columns.append(min(NUM_NODES - 1, int(NUM_NODES * rng.random() ** 2.5)))
        rows.append(len(columns))
    graph = gf.Graph.from_csr(
        tensor_from_bytes(rows.tobytes(), (len(rows),), gf.int64),
        tensor_from_bytes(columns.tobytes(), (len(columns),), gf.int64),
        num_src=NUM_NODES,
    )
    num_edges = graph.num_edges
    gf.save(graph, directory / "powerlaw.gfg", fields={
        "src": {"x": tensor_from_bytes(
            field_payload(NUM_NODES * FEAT), (NUM_NODES, FEAT), gf.float32)},
        "dst": {"bias": tensor_from_bytes(
            field_payload(NUM_NODES * FEAT), (NUM_NODES, FEAT), gf.float32)},
        "edge": {"w": tensor_from_bytes(
            field_payload(num_edges * FEAT), (num_edges, FEAT), gf.float32)},
    })
    print(json.dumps({"edges": num_edges}))
else:
    print(json.dumps(run_workload(mode)))
'''


def test_power_law_paged_fields_bound_rss(tmp_path):
    """Fields paging keeps peak RSS bounded on a power-law graph.

    A small ``driver`` subprocess spawns the ``setup`` / ``materialized`` /
    ``paged`` workers as its own children: ru_maxrss survives fork+exec, so
    measuring from under the (possibly already-large) pytest process would
    inherit its watermark.  ``materialized`` runs the kernel with topology
    and fields fully in RAM; ``paged`` streams it with disk-backed fields.
    On a power-law graph a destination page's source set spans most of the
    node set, so per-page unique-source materialization could not bound
    memory — only the mmap-backed field paging can.
    """
    worker = tmp_path / "_rss_worker.py"
    worker.write_text(_RSS_WORKER, encoding="utf-8")
    env = os.environ.copy()
    repo_root = Path(gf.__file__).resolve().parent.parent
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), str(repo_root.parent), env.get("PYTHONPATH", "")])

    completed = subprocess.run(
        [sys.executable, str(worker), "driver", str(tmp_path)],
        env=env, capture_output=True, text=True, check=True)
    results = json.loads(completed.stdout.strip().splitlines()[-1])
    materialized = results["materialized"]
    paged = results["paged"]

    assert materialized["checksum"] == pytest.approx(paged["checksum"], rel=1e-4)
    assert paged["peak_mb"] < 0.6 * materialized["peak_mb"], (
        f"paged peak {paged['peak_mb']:.0f}MB vs materialized "
        f"{materialized['peak_mb']:.0f}MB")
