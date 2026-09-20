"""Chart evidence must be portable and comparisons must remain matched."""
import hashlib
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_archived_chart_evidence():
    archive = ROOT / "benchmarks/evidence_snapshot"
    index = json.loads((archive / "index.json").read_text())
    assert len(index["files"]) == 50
    paths = set()
    for item in index["files"]:
        paths.add(item["path"])
        raw = (archive / item["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == item["sha256"]
        json.loads(raw)
    manifest = json.loads((ROOT / "benchmarks/evidence_manifest.json").read_text())
    for op, record in manifest["operations"].items():
        for case in record.get("cases", []):
            assert f"output/roofline/{op}/{case}/{record.get('artifact', 'roofline.json')}" in paths
    inputs = json.loads((ROOT / "docs/assets/results/chart-inputs.json").read_text())
    assert len(inputs["charts"]) == 12
    for stem, sources in inputs["charts"].items():
        assert sources and (ROOT / f"docs/assets/results/{stem}.svg").is_file()
    for item in inputs["files"]:
        assert hashlib.sha256((archive / "output" / item["path"]).read_bytes()).hexdigest() == item["sha256"]


def test_latency_buckets_do_not_mix_shapes_or_phases():
    pytest.importorskip("matplotlib")
    from benchmarks.common.plotting import _latency_panels
    row = {"provider": "tiga.auto", "cache": "hot", "features": 1, "milliseconds": 1}
    panels = _latency_panels([{**row, "nodes": 8}, {**row, "nodes": 16},
                             {**row, "nodes": 16, "phase": "build"}])
    assert len(panels) == 3
    with pytest.raises(ValueError, match="ambiguous"):
        _latency_panels([row, row])


def test_full_matrix_does_not_compare_different_shapes():
    from benchmarks.common.full_matrix import _provider_rows, LABELS
    rows = _provider_rows("sizes", [
        {"provider": "tiga.auto", "nodes": 8, "milliseconds": 2},
        {"provider": "torch.peer", "nodes": 8, "milliseconds": 4},
        {"provider": "tiga.auto", "nodes": 16, "milliseconds": 8},
        {"provider": "torch.peer", "nodes": 16, "milliseconds": 2},
    ], LABELS["en"])
    candidates = [r for r in rows if "`tiga.auto`" in r]
    assert "2.00×" in candidates[0]
    assert "0.25×" in candidates[1]


def test_weighted_archive_gate_arithmetic_and_matching():
    from benchmarks.common.perf_protocol import BUCKET_FIELDS
    root = ROOT / "benchmarks/evidence_snapshot/output/roofline/weighted_aggregation"
    for path in root.glob("*/roofline.json"):
        payload = json.loads(path.read_text())
        for gate in payload.get("sota_gates", []):
            for key in ("candidate", "baseline"):
                matches = [r for r in payload["results"] if r["provider"] == gate[key]
                           and all(r.get(k) == gate.get(k) for k in BUCKET_FIELDS)]
                assert len(matches) == 1, (path, gate)
                assert matches[0]["milliseconds"] == pytest.approx(gate[key + "_ms"])
            ratio = gate["baseline_ms"] / gate["candidate_ms"]
            assert gate["speedup_vs_sota"] == pytest.approx(ratio)
            assert gate["speedup_ci_low"] <= gate["speedup_ci_high"]
            assert gate["passed"] == (ratio >= gate["threshold"]
                                      and gate["speedup_ci_low"] >= gate["threshold"])


def test_full_matrix_is_reproducible_from_archived_inputs():
    from benchmarks.common.full_matrix import build_matrix
    manifest = json.loads((ROOT / "benchmarks/evidence_manifest.json").read_text())
    root = ROOT / "benchmarks/evidence_snapshot/output/roofline"
    for lang, suffix in (("en", ""), ("zh", ".zh")):
        generated = build_matrix(manifest, root, lang)
        assert generated == (ROOT / f"docs/includes/full-matrix{suffix}.md").read_text()
        assert '??? info "' in generated


def test_documentation_ratios_are_selected_from_raw_data(monkeypatch):
    pytest.importorskip("matplotlib")
    from benchmarks.common import plot_docs_results as charts
    monkeypatch.setattr(charts, "ROOT", ROOT / "benchmarks/evidence_snapshot/output")
    ratio, _ = charts._point("cpu_relation", "regular_permuted_i32_n131072_degree16_t16",
                             "tiga.llvm.parallel_relation", "scipy.csr_matvec")
    assert ratio > 0
    with pytest.raises(ValueError, match="ambiguous/missing chart point"):
        charts._point("weighted_aggregation", "irregular_random_cuda_i32_n131072_degree16_f16_64",
                      "tiga.prepared_auto", "torch.sparse.mm")


def test_comparison_uses_latency_and_keeps_reference_in_bilingual_tables(monkeypatch, tmp_path):
    pytest.importorskip("matplotlib")
    from benchmarks.common import plot_docs_results as charts
    import re
    monkeypatch.setattr(charts, "ROOT", ROOT / "benchmarks/evidence_snapshot/output")
    monkeypatch.setattr(charts, "OUT", tmp_path)
    payload = charts._case_json("weighted_aggregation",
                               "irregular_random_cuda_i32_n131072_degree16_f16_64")
    rows = [r for r in payload["results"] if r["cache"] == "hot" and r["features"] == 16]
    expected = sorted(r["milliseconds"] for r in rows if r["provider"] != "tiga.reference")
    save = charts._save

    def check_and_save(fig, stem):
        ax = fig.axes[0]
        assert [bar.get_width() for bar in ax.patches] == pytest.approx(expected)
        assert ax.get_xlim()[0] == 0
        assert ax.yaxis_inverted()
        assert ax.get_xlabel() == "Execution time (ms) — shorter is faster"
        assert len({bar.get_facecolor() for bar in ax.patches}) == 2
        assert [label.get_text() for label in ax.get_yticklabels()][:2] == [
            "Tiga (prepared)*", "Tiga (regular call)"]
        save(fig, stem)

    monkeypatch.setattr(charts, "_save", check_and_save)
    charts._style()
    charts.figure_provider_comparison()
    svg = (tmp_path / "spmm-provider-compare.svg").read_text()
    assert "speedup" not in svg and "tiga.reference" not in svg
    assert "preparation is outside timing" in svg
    assert (tmp_path / "spmm-provider-compare.png").is_file()
    for suffix in ("", ".zh"):
        source = (ROOT / f"docs/comparison{suffix}.md").read_text()
        values = dict(re.findall(r"\| `([^`]+)` \|[^\n]*\| ([0-9.]+) \|", source))
        assert set(values) == {row["provider"] for row in rows}
        for row in rows:
            assert values[row["provider"]] == f"{row['milliseconds']:.3f}"
