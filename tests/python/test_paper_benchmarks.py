"""Paper benchmark probes must measure executed, checked paths in isolated outputs."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_capacity_sweep_fresh_process_smoke(tmp_path):
    output = tmp_path / 'output'
    command = [sys.executable, '-m', 'benchmarks.memory_hierarchy.capacity_scaling',
               '--sizes', '128', '--pages', '16', '--repeats', '1', '--budget-mib', '1',
               '--cache-dir', str(tmp_path / 'cache'), '--output', str(output)]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, timeout=60,
                   env={**os.environ, 'TIGA_TENSOR_BACKEND': 'native'})
    data = json.loads((output / 'results.json').read_text())
    assert len(data['rows']) == 2
    for row in data['rows']:
        assert row['status'] == 'ok'
        assert row['max_abs_error'] == 0
        assert row['total_s'] == pytest.approx(row['load_s'] + row['first_call_s'])
        assert row['memory_report']['peak_bytes']['ram'] <= row['budget_bytes']


def test_gpu_scaling_smoke(tmp_path):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    output = tmp_path / 'gpu.json'
    subprocess.run([sys.executable, '-m', 'benchmarks.sparse_compute.paper_scaling',
                    '--sizes', '256', '--features', '16', '--repeats', '2',
                    '--output', str(output)], cwd=ROOT, check=True, capture_output=True, timeout=90)
    rows = json.loads(output.read_text())['rows']
    assert {r['provider'] for r in rows} == {'tiga.auto', 'torch.sparse.mm', 'torch.index_add'}
    assert all(r['correct'] and len(r['samples_ms']) == 2 and r['median_ms'] > 0 for r in rows)
