"""Tensor spill API: .disk() / .cpu() / gf.from_disk(name)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import tiga as gf
import pytest
import torch


def test_native_tensor_disk_roundtrip_releases_and_reloads():
    tensor = gf.tensor([1.0, 2.0, 3.0])
    tensor.disk()
    assert tensor._buffer is None
    # Any read transparently reloads; .cpu() is an explicit reload.
    assert tensor.tolist() == [1.0, 2.0, 3.0]
    assert tensor._buffer is not None


def test_disk_forces_deferred_expression():
    tensor = gf.tensor([1.0, 2.0]) * 3.0
    tensor.disk()
    assert tensor.tolist() == [3.0, 6.0]


def test_from_torch_chainable_write_then_read():
    source = torch.arange(4, dtype=torch.float32)
    tensor = gf.from_torch(source).disk().cpu()
    assert tensor.tolist() == [0.0, 1.0, 2.0, 3.0]


def test_strided_view_spills_logical_values():
    tensor = gf.tensor([[1.0, 2.0], [3.0, 4.0]]).transpose(0, 1)
    tensor.disk()
    assert tensor.tolist() == [[1.0, 3.0], [2.0, 4.0]]


def test_named_spill_survives_and_attaches(tmp_path, monkeypatch):
    monkeypatch.setenv("TIGA_SPILL_DIR", str(tmp_path))
    gf.tensor([5.0, 6.0, 7.0]).disk(name="ckpt")
    assert (tmp_path / "ckpt.gfspill").exists()
    attached = gf.from_disk("ckpt")
    assert attached.tolist() == [5.0, 6.0, 7.0]
    with pytest.raises(FileNotFoundError, match="missing"):
        gf.from_disk("missing")


def test_named_spill_crosses_processes(tmp_path):
    writer = (
        "import tiga as gf; "
        "gf.tensor([9.0, 8.0]).disk(name='shared')"
    )
    reader = (
        "import tiga as gf; "
        "t = gf.from_disk('shared'); "
        "assert t.tolist() == [9.0, 8.0], t.tolist()"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).parents[2] / "python")
    env["TIGA_SPILL_DIR"] = str(tmp_path)
    subprocess.run([sys.executable, "-c", writer], env=env, check=True)
    subprocess.run([sys.executable, "-c", reader], env=env, check=True)


def test_respill_requires_reload_first():
    tensor = gf.tensor([1.0]).disk()
    with pytest.raises(RuntimeError, match="already spilled"):
        tensor.disk()
    tensor.cpu().disk()  # reload, then spilling again is fine
    assert tensor.tolist() == [1.0]
