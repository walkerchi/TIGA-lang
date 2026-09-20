"""Temporary residency is distinct from persistent value snapshots."""

# --8<-- [start:core]
from pathlib import Path
import tempfile

import tiga as tg

with tempfile.TemporaryDirectory(prefix="tiga-memory-example-") as directory:
    with tg.execution(memory={"host": "1MiB", "nvme": "1MiB"},
                      spill_dir=directory) as run:
        values = tg.tensor([float(i) for i in range(8)], requires_grad=True)
        loss = (values * values).sum()
        loss.spill()  # temporary residency; autograd history survives
        gradients = tg.autograd.grad(loss, values)
        assert gradients.tolist() == [float(2 * i) for i in range(8)]
        back = values.cpu()
        path = Path(directory) / "activations.tiga"
        tg.save(values, path)  # persistent value snapshot, not an eviction
        restored = tg.load(path)  # metadata only until first observation
        assert restored.tolist() == back.tolist()
        print(run.memory_report())
# --8<-- [end:core]

# This example owns its temporary directory. Application snapshots otherwise
# remain until explicitly removed; tg.load does not restore an autograd graph.
