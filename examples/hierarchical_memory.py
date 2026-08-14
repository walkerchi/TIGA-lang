"""Compiler-runtime storage instances: RAM -> NVMe -> RAM."""

import graphforge as gf


with gf.runtime.HierarchyRuntime(budgets={"ram": 1024, "nvme": 1024}) as runtime:
    source = runtime.allocate("state", 7, tier="ram", capacity_bytes=16)
    source.buffer.write(b"GraphForge-state")
    spill = runtime.allocate("state", 7, tier="nvme", capacity_bytes=16)
    runtime.transfer(source, spill).wait()
    source.close()
    restored = runtime.allocate("state", 7, tier="ram", capacity_bytes=16)
    runtime.transfer(spill, restored).wait()
    assert restored.buffer.read() == b"GraphForge-state"
    print(runtime.peak_live_bytes)
