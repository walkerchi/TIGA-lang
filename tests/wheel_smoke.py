"""Post-install smoke test for a Tiga binary wheel.

Run this with the Python interpreter from a clean environment containing only
the wheel.  It deliberately avoids Torch and the source-tree build directory.
"""

from __future__ import annotations

from pathlib import Path
import subprocess

import tiga as gf
from tiga.compiler.tensor_mlir import tensor_mlir


def main() -> None:
    value = gf.tensor([1.0, 2.0]) * 2.0 + 1.0
    module = tensor_mlir(value)
    assert '"gf_tensor.mul"' in module
    assert '"gf_tensor.add"' in module
    assert value.tolist() == [3.0, 5.0]

    package = Path(gf.__file__).resolve().parent
    for tool in ("gf-opt", "gf-translate"):
        executable = package / "bin" / tool
        assert executable.is_file(), f"wheel is missing {executable}"
        completed = subprocess.run(
            [str(executable), "--version"],
            check=True,
            text=True,
            capture_output=True,
        )
        assert completed.stdout.strip(), f"{tool} produced no version output"
    print(f"Tiga wheel smoke passed: {package}")


if __name__ == "__main__":
    main()
