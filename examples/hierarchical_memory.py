"""Chainable tensor spill: .disk() writes, .cpu() reads — gf owns the files."""

import torch

import tiga as gf

# --8<-- [start:core]
values = gf.from_torch(torch.arange(8, dtype=torch.float32))  # (8,)
values.disk()  # write the payload to the spill store, free the buffer
back = values.cpu()  # read it back — any later use would reload lazily anyway

# Persist across processes: name the spill. Another process attaches it with
#   gf.from_disk("layer-3-activations")
values.disk(name="layer-3-activations")
# --8<-- [end:core]

# Anonymous spills are deleted when the tensor is collected or the process
# exits. Named spills live in TIGA_SPILL_DIR or ~/.cache/tiga/spill
# until you delete them.
