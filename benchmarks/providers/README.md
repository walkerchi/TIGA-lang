# Provider conformance

`conformance.py` discovers `tiga.codegen` entry points and verifies the
versioned provider ABI, target devices, input IR, output artifact and async
capability. It deliberately reports `PENDING_EXTERNAL_PLUGIN` when a vendor
SDK/plugin or device is absent; interface declarations never count as backend
support. Hardware CI adds `--require rocm-dcu`, `--require metal`, or
`--require ppu` before correctness, artifact, compile-latency, roofline and
matched-peer gates.
