"""Validate that every registered measured operation has complete artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def is_formal_operation(record: dict) -> bool:
    """Return whether *record* publishes case directories as formal evidence.

    Status is descriptive metadata, not a registration mechanism.  In
    particular, promoting ``measured-*`` to ``performance-ready-*`` must not
    silently unregister already-published artifacts.  A non-empty ``cases``
    list is the manifest's explicit declaration that those cases are formal.
    """

    return bool(record.get("cases"))


def semantic_x_violations(payload: dict, label: str = "<payload>") -> list[str]:
    groups: dict[tuple[object, ...], set[float]] = {}
    for item in payload.get("results", ()):
        key = (
            item.get("topology"), item.get("nodes"),
            item.get("edges"), item.get("features"),
        )
        intensity = round(float(
            item["arithmetic_intensity_flop_per_byte"]), 12)
        groups.setdefault(key, set()).add(intensity)
    return [
        f"{label}: workload {key} has semantic x values {sorted(intensities)}"
        for key, intensities in groups.items() if len(intensities) != 1
    ]


def main() -> None:
    root = Path("output/roofline")
    manifest = json.loads((root / "MANIFEST.json").read_text())
    required = tuple(manifest["required_case_artifacts"])
    missing = []
    inconsistent_x = []
    registered_json = set()
    for operation, record in manifest["operations"].items():
        if not is_formal_operation(record):
            continue
        for case in record["cases"]:
            for filename in required:
                path = root / operation / case / filename
                if not path.is_file():
                    missing.append(str(path))
            json_path = root / operation / case / "roofline.json"
            if json_path.is_file():
                registered_json.add(json_path.resolve())
                payload = json.loads(json_path.read_text())
                inconsistent_x.extend(
                    semantic_x_violations(payload, str(json_path)))
        for relative in record.get("auxiliary", ()):
            path = root / operation / relative
            if not path.is_file():
                missing.append(str(path))
    unregistered = sorted(
        str(path) for path in root.glob("*/*/roofline.json")
        if path.resolve() not in registered_json
    )
    if missing or inconsistent_x or unregistered:
        sections = []
        if missing:
            sections.append("missing roofline artifacts:\n" + "\n".join(missing))
        if inconsistent_x:
            sections.append(
                "providers of the same operation must share semantic x:\n"
                + "\n".join(inconsistent_x))
        if unregistered:
            sections.append(
                "formal roofline artifacts missing from MANIFEST.json:\n"
                + "\n".join(unregistered))
        raise SystemExit("\n".join(sections))
    formal = sum(
        is_formal_operation(record)
        for record in manifest["operations"].values())
    print(f"roofline artifact contract passed for {formal} formal operations")


if __name__ == "__main__":
    main()
