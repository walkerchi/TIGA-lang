from __future__ import annotations

import unittest

from graphforge.codegen import (
    ProviderCapabilities, get_provider, provider_conformance,
    register_provider,
)


class FakeVendor:
    capabilities = ProviderCapabilities(
        name="fake-dcu", target_family="rocm:dcu",
        devices=("hip", "dcu"), input_ir="ttir", artifacts=("hsaco",),
        supports_async=True, supports_distributed=True,
    )

    def compile(self, module, *, target, options):
        return {"module": module, "target": target, "options": dict(options)}


class ProviderRegistryTest(unittest.TestCase):
    def test_vendor_plugin_abi_and_conformance(self):
        register_provider("test-fake-dcu", FakeVendor(), replace=True)
        provider = get_provider("test-fake-dcu")
        report = provider_conformance(provider)
        self.assertEqual(report["input_ir"], "ttir")
        self.assertEqual(report["artifacts"], ("hsaco",))
        self.assertTrue(report["supports_distributed"])
        self.assertEqual(
            provider.compile("module {}", target="gfx", options={"waves": 4})[
                "target"
            ],
            "gfx",
        )

    def test_missing_provider_fails_with_inventory(self):
        with self.assertRaisesRegex(LookupError, "unavailable"):
            get_provider("does-not-exist")


if __name__ == "__main__":
    unittest.main()
