"""
     Identity resolution for physical/virtual assets: maps a raw identity
     (hostname, GPU UUID, PDU outlet, etc.) reported by one telemetry system to
     a single canonical asset ID, so power and compute readings for the same
     physical asset can be joined even though each source names it differently.

     Identities are a generic {identity_type: value} mapping rather than fixed
     fields — which identity types actually exist depends on which telemetry
     sources get wired up (Step 5.5), and shouldn't be hardcoded here.

     "In-memory first" by design, same as InMemoryVectorIndex: seeded with its
     mapping at construction. A real backing store (CMDB, DCIM export) can
     replace the seed source later behind the same resolve() shape.
"""
from __future__ import annotations


class AssetRegistry:
    def __init__(self, assets: dict[str, dict[str, str]]) -> None:
        """
           assets maps asset_id -> {identity_type: value, ...}, e.g.
           {"FBA-ASSET-928374": {"hostname": "gpu-node-027", "gpu_uuid": "GPU-a36..."}}.
        """
        self._asset_count = len(assets)
        self._by_identity: dict[tuple[str, str], str] = {
            (identity_type, value): asset_id
            for asset_id, identities in assets.items()
            for identity_type, value in identities.items()
        }

    @property
    def asset_count(self) -> int:
        return self._asset_count

    def resolve(self, identity_type: str, value: str) -> str | None:
        """Look up the canonical asset ID for one raw identity, or None if unknown."""
        return self._by_identity.get((identity_type, value))
