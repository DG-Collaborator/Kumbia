"""
     Identity resolution for physical/virtual assets: maps a raw identity
     (hostname, GPU UUID, PDU outlet, IP address, etc.) reported by one
     telemetry system to a single canonical asset ID, so readings for the same
     physical asset can be joined even though each source names it differently.

     Identities are a generic {identity_type: value} mapping rather than fixed
     fields — which identity types exist depends on which telemetry sources
     are wired up, and shouldn't be hardcoded here.

     Lookups go both ways: resolve() maps one raw identity to its asset (used
     by the correlation engine on every fetched row), and identities() lists
     every registered value of one type (used by query builders to filter a
     source to the assets it actually knows about). Every consumer works on
     that list, so one asset and a thousand run the same code.

     In-memory, seeded at construction, same as InMemoryVectorIndex. Where
     the seed comes from is the caller's choice: load_asset_registry() reads
     a JSON file today; a CMDB/DCIM-backed loader later only has to produce
     the same {asset_id: {identity_type: value}} dict.
"""
from __future__ import annotations

import json
from pathlib import Path


class AssetRegistry:
    def __init__(self, assets: dict[str, dict[str, str]]) -> None:
        """
           assets maps asset_id -> {identity_type: value, ...}, e.g.
           {"FBA-ASSET-928374": {"hostname": "gpu-node-027", "pdu_outlet": "R42-U18"}}.
           Two assets claiming the same (identity_type, value) is rejected:
           readings for that identity couldn't be attributed to either one.
        """
        self._assets: dict[str, dict[str, str]] = {
            asset_id: dict(identities) for asset_id, identities in assets.items()
        }
        self._by_identity: dict[tuple[str, str], str] = {}
        for asset_id, identities in sorted(self._assets.items()):
            for identity_type, value in identities.items():
                key = (identity_type, value)
                owner = self._by_identity.get(key)
                if owner is not None:
                    raise ValueError(
                        f"{identity_type}={value!r} is registered to both {owner!r} and {asset_id!r}."
                    )
                self._by_identity[key] = asset_id

    @property
    def asset_count(self) -> int:
        return len(self._assets)

    @property
    def asset_ids(self) -> list[str]:
        return sorted(self._assets)

    def resolve(self, identity_type: str, value: str) -> str | None:
        """Look up the canonical asset ID for one raw identity, or None if unknown."""
        return self._by_identity.get((identity_type, value))

    def identities(self, identity_type: str) -> list[str]:
        """
           Every registered value of one identity type, in asset-ID order —
           e.g. identities("pdu_outlet") is every outlet a power query
           should read. Assets without that identity are skipped.
        """
        return [
            identities[identity_type]
            for _, identities in sorted(self._assets.items())
            if identity_type in identities
        ]


def load_asset_registry(path: Path) -> AssetRegistry:
    """
       Reads {asset_id: {identity_type: value}} from a JSON file. Every key
       and value must be a non-empty string; anything else fails loudly at
       startup rather than silently resolving nothing at query time.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{path}: expected a non-empty object of asset_id -> identities.")
    for asset_id, identities in data.items():
        if not isinstance(identities, dict) or not identities:
            raise ValueError(f"{path}: asset {asset_id!r} needs a non-empty object of identities.")
        for identity_type, value in identities.items():
            if not (isinstance(identity_type, str) and identity_type and isinstance(value, str) and value):
                raise ValueError(f"{path}: asset {asset_id!r} has a non-string or empty identity: {identity_type!r}.")
    return AssetRegistry(data)
