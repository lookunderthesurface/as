from __future__ import annotations

from ..cloud.base import CloudProvider


def ask_cloud_if_configured(provider: CloudProvider | None, reason: str) -> str | None:
    if provider is None:
        return None
    return provider.investigate(reason)

