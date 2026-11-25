"""Hash dei parametri: identifica in modo stabile la configurazione di uno stadio."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping


def params_hash(params: Mapping) -> str:
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:8]
