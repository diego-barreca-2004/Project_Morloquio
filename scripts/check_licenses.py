"""Controllo licenze dell'ambiente Python: esce con 1 se una dipendenza non è open source ammessa."""

from __future__ import annotations

import sys

from morloquio.common.paths import repo_root
from morloquio.sources.licenses import audit_environment


def main() -> int:
    overrides = repo_root() / "configs" / "license_overrides.yaml"
    violations = audit_environment(overrides)
    for v in violations:
        print(f"VIOLAZIONE {v.package}: {v.reason} [dichiarata: {v.raw_license or '-'}]")
    if violations:
        print(f"{len(violations)} dipendenze non ammesse. Verifica a mano e documenta in {overrides}.")
        return 1
    print("license gate: tutte le dipendenze sono open source ammesse")
    return 0


if __name__ == "__main__":
    sys.exit(main())
