"""License gate: il progetto usa solo software open source e dati con licenza aperta.

- Dati: allowlist di licenze aperte; licenza ambigua o assente = fonte rifiutata.
- Codice: allowlist di licenze OSI compatibili con Apache-2.0; GPL rifiutata salvo override documentato.
- Strato di sistema (driver NVIDIA, CUDA): pacchetti ``nvidia-*`` esclusi dal controllo per scelta esplicita.
"""

from __future__ import annotations

import importlib.metadata
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

DATA_ALLOWED = {
    "CC0-1.0",
    "CC-BY-2.5",
    "CC-BY-3.0",
    "CC-BY-4.0",
    "CC-BY-SA-3.0",
    "CC-BY-SA-4.0",
    "ODbL-1.0",
    "ODC-By-1.0",
    "PDDL-1.0",
    "IODL-2.0",
    "etalab-2.0",
    "OGL-UK-3.0",
    "public-domain",
    # Licenze Copernicus (Sentinel, Copernicus DEM): uso libero, riproduzione, modifica e ridistribuzione con
    # attribuzione. Verifica e citazioni in docs/decisions/0003-licenze-copernicus.md
    "Copernicus-open",
}

CODE_ALLOWED = {
    "Apache-2.0",
    "MIT",
    "MIT-CMU",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "PSF-2.0",
    "MPL-2.0",
    "LGPL-2.1-or-later",
    "LGPL-3.0-or-later",
    "HPND",
    "Unlicense",
    "Zlib",
    "0BSD",
    "CC0-1.0",
    "public-domain",
}

# Licenze riconosciute ma non ammesse di default (servono per dare un motivo chiaro)
_KNOWN_REJECTED = {"GPL-2.0-only", "GPL-2.0-or-later", "GPL-3.0-only", "GPL-3.0-or-later", "AGPL-3.0-only"}

# Frasi comuni nei metadati Python, già minuscole e senza punteggiatura (vedi _key)
_ALIASES = {
    "mit": "MIT",
    "mit license": "MIT",
    "the mit license": "MIT",
    "expat": "MIT",
    "apache": "Apache-2.0",
    "apache 2": "Apache-2.0",
    "apache 20": "Apache-2.0",
    "apache software license": "Apache-2.0",
    "apache license 20": "Apache-2.0",
    "apache license version 20": "Apache-2.0",
    "apache software license 20": "Apache-2.0",
    "bsd": "BSD-3-Clause",
    "bsd license": "BSD-3-Clause",
    "bsd 3clause": "BSD-3-Clause",
    "bsd 3 clause": "BSD-3-Clause",
    "3clause bsd": "BSD-3-Clause",
    "new bsd license": "BSD-3-Clause",
    "modified bsd license": "BSD-3-Clause",
    "bsd 2clause": "BSD-2-Clause",
    "simplified bsd": "BSD-2-Clause",
    "isc": "ISC",
    "isc license": "ISC",
    "isc license iscl": "ISC",
    "python software foundation license": "PSF-2.0",
    "psf": "PSF-2.0",
    "mozilla public license 20 mpl 20": "MPL-2.0",
    "mpl 20": "MPL-2.0",
    "historical permission notice and disclaimer hpnd": "HPND",
    "the unlicense unlicense": "Unlicense",
    "zlib": "Zlib",
    "zliblibpng license": "Zlib",
    "gnu lesser general public license v2 or later lgplv2": "LGPL-2.1-or-later",
    "gnu lesser general public license v3 or later lgplv3": "LGPL-3.0-or-later",
    "gnu general public license v2 gplv2": "GPL-2.0-only",
    "gnu general public license v3 gplv3": "GPL-3.0-only",
    "gnu general public license v3 or later gplv3": "GPL-3.0-or-later",
    "public domain": "public-domain",
    "cc0 10 universal cc0 10 public domain dedication": "CC0-1.0",
}

_ALL_IDS = DATA_ALLOWED | CODE_ALLOWED | _KNOWN_REJECTED
_IDS_BY_KEY = {i.lower(): i for i in _ALL_IDS}
_EXPR_TOKEN = re.compile(r"\(|\)|[^\s()]+")


class LicenseRejected(ValueError):
    """Licenza non ammessa dal progetto."""


@dataclass(frozen=True)
class Violation:
    package: str
    raw_license: str
    reason: str


def _key(raw: str) -> str:
    s = raw.strip().lower().replace("-", "").replace("+", "")
    s = re.sub(r"[^a-z0-9/ ]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_license(raw: str) -> str | None:
    """Identificatore SPDX (o ``public-domain``) per una stringa di licenza, ``None`` se non riconosciuta."""
    if not raw or not raw.strip():
        return None
    direct = _IDS_BY_KEY.get(raw.strip().lower())
    if direct:
        return direct
    return _ALIASES.get(_key(raw))


def check_data_license(spdx: str) -> None:
    """Solleva :class:`LicenseRejected` se la licenza di una fonte dati non è aperta."""
    norm = normalize_license(spdx or "")
    if norm not in DATA_ALLOWED:
        raise LicenseRejected(f"licenza dati non ammessa: {spdx!r}")


def _eval_expression(tokens: list[str]) -> bool:
    """Valuta un'espressione SPDX: OR = almeno un ramo ammesso, AND = tutti ammessi, WITH = conta la base."""
    pos = 0

    def parse_or() -> bool:
        nonlocal pos
        result = parse_and()
        while pos < len(tokens) and tokens[pos].upper() == "OR":
            pos += 1
            rhs = parse_and()
            result = result or rhs
        return result

    def parse_and() -> bool:
        nonlocal pos
        result = parse_atom()
        while pos < len(tokens) and tokens[pos].upper() == "AND":
            pos += 1
            rhs = parse_atom()
            result = result and rhs
        return result

    def parse_atom() -> bool:
        nonlocal pos
        if pos >= len(tokens):
            raise LicenseRejected("espressione di licenza incompleta")
        tok = tokens[pos]
        pos += 1
        if tok == "(":
            value = parse_or()
            if pos >= len(tokens) or tokens[pos] != ")":
                raise LicenseRejected("parentesi non chiusa nell'espressione di licenza")
            pos += 1
            return value
        if pos < len(tokens) and tokens[pos].upper() == "WITH":
            pos += 2  # l'eccezione (es. LLVM-exception) non restringe la licenza base
        return normalize_license(tok) in CODE_ALLOWED

    ok = parse_or()
    if pos != len(tokens):
        raise LicenseRejected("espressione di licenza non valida")
    return ok


def check_code_license(raw: str) -> str:
    """Ritorna la licenza se ammessa per il codice, altrimenti solleva :class:`LicenseRejected`."""
    text = (raw or "").strip()
    tokens = _EXPR_TOKEN.findall(text)
    # Le parentesi da sole non bastano: "Mozilla Public License 2.0 (MPL 2.0)" è una frase, non un'espressione
    is_expression = any(t.upper() in {"AND", "OR", "WITH"} for t in tokens)
    if is_expression:
        if _eval_expression(tokens):
            return text
        raise LicenseRejected(f"espressione di licenza non ammessa: {text!r}")
    norm = normalize_license(text)
    if norm in CODE_ALLOWED:
        return text
    if norm in _KNOWN_REJECTED:
        raise LicenseRejected(f"{norm} non ammessa senza override documentato")
    raise LicenseRejected(f"licenza non riconosciuta o non ammessa: {text!r}")


def _distributions() -> Iterable:
    return importlib.metadata.distributions()


def _declared_licenses(meta) -> list[str]:
    """Licenze dichiarate, in ordine di affidabilità: License-Expression, classifier, campo License.

    Più classifier di licenza valgono come alternative (basta che una sia ammessa).
    """
    expr = (meta.get("License-Expression") or "").strip()
    if expr:
        return [expr]
    classifiers = [
        c.split("::")[-1].strip()
        for c in meta.get_all("Classifier") or []
        if c.startswith("License ::") and c.split("::")[-1].strip() != "OSI Approved"
    ]
    if classifiers:
        return classifiers
    first = ((meta.get("License") or "").strip().splitlines() or [""])[0].strip()
    if first and first.upper() != "UNKNOWN" and len(first) < 120:
        return [first]
    return []


def audit_environment(overrides_path: Path) -> list[Violation]:
    """Controlla le licenze di tutte le distribuzioni installate nell'ambiente corrente."""
    overrides = yaml.safe_load(Path(overrides_path).read_text(encoding="utf-8")) or {}
    prefixes = tuple(overrides.get("system_layer_prefixes", []))
    pkg_overrides = {k.lower(): v for k, v in (overrides.get("packages") or {}).items()}
    seen: set[str] = set()
    violations: list[Violation] = []
    for dist in _distributions():
        name = (dist.metadata.get("Name") or "").strip()
        key = name.lower()
        if not name or key in seen or key == "morloquio":
            continue
        seen.add(key)
        if key.startswith(prefixes):
            continue
        if key in pkg_overrides:
            continue  # verificata a mano: licenza e fonte documentate nel file di override
        declared = _declared_licenses(dist.metadata)
        if not declared:
            violations.append(Violation(name, "", "licenza assente nei metadati"))
            continue
        errors = []
        for candidate in declared:
            try:
                check_code_license(candidate)
                break
            except LicenseRejected as exc:
                errors.append(str(exc))
        else:
            violations.append(Violation(name, " | ".join(declared), "; ".join(errors)))
    return sorted(violations, key=lambda v: v.package.lower())
