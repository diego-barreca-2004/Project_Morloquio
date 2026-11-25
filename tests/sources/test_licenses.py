from email.message import Message

import pytest

from morloquio.sources import licenses
from morloquio.sources.licenses import (
    LicenseRejected,
    audit_environment,
    check_code_license,
    check_data_license,
    normalize_license,
)


@pytest.mark.parametrize("ok", ["CC-BY-4.0", "ODbL-1.0", "CC0-1.0", "IODL-2.0", "etalab-2.0", "CC-BY-SA-4.0"])
def test_data_allowed(ok):
    check_data_license(ok)


@pytest.mark.parametrize("bad", ["CC-BY-NC-4.0", "CC-BY-NC-ND-3.0", "unverified", "proprietary", ""])
def test_data_rejected(bad):
    with pytest.raises(LicenseRejected):
        check_data_license(bad)


def test_normalize_common_strings():
    assert normalize_license("MIT License") == "MIT"
    assert normalize_license("Apache Software License") == "Apache-2.0"
    assert normalize_license("Apache License, Version 2.0") == "Apache-2.0"
    assert normalize_license("BSD 3-Clause") == "BSD-3-Clause"
    assert normalize_license("mit") == "MIT"
    assert normalize_license("something weird") is None


def test_spdx_expressions():
    assert check_code_license("MIT OR Apache-2.0") == "MIT OR Apache-2.0"
    assert (
        check_code_license("(Apache-2.0 OR BSD-3-Clause) AND MIT") == "(Apache-2.0 OR BSD-3-Clause) AND MIT"
    )
    with pytest.raises(LicenseRejected):
        check_code_license("MIT AND LicenseRef-Proprietary")


def test_classifier_phrase_with_parentheses_is_not_expression():
    assert (
        check_code_license("Mozilla Public License 2.0 (MPL 2.0)") == "Mozilla Public License 2.0 (MPL 2.0)"
    )


def test_gpl_rejected_without_override():
    with pytest.raises(LicenseRejected):
        check_code_license("GPL-3.0-only")


class _FakeDist:
    def __init__(self, name, **fields):
        self.metadata = Message()
        self.metadata["Name"] = name
        for key, value in fields.items():
            if isinstance(value, list):
                for v in value:
                    self.metadata[key] = v
            else:
                self.metadata[key] = value


@pytest.fixture
def overrides(tmp_path):
    p = tmp_path / "overrides.yaml"
    p.write_text(
        "system_layer_prefixes: ['nvidia-']\n"
        "packages:\n"
        "  oddpkg: {license: BSD-3-Clause, source: 'https://example.org/LICENSE', note: verificata}\n"
    )
    return p


def test_audit_skips_system_layer(monkeypatch, overrides):
    dists = [_FakeDist("nvidia-cublas-cu12", License="Other/Proprietary License")]
    monkeypatch.setattr(licenses, "_distributions", lambda: dists)
    assert audit_environment(overrides) == []


def test_audit_reads_expression_classifier_and_override(monkeypatch, overrides):
    dists = [
        _FakeDist("a", **{"License-Expression": "MIT"}),
        _FakeDist("b", Classifier=["License :: OSI Approved :: BSD License"]),
        _FakeDist("oddpkg", License="UNKNOWN"),
        _FakeDist("evil", License="Proprietary"),
    ]
    monkeypatch.setattr(licenses, "_distributions", lambda: dists)
    violations = audit_environment(overrides)
    assert [v.package for v in violations] == ["evil"]
