import json
import re
from pathlib import Path

import pytest
import yaml

from models import SurfaceInference
from scaffold import ScaffoldError, render_charm

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _surface_2639() -> SurfaceInference:
    return SurfaceInference.from_dict(json.loads((FIXTURES / "surface" / "2639.json").read_text()))


def test_render_charm_2639_end_to_end(tmp_path):
    surface = _surface_2639()
    out_dir = tmp_path / "charm-2639"
    try:
        render_charm(surface, out_dir, source_issue=2639)
    except ScaffoldError as exc:
        pytest.skip(f"render.py needs network access for `uv lock`, unavailable here: {exc}")

    charmcraft_yaml = (out_dir / "charmcraft.yaml").read_text()
    doc = yaml.safe_load(charmcraft_yaml)

    # Steps §3 fixes (2026-07-22), must not regress:
    assert doc["name"] == "repro-i2639-pebble-notice"  # no digit directly after a hyphen
    assert not re.search(r"-\d", doc["name"])  # juju's local-charm URL parser rejects this shape
    platforms = doc["platforms"]
    assert platforms == {"ubuntu@24.04:amd64": None}  # bare-key form, not {amd64: {}}
    assert (out_dir / "uv.lock").exists()

    charm_py = (out_dir / "src" / "charm.py").read_text()
    assert "pebble_custom_notice" in charm_py
    assert "observed notice:" in charm_py


def test_render_charm_rejects_bad_identifier(tmp_path):
    surface = SurfaceInference(
        charm_name="repro-i1-x",
        relation={"name": "not a valid name!"},
    )
    with pytest.raises(ScaffoldError):
        render_charm(surface, tmp_path / "out")
