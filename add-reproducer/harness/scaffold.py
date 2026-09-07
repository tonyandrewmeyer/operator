"""Scratch-charm scaffolding (PLAN.md Approach §4's k8s-scratch branch).

Thin wrapper around `spike-step-3/charm/render.py`, invoked exactly as its
own docstring documents (`render.py params/<name>.yaml out/<name>/`) --
deliberately *not* reimplemented here. render.py carries three real
charmcraft/juju bugfixes (Steps §3, landed 2026-07-22: `platforms:` shape,
missing `uv.lock`, charm-name shape); invoking the file directly rather
than duplicating its logic means this harness can't silently drift out of
sync with those fixes.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from models import SurfaceInference

HERE = Path(__file__).parent
SPIKE3_RENDER = HERE.parent / "spike-step-3" / "charm" / "render.py"


class ScaffoldError(RuntimeError):
    pass


def expected_signal_for(surface: SurfaceInference) -> str | None:
    """The status-message substring the *rendered charm* will actually emit
    when its surface fires.

    Read from `render.py`'s own `SURFACE_SIGNALS` table rather than asked
    of an LLM: the string is decided by the template, so a guess can only
    match by luck. The first successful k8s stimulus (2026-08-18) was
    scored against an invented "Received Pebble custom notice" while the
    charm emitted "observed notice: ..." -- the signal could not have been
    found however the bug behaved, and the run was recorded as
    `did_not_reproduce`.
    """
    signals = _render_module().SURFACE_SIGNALS
    return signals.get(surface.ops_api_surface)


def _render_module():
    """Import `spike-step-3/charm/render.py` as a module, so its constants
    are read from the one file that defines them."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_add_reproducer_render", SPIKE3_RENDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def surface_to_params(surface: SurfaceInference, source_issue: int | None) -> dict:
    """Build the params.yaml shape render.py's `load_params()` expects."""
    return {
        "charm_name": surface.charm_name,
        "source_issue": source_issue,
        "relation": surface.relation,
        "storage": surface.storage,
        "pebble_service": surface.pebble_service,
        "ops_api_surface": surface.ops_api_surface,
    }


def render_charm(surface: SurfaceInference, out_dir: Path, *, source_issue: int | None = None) -> Path:
    """Render a scratch charm from `surface` into `out_dir`. Raises
    `ScaffoldError` on any render.py failure (e.g. a rejected identifier --
    render.py treats every params.yaml value as untrusted LLM output, see
    its module docstring)."""
    if not SPIKE3_RENDER.exists():
        raise ScaffoldError(f"render.py not found at {SPIKE3_RENDER}")

    out_dir = Path(out_dir)
    params = surface_to_params(surface, source_issue)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(params, f)
        params_path = Path(f.name)
    try:
        proc = subprocess.run(
            [sys.executable, str(SPIKE3_RENDER), str(params_path), str(out_dir)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            raise ScaffoldError(f"render.py failed:\n{proc.stderr}")
    finally:
        params_path.unlink(missing_ok=True)
    return out_dir
