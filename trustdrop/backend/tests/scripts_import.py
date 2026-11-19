"""Lets tests reuse the synthetic generator from /scripts (test fixtures only)."""
import importlib.util
from pathlib import Path

_p = Path(__file__).resolve().parents[2] / "scripts" / "make_synthetic_dev_data.py"
_spec = importlib.util.spec_from_file_location("make_synth_mod", _p)
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)


def make_synth(n: int = 600):
    return _mod.make(n)
