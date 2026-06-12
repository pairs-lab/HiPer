"""
tests/conftest.py
=================
Pytest configuration for running tests WITHOUT catkin install.

The catkin setup.py declares package_dir={"seliSlam": "src"}, which
means Python normally only resolves the import after `catkin_make &&
source devel/setup.bash`. This conftest manually registers the same
mapping so all tests run standalone (no ROS master required).
"""

import importlib.util
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]   # .../seliSlam/
_SRC_DIR  = _PKG_ROOT / "src"


def _register_seliSlam_package():
    """
    Register src/ as the 'seliSlam' top-level package so that
    `from seliSlam.vio.replay_buffer import ...` resolves correctly.
    """
    if "seliSlam" in sys.modules:
        return

    init_path = _SRC_DIR / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "seliSlam",
        str(init_path),
        submodule_search_locations=[str(_SRC_DIR)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["seliSlam"] = mod
    spec.loader.exec_module(mod)

    # Also register each sub-package so nested imports work
    for subpkg in _SRC_DIR.iterdir():
        if subpkg.is_dir() and (subpkg / "__init__.py").exists():
            name = f"seliSlam.{subpkg.name}"
            if name not in sys.modules:
                sub_spec = importlib.util.spec_from_file_location(
                    name,
                    str(subpkg / "__init__.py"),
                    submodule_search_locations=[str(subpkg)],
                )
                sub_mod = importlib.util.module_from_spec(sub_spec)
                sys.modules[name] = sub_mod
                sub_spec.loader.exec_module(sub_mod)


_register_seliSlam_package()


# ── Lightweight rospy stub for offline testing ────────────────────────────────
# This lets modules that `import rospy` at module level load without a ROS master.

import types as _types      # noqa: E402  (must follow package registration)
import logging as _logging  # noqa: E402

def _make_rospy_stub():
    """Return a minimal rospy stub module."""
    stub = _types.ModuleType("rospy")

    # Logging — forward to Python logging
    _log = _logging.getLogger("rospy_stub")
    stub.loginfo      = _log.info
    stub.logwarn      = _log.warning
    stub.logwarn_throttle  = lambda period, msg, *a, **kw: _log.warning(msg)
    stub.loginfo_throttle  = lambda period, msg, *a, **kw: _log.info(msg)
    stub.logdebug          = _log.debug
    stub.logdebug_throttle = lambda period, msg, *a, **kw: _log.debug(msg)
    stub.logerr            = _log.error

    # Time
    class _Time:
        def __init__(self, secs=0): self.secs = secs
        def to_sec(self): return float(self.secs)
        @staticmethod
        def now(): return _Time(0)
    stub.Time     = _Time
    stub.Duration = lambda s: s

    # Node / param / topic stubs (no-ops)
    stub.init_node      = lambda *a, **kw: None
    stub.get_param      = lambda key, default=None: default
    stub.Publisher      = lambda *a, **kw: _types.SimpleNamespace(
        publish=lambda *a, **kw: None,
        get_num_connections=lambda: 0)
    stub.Subscriber     = lambda *a, **kw: None
    stub.Service        = lambda *a, **kw: None
    stub.Timer          = lambda *a, **kw: None
    stub.on_shutdown    = lambda *a, **kw: None
    stub.spin           = lambda: None
    stub.is_shutdown    = lambda: True
    stub.ROSException   = Exception

    return stub

if "rospy" not in sys.modules:
    sys.modules["rospy"] = _make_rospy_stub()
