"""
hiper.evaluation.paths
~~~~~~~~~~~~~~~~~~~~~~
Data locations shared by every script. Set ``HIPER_DATA`` to move them;
the default is ``~/lifelong_ws/data``:

  $HIPER_DATA/hiper_thor/{raw,sessions,results,logs,object_types.json}
  $HIPER_DATA/obchange/{extracted,sessions}
"""

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get(
    "HIPER_DATA", str(Path.home() / "lifelong_ws" / "data"))).expanduser()
THOR_ROOT = DATA_ROOT / "hiper_thor"
THOR_SESSIONS = THOR_ROOT / "sessions"
THOR_TYPES_JSON = THOR_ROOT / "object_types.json"
RESULTS_ROOT = THOR_ROOT / "results"
OBCHANGE_ROOT = DATA_ROOT / "obchange"
OBCHANGE_EXTRACTED = OBCHANGE_ROOT / "extracted"
OBCHANGE_SESSIONS = OBCHANGE_ROOT / "sessions"
