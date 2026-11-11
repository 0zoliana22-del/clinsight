#!/usr/bin/env python
"""零依赖启动器：不用 pip install 也能直接跑。

    python run.py list
    python run.py analyze INT-001
    python run.py demo
    python run.py eval
    python run.py web
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from clinsight.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
