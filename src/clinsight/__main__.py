"""让 `python -m clinsight` 可用（需要 src 在 sys.path 上，或已 pip install -e .）。"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
