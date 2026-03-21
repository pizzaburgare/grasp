#!/usr/bin/env python3
"""Pre-commit hook: fail if any file exceeds MAX_LINES lines."""

import sys

MAX_LINES = 600

failed = False
for path in sys.argv[1:]:
    with open(path) as f:
        lines = sum(1 for _ in f)
    if lines > MAX_LINES:
        print(f"  {path}: {lines} lines (max {MAX_LINES})")
        failed = True

if failed:
    print(f"\nFiles exceed the {MAX_LINES}-line limit. Split them into smaller modules.")
    sys.exit(1)
