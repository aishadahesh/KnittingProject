"""Pytest bootstrap.

The application modules live in src/ and import each other by plain name, the
same way app.py arranges at startup. pytest loads this file before collecting
any tests, so putting src/ on the path here is what lets the test modules import
them without each one repeating the path juggling.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(PROJECT_ROOT, "src")

for path in (PROJECT_ROOT, SRC):
    if path not in sys.path:
        sys.path.insert(0, path)
