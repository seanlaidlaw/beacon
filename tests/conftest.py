"""
Ensure the project root is on sys.path so `import beacon` works whether
beacon is installed via `pip install -e .` (CI) or run directly from the
source tree (local development without installation).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
