"""Put the repository root on sys.path so `optlab` and `views` import in tests."""
import sys
from pathlib import Path

ROOT = str(Path(__file__).parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
