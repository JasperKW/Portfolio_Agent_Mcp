import sys
from pathlib import Path

root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "src" / "tools"))
sys.path.insert(0, str(root / "src" / "rag"))