import sys
import io

# 避免打包后 sys.stdout / stderr 为 None 报错
if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()

from gui.app import boot

if __name__ == "__main__":
    boot()
