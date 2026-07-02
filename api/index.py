# api/index.py
import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))

# 导入你的 main 应用
try:
    from app.main import app
except ImportError:
    # 如果上面的导入失败，尝试直接导入
    sys.path.insert(0, str(root_path / "app"))
    from main import app

# Vercel 需要这个变量名
handler = app