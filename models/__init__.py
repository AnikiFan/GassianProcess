import os
import sys
from .MyGP import GP
from .kernel import rbf

# 获取上一级目录的路径
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

# 导入 models 文件夹中的模块