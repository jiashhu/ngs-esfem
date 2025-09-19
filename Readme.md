

# 假设有一个模块 mymodule.py

import mymodule
import importlib

# 修改了 mymodule.py 的内容后，想重新加载
importlib.reload(mymodule)

# 可视化
是使用 from ngsolve.webgui import Draw 而不是下面的 from netgen.webgui import Draw