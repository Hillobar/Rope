#!/usr/bin/env python3

import sys

if __name__ == "__main__":
    # 中文汉化插件（rope_zh）：菜单汉化 + 设置项 "?" 悬停中文说明。
    # 删除 rope_zh 文件夹即可整体卸载。
    try:
        import rope_zh
        rope_zh.install()
    except Exception as _e:
        print("[rope_zh] plugin load failed: %s" % _e)
    from rope.qt import app
    sys.exit(app.run())
