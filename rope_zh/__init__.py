# -*- coding: utf-8 -*-
"""Rope-Bronze 汉化 / 注释插件（语言下拉框 + 立即生效 + 注释开关）。

功能
----
* 设置页顶部新增"语言"分组：
  - LanguageSelect（语言下拉框）：English / 中文 / English + 中文注释
  - TipsSwitch（中文注释开关）：关闭后悬停不再弹出中文注释
* **选择语言 / 切换注释开关后立即生效**，无需重启（同时写入
  rope_zh/config.json，下次启动保持）。
* 实现方式：窗口按原始英文构建，构建完成后建立"文本注册表"（记录每个
  可翻译控件的英文原文），apply_language() 依据当前语言逐项翻译或还原，
  因此可以在任意时刻切换而无需重建界面。
* 文案拆分：中文界面文案在 zh_text.py，中文注释文案在 zh_tips.py。

用法
----
在 Rope.py 的 `if __name__ == "__main__":` 里、app.run() 之前：

    import rope_zh
    rope_zh.install()

卸载：删除 rope_zh 文件夹，去掉 Rope.py 里两行即可。
"""

from __future__ import annotations

import json
import os
import sys
import traceback

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QWidget,
)

from . import zh_text
from . import zh_tips

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(_ROOT, "config.json")

DEFAULT_CONFIG = {"language": "zh", "show_tips": True}

LANGUAGE_MODES = ("English", "中文", "English + 中文注释")
_MODE_KEY = {"English": "en", "中文": "zh", "English + 中文注释": "en_zh"}
_KEY_MODE = {v: k for k, v in _MODE_KEY.items()}


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    except Exception:
        pass
    if cfg.get("language") not in _MODE_KEY.values():
        cfg["language"] = "zh"
    if not isinstance(cfg.get("show_tips"), bool):
        cfg["show_tips"] = True
    return cfg


def save_config(cfg: dict) -> None:
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        traceback.print_exc()


# ---------------------------------------------------------------------------
# 语言下拉框控件（替换原来的分段按钮组）
# ---------------------------------------------------------------------------


class LangCombo(QWidget):
    value_changed = Signal(str, object)  # (name, mode str)

    def __init__(self, param, *, parent: QWidget | None = None):
        super().__init__(parent)
        self._param = param
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self._label = QLabel(param.label + " ")
        self._label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self._label, stretch=38)
        self._combo = QComboBox()
        self._combo.setProperty("class", "langCombo")
        for mode in param.modes:
            self._combo.addItem(mode)
        self._combo.currentTextChanged.connect(self._on_changed)
        layout.addWidget(self._combo, stretch=62)

    def _on_changed(self, text: str) -> None:
        self.value_changed.emit(self._param.name, text)

    def get(self) -> str:
        return self._combo.currentText()

    def set(self, value: str, request_frame: bool = True) -> None:
        idx = self._combo.findText(value)
        if idx < 0:
            idx = 0
        blocked = self._combo.blockSignals(not request_frame)
        try:
            self._combo.setCurrentIndex(idx)
        finally:
            self._combo.blockSignals(blocked)
        if request_frame:
            self.value_changed.emit(self._param.name, value)

    def load_default(self) -> None:
        self.set(self._param.default, request_frame=False)


# ---------------------------------------------------------------------------
# 注册语言控件（英文基底构建；翻译交给 apply_language 实时处理）
# ---------------------------------------------------------------------------


def _register_language_controls() -> None:
    """把 LanguageSelect（下拉）/ TipsSwitch 加入参数体系，并在设置页
    顶部插入"语言"分组。标题与控件标签统一用英文基底（'Language'），
    具体翻译由 apply_language 按当前语言实时切换。"""
    import rope.qt.parameters as pm
    from rope.qt._default_data import DEFAULT_DATA

    cfg = load_config()
    mode_name = _KEY_MODE[cfg["language"]]
    show_tips = bool(cfg.get("show_tips", True))

    lang_param = pm.SelectParam("LanguageSelect", "Language")
    tips_param = pm.SwitchParam("TipsSwitch", "Chinese Annotations")
    pm.PARAMETERS.append(lang_param)
    pm.PARAMETERS.append(tips_param)
    pm.PARAMETER_BY_NAME["LanguageSelect"] = lang_param
    pm.PARAMETER_BY_NAME["TipsSwitch"] = tips_param

    DEFAULT_DATA["LanguageSelectModes"] = list(LANGUAGE_MODES)
    DEFAULT_DATA["LanguageSelectMode"] = mode_name
    DEFAULT_DATA["TipsSwitchState"] = show_tips
    DEFAULT_DATA["LanguageSelectInfoText"] = zh_tips.EN_INFO["LanguageSelect"]
    DEFAULT_DATA["TipsSwitchInfoText"] = zh_tips.EN_INFO["TipsSwitch"]

    # LanguageSelect 使用自定义下拉控件
    import rope.qt.panes.parameters_pane as pane_mod

    _orig_widget_for = pane_mod._widget_for

    def _widget_for(param):
        if getattr(param, "name", None) == "LanguageSelect":
            return LangCombo(param)
        return _orig_widget_for(param)

    pane_mod._widget_for = _widget_for

    # 设置页顶部插入"语言"分组
    pane_mod.SETTINGS_SECTIONS.insert(0, ("Language", ["LanguageSelect", "TipsSwitch"]))


# ---------------------------------------------------------------------------
# 文本注册表：记录构建时的英文原文，实现任意时刻切换/还原
# ---------------------------------------------------------------------------

_ORIG_INFO: dict[str, str] = {}       # DEFAULT_DATA 里 InfoText 的英文原文
_BUTTON_TEXT_MAP: dict[str, str] = {} # 按钮英文可见文字 -> 中文
_REGISTRY = {                         # 捕获的可翻译控件（对象 -> 原文/标签）
    "tabs": [],       # list[(QTabWidget, [原文本...])]
    "buttons": [],    # list[(QAbstractButton, 原文)]
    "labels": [],     # list[(QLabel, 原文)]
    "param_labels": [],  # list[(name, QLabel, 原文)]
}
_CURRENT = {"lang": "zh", "show_tips": True}


def _build_button_text_map() -> None:
    import rope.qt.parameters as pm

    for p in pm.PARAMETERS:
        if p.kind == "button":
            text = p.text  # 构建时仍是英文（插件不再改写 Text）
            zh = zh_text.ZH_LABEL.get(p.name)
            if text and zh and text not in _BUTTON_TEXT_MAP:
                _BUTTON_TEXT_MAP[text] = zh


def _capture(window) -> None:
    """记录所有可翻译控件的英文原文（窗口刚构建完、尚未翻译）。"""
    from PySide6.QtWidgets import QAbstractButton, QLabel, QTabWidget

    _REGISTRY["tabs"].clear()
    _REGISTRY["buttons"].clear()
    _REGISTRY["labels"].clear()
    _REGISTRY["param_labels"].clear()

    pane = getattr(window, "_params_pane", None)

    tabs = getattr(pane, "_tabs", None) if pane is not None else None
    if tabs is not None:
        _REGISTRY["tabs"].append((tabs, [tabs.tabText(i) for i in range(tabs.count())]))

    seen = set()
    for btn in window.findChildren(QAbstractButton):
        t = btn.text()
        if t and id(btn) not in seen:
            seen.add(id(btn))
            _REGISTRY["buttons"].append((btn, t))

    if pane is not None:
        for name, w in list(pane.widgets.items()):
            label = getattr(w, "_label", None)
            if isinstance(label, QLabel):
                _REGISTRY["param_labels"].append((name, label, label.text()))
        for lab in pane.findChildren(QLabel):
            t = lab.text()
            if t and t.strip() in zh_text.ZH_UI and id(lab) not in seen:
                seen.add(id(lab))
                _REGISTRY["labels"].append((lab, t))


def _set_text(obj, text: str) -> None:
    try:
        if obj.text() != text:
            obj.setText(text)
    except Exception:
        traceback.print_exc()


def _apply_mode_tooltips(w, mode_map: dict) -> None:
    """给分段按钮（TextSelection）或下拉项（LangCombo）逐项挂注释。"""
    from PySide6.QtCore import Qt
    from rope.qt.widgets.text_selection import TextSelection

    if isinstance(w, TextSelection):
        for mode, m_info in mode_map.items():
            b = w._buttons.get(mode)
            if b is not None:
                b.setToolTip(m_info)
    elif hasattr(w, "_combo"):
        combo = w._combo
        for idx in range(combo.count()):
            m = combo.itemText(idx)
            if m in mode_map:
                combo.setItemData(idx, mode_map[m], Qt.ToolTipRole)


def _clear_mode_tooltips(w) -> None:
    from PySide6.QtCore import Qt
    from rope.qt.widgets.text_selection import TextSelection

    if isinstance(w, TextSelection):
        for b in getattr(w, "_buttons", {}).values():
            b.setToolTip("")
    elif hasattr(w, "_combo"):
        combo = w._combo
        for idx in range(combo.count()):
            combo.setItemData(idx, "", Qt.ToolTipRole)


def apply_language(lang: str, show_tips: bool) -> None:
    """依据当前语言实时翻译/还原全部界面文字与悬停注释。"""
    from PySide6.QtWidgets import QAbstractButton, QLabel
    from rope.qt._default_data import DEFAULT_DATA
    from rope.qt.widgets.text_selection import TextSelection

    _CURRENT["lang"] = lang
    _CURRENT["show_tips"] = show_tips
    is_zh = lang == "zh"

    # 1) 标签页
    for tabs, orig in _REGISTRY["tabs"]:
        for i, t in enumerate(orig):
            txt = zh_text.ZH_UI.get(t, t) if is_zh else t
            try:
                if tabs.tabText(i) != txt:
                    tabs.setTabText(i, txt)
            except Exception:
                traceback.print_exc()

    # 2) 按钮（含分组标题）
    for btn, orig in _REGISTRY["buttons"]:
        zh = _BUTTON_TEXT_MAP.get(orig) or zh_text.ZH_UI.get(orig)
        _set_text(btn, zh if (is_zh and zh) else orig)

    # 3) 参数控件标题（滑块/开关/下拉/切换控件）
    for name, label, orig in _REGISTRY["param_labels"]:
        suffix = orig[len(orig.rstrip()):]
        zh = zh_text.ZH_LABEL.get(name)
        _set_text(label, (zh + suffix) if (is_zh and zh) else orig)

    # 4) 其他界面 QLabel（目录/文件行等）
    for lab, orig in _REGISTRY["labels"]:
        suffix = orig[len(orig.rstrip()):]
        zh = zh_text.ZH_UI.get(orig.strip())
        _set_text(lab, (zh + suffix) if (is_zh and zh) else orig)

    # 5) DEFAULT_DATA InfoText（应用自身悬停信息区也随语言切换）
    if lang != "en" and show_tips:
        for name, zh in zh_tips.ZH_INFO.items():
            DEFAULT_DATA[name + "InfoText"] = zh
    else:
        for key, orig in _ORIG_INFO.items():
            DEFAULT_DATA[key] = orig
    ctrl_info = (zh_tips.EN_INFO if lang == "en" else zh_tips.ZH_INFO)
    DEFAULT_DATA["LanguageSelectInfoText"] = ctrl_info["LanguageSelect"]
    DEFAULT_DATA["TipsSwitchInfoText"] = ctrl_info["TipsSwitch"]

    # 6) 悬停注释（tooltip）
    pane = _CURRENT.get("pane")
    if pane is not None:
        for name, w in list(pane.widgets.items()):
            if show_tips:
                info = (zh_tips.EN_INFO if lang == "en" else zh_tips.ZH_INFO).get(name)
                try:
                    w.setToolTip(info or "")
                except Exception:
                    traceback.print_exc()
                if lang != "en":
                    mode_map = zh_tips.ZH_MODE_INFO.get(name)
                    if mode_map:
                        _apply_mode_tooltips(w, mode_map)
            else:
                try:
                    w.setToolTip("")
                except Exception:
                    traceback.print_exc()
                _clear_mode_tooltips(w)


# ---------------------------------------------------------------------------
# 窗口构建完成后的收尾
# ---------------------------------------------------------------------------


def _post_window(window) -> None:
    from PySide6.QtWidgets import QAbstractButton

    _capture(window)

    pane = getattr(window, "_params_pane", None)
    if pane is not None:
        _CURRENT["pane"] = pane

        # 预加载按钮状态文字：包一层，状态变化后按当前语言翻译
        try:
            cp = getattr(window, "_center_pane", None)
            if cp is not None:
                _orig_preload = cp.set_preload_button_state

                def _zh_preload(state):
                    _orig_preload(state)
                    b = cp.buttons.get("PreloadModelsButton")
                    if b is not None:
                        zh = zh_text.ZH_UI.get(b.text())
                        if _CURRENT["lang"] == "zh" and zh:
                            b.setText(zh)

                cp.set_preload_button_state = _zh_preload
        except Exception:
            traceback.print_exc()

        # 语言下拉：切换后立即生效
        combo = pane.widgets.get("LanguageSelect")
        if combo is not None:
            def _on_lang(name, value):
                cfg = load_config()
                cfg["language"] = _MODE_KEY.get(value, cfg["language"])
                save_config(cfg)
                apply_language(cfg["language"], cfg["show_tips"])
            combo.value_changed.connect(_on_lang)

        # 注释开关：切换后立即生效
        tips = pane.widgets.get("TipsSwitch")
        if tips is not None:
            def _on_tips(name, value):
                cfg = load_config()
                cfg["show_tips"] = bool(value)
                save_config(cfg)
                apply_language(cfg["language"], cfg["show_tips"])
            tips.value_changed.connect(_on_tips)

    # 按当前配置应用一次（启动即翻译）
    cfg = load_config()
    apply_language(cfg["language"], bool(cfg.get("show_tips", True)))


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def install() -> None:
    if getattr(install, "_done", False):
        return

    from rope.qt._default_data import DEFAULT_DATA

    # 保存 InfoText 英文原文，供还原用
    for key, val in DEFAULT_DATA.items():
        if key.endswith("InfoText") and isinstance(val, str):
            _ORIG_INFO[key] = val

    _build_button_text_map()
    _register_language_controls()

    import rope.qt.main_window as mw

    _orig_init = mw.MainWindow.__init__

    def _patched_init(self, *a, **k):
        _orig_init(self, *a, **k)
        try:
            _post_window(self)
        except Exception:
            print("[rope_zh] post-window pass failed:", file=sys.stderr)
            traceback.print_exc()

    mw.MainWindow.__init__ = _patched_init
    install._done = True
