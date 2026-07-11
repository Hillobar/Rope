"""Visual harness: instantiate one of every widget, render, screenshot.

Phase B validation: every widget loads, paints, and accepts get()/set() calls.

Run:  venv\\Scripts\\python.exe -m rope.qt.tests.widget_harness
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from rope.qt.parameters import (
    PARAMETER_BY_NAME,
    ButtonParam,
    EntryParam,
    SelectParam,
    SliderParam,
    SwitchParam,
)
from rope.qt.widgets.button import IconButton
from rope.qt.widgets.slider import ParameterSlider
from rope.qt.widgets.switch import Switch
from rope.qt.widgets.text import Text
from rope.qt.widgets.text_selection import TextSelection
from rope.qt.widgets.timeline import Timeline
from rope.qt.widgets.vram_indicator import VRAMIndicator


def _load_stylesheet(app: QApplication) -> None:
    qss = Path(__file__).resolve().parent.parent / "rope.qss"
    if qss.is_file():
        app.setStyleSheet(qss.read_text(encoding="utf-8"))


def _section(title: str, body: QWidget) -> QFrame:
    f = QFrame()
    f.setProperty("panelTier", "3")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(8, 6, 8, 6)
    lay.setSpacing(4)
    lay.addWidget(Text(text=title, tier=3))
    lay.addWidget(body)
    return f


def _wrap(widget: QWidget) -> QFrame:
    f = QFrame()
    f.setProperty("panelTier", "2")
    lay = QHBoxLayout(f)
    lay.setContentsMargins(6, 4, 6, 4)
    lay.addWidget(widget)
    return f


def build_harness() -> QWidget:
    root = QFrame()
    root.setProperty("panelTier", "1")
    layout = QVBoxLayout(root)
    layout.setContentsMargins(12, 12, 12, 12)
    layout.setSpacing(10)

    # Header
    header = Text(text="rope/qt widget harness — Phase B", tier=1)
    header.setStyleSheet("font-size: 14pt; color: #FFFFFF; padding: 4px;")
    layout.addWidget(header)

    # Buttons
    bp_start = PARAMETER_BY_NAME["StartRope"]
    bp_play = PARAMETER_BY_NAME["Play"]
    bp_save = PARAMETER_BY_NAME["SaveImageButton"]
    btn_row = QFrame(); btn_row.setProperty("panelTier", "2")
    btn_lay = QHBoxLayout(btn_row); btn_lay.setContentsMargins(6, 4, 6, 4); btn_lay.setSpacing(8)
    btn_start = IconButton(bp_start)
    btn_play = IconButton(bp_play)
    btn_save = IconButton(bp_save)
    btn_error = IconButton(PARAMETER_BY_NAME["LoadTVideos"]); btn_error.error_button()
    btn_lay.addWidget(btn_start)
    btn_lay.addWidget(btn_play)
    btn_lay.addWidget(btn_save)
    btn_lay.addWidget(btn_error)
    btn_lay.addStretch()
    layout.addWidget(_section("IconButton — text/both/icon modes + error state", btn_row))

    # Switch
    sw_param = PARAMETER_BY_NAME["RestorerSwitch"]
    sw = Switch(sw_param)
    sw.set(True, request_frame=False)
    sw_off = Switch(PARAMETER_BY_NAME["OrientSwitch"])
    sw_row = QFrame(); sw_row.setProperty("panelTier", "2")
    sw_lay = QVBoxLayout(sw_row); sw_lay.setContentsMargins(6, 4, 6, 4); sw_lay.setSpacing(4)
    sw_lay.addWidget(sw); sw_lay.addWidget(sw_off)
    layout.addWidget(_section("Switch — on / off", sw_row))

    # Sliders: integral + fractional
    slider_int = ParameterSlider(PARAMETER_BY_NAME["RestorerSlider"])
    slider_frac = ParameterSlider(PARAMETER_BY_NAME["ColorGammaSlider"])
    slider_neg = ParameterSlider(PARAMETER_BY_NAME["ColorRedSlider"])  # -100..100
    slider_int.set(75, request_frame=False)
    slider_frac.set(1.25, request_frame=False)
    slider_neg.set(-40, request_frame=False)
    sl_row = QFrame(); sl_row.setProperty("panelTier", "2")
    sl_lay = QVBoxLayout(sl_row); sl_lay.setContentsMargins(6, 4, 6, 4); sl_lay.setSpacing(4)
    sl_lay.addWidget(slider_int); sl_lay.addWidget(slider_frac); sl_lay.addWidget(slider_neg)
    layout.addWidget(_section("ParameterSlider — integral / fractional / negative range", sl_row))

    # TextSelection
    sel = TextSelection(PARAMETER_BY_NAME["DetectTypeTextSel"])
    sel_merge = TextSelection(PARAMETER_BY_NAME["MergeTextSel"])
    sel_merge.set("Sph", request_frame=False)
    se_row = QFrame(); se_row.setProperty("panelTier", "2")
    se_lay = QVBoxLayout(se_row); se_lay.setContentsMargins(6, 4, 6, 4); se_lay.setSpacing(4)
    se_lay.addWidget(sel); se_lay.addWidget(sel_merge)
    layout.addWidget(_section("TextSelection — exclusive group", se_row))

    # Timeline
    tl = Timeline()
    tl.set_length(500)
    tl.set(180)
    tl.set_markers([60, 180, 300, 420])
    layout.addWidget(_section("Timeline — 500 frames, 4 markers, playhead at 180", _wrap(tl)))

    # VRAM indicator: 9.2 / 24 (~38%) and 22 / 24 (>90%)
    v1 = VRAMIndicator(); v1.set(9.2, 24.0); v1.setFixedHeight(20)
    v2 = VRAMIndicator(); v2.set(22.5, 24.0); v2.setFixedHeight(20)
    vr_row = QFrame(); vr_row.setProperty("panelTier", "2")
    vr_lay = QVBoxLayout(vr_row); vr_lay.setContentsMargins(6, 4, 6, 4); vr_lay.setSpacing(4)
    vr_lay.addWidget(v1); vr_lay.addWidget(v2)
    layout.addWidget(_section("VRAMIndicator — normal / over-limit", vr_row))

    layout.addStretch()
    return root


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    _load_stylesheet(app)

    container = QScrollArea()
    container.setWidgetResizable(True)
    container.setWidget(build_harness())
    container.setWindowTitle("Rope-Pearl widget harness (Phase B)")
    container.resize(820, 980)
    container.show()

    shot = Path(__file__).resolve().parent / "phase_b_widgets.png"

    def _capture():
        container.widget().adjustSize()
        # grab the inner widget to avoid clipping by the scroll viewport
        pixmap = container.widget().grab()
        pixmap.save(str(shot), "PNG")
        print(f"[harness] screenshot: {shot}")
        print(f"[harness] inner size: {container.widget().size().width()}x{container.widget().size().height()}")
        app.quit()

    QTimer.singleShot(700, _capture)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
