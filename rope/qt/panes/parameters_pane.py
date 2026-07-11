"""Right-pane parameter list — generated from rope.qt.parameters.PARAMETERS.

Replaces the manually-laid-out ~800-line parameters section in rope/GUI.py.
Sections mirror the original column layout; the widget type is chosen by
the parameter's `kind` field, eliminating the per-parameter constructor
churn that dominated the Tk port.

Each widget's `value_changed(name, value)` signal is connected to a single
slot that updates a {name: value} dict and emits a `params_changed` signal
back to the main window.
"""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import os

from rope.Models import MODEL_INVENTORY, MODEL_INVENTORY_DETAILS
from rope.qt.parameters import (
    PARAMETER_BY_NAME,
    Parameter,
    parameters_by_kind,
)
from rope.qt.widgets.button import IconButton
from rope.qt.widgets.slider import ParameterSlider
from rope.qt.widgets.switch import Switch
from rope.qt.widgets.text_entry import TextEntry
from rope.qt.widgets.text_selection import TextSelection


# Section layout mirroring the Tk parameters column ordering. Order
# matters — the original GUI's vertical layout is reproduced top-to-bottom.
SECTIONS: list[tuple[str, list[str]]] = [
    ("Similarity", ["ThresholdSlider"]),
    ("Swapper", [
        "MergeTextSel", "SwapperTypeTextSel",
    ]),
    ("Restorer", [
        "RestorerSwitch", "RestorerTypeTextSel",
        "RestorerDetTypeTextSel", "RestorerSlider",
    ]),
    ("Orientation", ["OrientSwitch", "OrientAutoSwitch", "OrientSlider"]),
    ("Likeness / Fidelity", [
        "LikenessSlider", "EmbExtrapSlider",
        "HighFidelitySwitch", "HighFidelityAlphaSlider",
        "HighFidelityModeTextSel",
        "HFRefineButton", "HFClearCacheButton",
        "StrengthSwitch", "StrengthSlider",
    ]),
    ("Masking", [
        "BorderTopSlider", "BorderSidesSlider",
        "BorderBottomSlider", "BorderBlurSlider",
        "DiffSwitch", "DiffSlider",
        "OccluderSwitch", "OccluderSlider",
        "DFLXSegSwitch", "DFLXSegSizeSlider", "DFLXSegBlurSlider",
        "FaceParserSwitch", "FaceParserSlider", "MouthParserSlider",
        "BlendSlider",
    ]),
    ("Color", [
        "ColorMatchSwitch",
        "ColorSwitch",
        "ColorRedSlider", "ColorGreenSlider", "ColorBlueSlider",
        "ColorGammaSlider", "ColorContrastSlider", "ColorSaturationSlider",
    ]),
    ("Face Adjustments", [
        "FaceAdjSwitch",
        "KPSXSlider", "KPSYSlider", "KPSScaleSlider", "FaceScaleSlider",
    ]),
]


# Settings tab sections — system-level controls split out of the
# Parameters tab so the swap-tuning workflow isn't cluttered with
# threading / detection / encoder knobs.
SETTINGS_SECTIONS: list[tuple[str, list[str]]] = [
    ("Threading", ["ThreadsSlider", "ModelSessionsTextSel"]),
    ("Detection", ["DetectTypeTextSel", "DetectInputSizeTextSel", "DetectScoreSlider"]),
    ("Recording", ["RecordTypeTextSel", "VideoQualSlider"]),
    # Live screen-capture knobs. CaptureFPSSlider is read each capture
    # tick by WindowCapture via the params-pane value mirror; the swap
    # worker count reuses ThreadsSlider above.
    ("Capture", ["CaptureFPSSlider"]),
]


def _widget_for(param: Parameter) -> QWidget:
    if param.kind == "slider":
        return ParameterSlider(param)
    if param.kind == "switch":
        return Switch(param)
    if param.kind == "select":
        return TextSelection(param)
    if param.kind == "entry":
        return TextEntry(param)
    if param.kind == "button":
        return IconButton(param)
    raise ValueError(f"unsupported kind {param.kind!r} for {param.name}")


class ParametersPane(QFrame):
    """Right-pane scroll area listing every parameter widget by section.

    Public API:
        widgets: dict[str, QWidget]      — name -> widget for set/get
        values:  dict[str, Any]          — live mirror of widget values
        params_changed: Signal(dict)     — fires on any change with full snapshot
        io_action: Signal(str)           — emits 'save' / 'load' / 'default'
    """

    params_changed = Signal(dict)
    io_action = Signal(str)
    # Emitted when an in-section button is clicked. Payload is the
    # button param's `name`. Host wires this to per-button handlers
    # (e.g. HF refine / clear cache).
    button_clicked = Signal(str)
    # Emitted whenever a collapsible section is toggled, so the host can
    # persist per-section collapsed state. `collapsed=True` means the
    # section was just hidden.
    section_toggled = Signal(str, bool)
    # Emitted when the user clicks Browse on a path-picker row in the
    # Settings tab. The host shows the directory / file chooser,
    # persists the result, and calls back into the matching setter to
    # refresh the displayed path.
    models_folder_pick_requested = Signal()
    output_folder_pick_requested = Signal()
    embeddings_file_pick_requested = Signal()
    # Settings tab → MainWindow: user clicked Unload on an inventory row.
    # Payload is the Models attribute name (e.g. "swapper_model"). The
    # host calls models.unload_model(attr); bus.vram_updated fires on
    # the next coordinator tick and triggers a row refresh.
    model_unload_requested = Signal(str)
    # Settings tab → MainWindow: user clicked the Backend toggle on a
    # row. Payload is (models_attr, backend) where backend is "trt" or
    # "onnx". The host calls models.set_backend_preference(attr,
    # backend) (which unloads the model if loaded) and persists the
    # choice via Settings.model_backends.
    model_backend_changed = Signal(str, str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setProperty("panelTier", "3")

        # Hard minimum width for the pane — applies to BOTH the Parameters
        # and Settings tabs, since they share this one widget. Without an
        # explicit minimumWidth (only a minimumSizeHint), the main QSplitter
        # — which is childrenCollapsible — lets the user drag this pane
        # *below* its content minimum, at which point the widgets can't
        # shrink further and spill off the right edge under the window
        # frame. Setting a real minimumWidth makes the splitter clamp the
        # shrink here and, past this point, collapse the whole pane to 0.
        # Budget = the widest non-compressible row across the two tabs. The
        # Settings tab's "Actions" row (Clear VRAM 85 + Benchmark 100 +
        # Benchmark (Headless) 140, ~357 with margins) is wider than the
        # Parameters IO row, so it's the binding one; + the vertical
        # scrollbar extent (14 px, from rope.qss) so it clears the scrollbar
        # once content scrolls, + 2 px to back it off a hair. The models-
        # inventory filename column is made h-compressible (see
        # _build_models_inventory) so it never drives this wider.
        self.setMinimumWidth(357 + 14 + 2)

        self.widgets: dict[str, QWidget] = {}
        self.values: dict[str, Any] = {}
        # Per-section header/body so collapsed state can be applied later.
        self._section_headers: dict[str, QToolButton] = {}
        self._section_bodies: dict[str, QWidget] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Tabbed container: existing parameter UI under "Parameters",
        # an empty "Settings" page reserved for future controls.
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        outer.addWidget(self._tabs)

        # --- "Parameters" tab: IO row + scrollable parameter list ----------
        params_tab = QWidget()
        params_tab.setProperty("panelTier", "3")
        params_layout = QVBoxLayout(params_tab)
        params_layout.setContentsMargins(0, 0, 0, 0)
        params_layout.setSpacing(0)

        # IO row (Save / Load / Defaults)
        io_row = QFrame(); io_row.setProperty("panelTier", "3")
        io_lay = QHBoxLayout(io_row); io_lay.setContentsMargins(8, 6, 8, 6); io_lay.setSpacing(6)
        for action_name, btn_name in (
            ("save", "SaveParamsButton"),
            ("load", "LoadParamsButton"),
            ("default", "DefaultParamsButton"),
        ):
            btn = IconButton(
                PARAMETER_BY_NAME[btn_name],
                callback=lambda act=action_name: self.io_action.emit(act),
            )
            io_lay.addWidget(btn)
        io_lay.addStretch()
        params_layout.addWidget(io_row)

        # Scrollable parameter list
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        params_layout.addWidget(self._scroll, stretch=1)

        body = QFrame()
        body.setProperty("panelTier", "1")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 3, 0, 0)
        body_layout.setSpacing(3)

        for title, names in SECTIONS:
            section = self._build_section(title, names)
            body_layout.addWidget(section)

        # Tier-3 filler so the empty space below the last section reads
        # as a continuation of the panel surface, not the tier-1 gutter.
        params_filler = QFrame()
        params_filler.setProperty("panelTier", "3")
        body_layout.addWidget(params_filler, stretch=1)
        self._scroll.setWidget(body)

        self._tabs.addTab(params_tab, "Parameters")

        # --- "Settings" tab: system-level controls (threading,
        # detection, recording). Built with the same _build_section
        # machinery as the Parameters tab so widgets register into
        # self.widgets / self.values uniformly — host code reads
        # everything via self._params_pane.values regardless of tab.
        settings_tab = QWidget()
        settings_tab.setProperty("panelTier", "3")
        settings_outer = QVBoxLayout(settings_tab)
        settings_outer.setContentsMargins(0, 0, 0, 0)
        settings_outer.setSpacing(0)

        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QFrame.NoFrame)
        settings_outer.addWidget(settings_scroll, stretch=1)

        settings_body = QFrame()
        settings_body.setProperty("panelTier", "1")
        settings_body_lay = QVBoxLayout(settings_body)
        settings_body_lay.setContentsMargins(0, 3, 0, 0)
        settings_body_lay.setSpacing(3)

        # --- Directories / Files group -----------------------------------
        # Wraps the three path pickers (Models / Output / Embeddings)
        # in a collapsible section matching the Parameters tab's
        # section style for visual consistency.
        dirs_frame, dirs_body = self._build_collapsible_group("Directories / Files")
        mf_frame, self._models_folder_label = self._build_path_picker_row(
            "Models Folder", "./models  (default)",
            self.models_folder_pick_requested,
        )
        dirs_body.addWidget(mf_frame)
        of_frame, self._output_folder_label = self._build_path_picker_row(
            "Output Folder", "(not set)",
            self.output_folder_pick_requested,
        )
        dirs_body.addWidget(of_frame)
        ef_frame, self._embeddings_file_label = self._build_path_picker_row(
            "Embeddings File", "(not set)",
            self.embeddings_file_pick_requested,
        )
        dirs_body.addWidget(ef_frame)
        settings_body_lay.addWidget(dirs_frame)

        # --- Models group ------------------------------------------------
        # Holds the Required Models inventory (Present / Missing badge
        # per file). Wrapped in the same collapsible-group style as
        # Directories / Files so both Settings groups read the same.
        models_frame, models_body = self._build_collapsible_group("Models")
        models_body.addWidget(self._build_models_inventory())
        settings_body_lay.addWidget(models_frame)

        # --- Actions group ----------------------------------------------
        # Build TensorRT, Clear VRAM, Benchmark — process-level actions
        # that don't fit in any one model-tuning section. Adjacent to
        # the Models group above so model-state operations cluster.
        # Buttons are stored in self.widgets so the host can grab refs
        # for enable/disable; clicks route through button_clicked just
        # like in-section buttons (HF refine / clear).
        actions_frame, actions_body = self._build_collapsible_group("Actions")
        actions_row = QHBoxLayout()
        actions_row.setContentsMargins(0, 0, 0, 0)
        actions_row.setSpacing(6)
        for btn_name in ("ClearVramButton", "BenchmarkButton", "BenchmarkHeadlessButton"):
            param = PARAMETER_BY_NAME.get(btn_name)
            if param is None:
                continue
            btn = IconButton(
                param,
                callback=lambda _arg=None, n=btn_name: self.button_clicked.emit(n),
            )
            self.widgets[btn_name] = btn
            actions_row.addWidget(btn)
        actions_row.addStretch()
        actions_body.addLayout(actions_row)
        settings_body_lay.addWidget(actions_frame)

        for title, names in SETTINGS_SECTIONS:
            section = self._build_section(title, names)
            settings_body_lay.addWidget(section)

        # Tier-3 filler so the empty space below the last section reads
        # as a continuation of the panel surface, not the tier-1 gutter.
        settings_filler = QFrame()
        settings_filler.setProperty("panelTier", "3")
        settings_body_lay.addWidget(settings_filler, stretch=1)
        settings_scroll.setWidget(settings_body)

        self._tabs.addTab(settings_tab, "Settings")

    def _build_collapsible_group(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        """Collapsible-section shell for arbitrary child widgets.

        Visual is identical to _build_section (QToolButton header with
        a disclosure arrow, body that hides on collapse), but takes a
        bare title instead of a PARAMETER_BY_NAME list — so the
        Settings tab can group non-parameter content (path pickers,
        models inventory) in the same UI vocabulary as the Parameters
        tab. Returns (wrapper_frame, body_layout). Append widgets to
        body_layout to populate the group."""
        frame = QFrame()
        frame.setProperty("panelTier", "3")
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        header = QToolButton()
        header.setText(title)
        header.setCheckable(True)
        header.setChecked(True)
        header.setArrowType(Qt.DownArrow)
        header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        header.setAutoRaise(True)
        header.setCursor(Qt.PointingHandCursor)
        header.setStyleSheet(
            "QToolButton {"
            "  color: #D0D0D0; font-size: 9pt; font-weight: bold;"
            "  padding: 4px 8px; border: 0; text-align: left;"
            "  background-color: transparent;"
            "}"
        )
        lay.addWidget(header)

        body = QFrame()
        body.setProperty("panelTier", "3")
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(8, 4, 8, 6)
        body_lay.setSpacing(4)
        lay.addWidget(body)

        def _toggle(checked: bool, _body=body, _header=header) -> None:
            _body.setVisible(checked)
            _header.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

        header.toggled.connect(_toggle)
        return frame, body_lay

    def _build_path_picker_row(self, title_text, default_label, signal):
        """Build one Settings-tab path picker (Models / Output /
        Embeddings). All three share the same single-row layout:
        bold title above, current path + Browse button below.

        Returns (frame, value_label) so the caller stores the value
        label and can mutate its text via the matching setter."""
        frame = QFrame()
        frame.setProperty("panelTier", "3")
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(8, 6, 8, 8)
        lay.setSpacing(4)

        title = QLabel(title_text)
        title.setStyleSheet(
            "color: #FFFFFF; font-size: 10pt; font-weight: bold;"
        )
        lay.addWidget(title)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        value_label = QLabel(default_label)
        value_label.setStyleSheet("color: #BBBBBB;")
        value_label.setWordWrap(False)
        # Ignored h-policy so a long path (models / output / embeddings)
        # doesn't force the whole Settings tab wider than the pane —
        # setMinimumWidth(0) alone doesn't help because a QLabel's
        # minimumSizeHint is still the full text width. It clips when
        # narrow; the setters attach a tooltip with the full path.
        value_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        row.addWidget(value_label, stretch=1)

        browse = QPushButton("Browse…")
        browse.setCursor(Qt.PointingHandCursor)
        browse.clicked.connect(signal.emit)
        row.addWidget(browse)
        lay.addLayout(row)

        return frame, value_label

    def set_output_folder(self, path: str | None) -> None:
        """Update the Settings tab's Output Folder display."""
        if not hasattr(self, "_output_folder_label"):
            return
        if path:
            self._output_folder_label.setText(path)
            self._output_folder_label.setToolTip(path)
        else:
            self._output_folder_label.setText("(not set)")
            self._output_folder_label.setToolTip("")

    def set_embeddings_file(self, path: str | None) -> None:
        """Update the Settings tab's Embeddings File display."""
        if not hasattr(self, "_embeddings_file_label"):
            return
        if path:
            self._embeddings_file_label.setText(path)
            self._embeddings_file_label.setToolTip(path)
        else:
            self._embeddings_file_label.setText("(not set)")
            self._embeddings_file_label.setToolTip("")

    def set_models_folder(self, path: str | None) -> None:
        """Update the displayed path under "Models Folder" in the
        Settings tab and re-check the inventory's Present/Missing
        statuses against the new folder. Pass None to revert to the
        default-label."""
        # Track the resolved folder separately so the Refresh button
        # doesn't have to round-trip through the label text.
        self._models_folder_value = path if path else "./models"
        if hasattr(self, "_models_folder_label"):
            if path:
                self._models_folder_label.setText(path)
                self._models_folder_label.setToolTip(path)
            else:
                self._models_folder_label.setText("./models  (default)")
                self._models_folder_label.setToolTip("")
        self.refresh_models_inventory(self._models_folder_value)

    def _build_models_inventory(self) -> QFrame:
        """Build the "Required Models" table inside the Settings tab.

        Columns: filename | ONNX | Backend | Loaded | Unload.
        ONNX column reports on-disk presence. Backend is a toggle
        button that switches between ORT TensorRT-EP (default) and
        ORT CUDA-EP per model — ORT-TRT-EP builds its own engines on
        first run and caches them under models/ort_trt_cache/, so no
        on-disk .engine file check is needed. Loaded shows "✓ TRT" /
        "✓ ONNX" reflecting which provider ORT actually picked. The
        Unload button drops the model so the next call reloads it
        (picking up a backend change). The filename carries a trailing
        " *" when the row is required for a baseline swap."""
        frame = QFrame()
        frame.setProperty("panelTier", "3")
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(8, 6, 8, 8)
        lay.setSpacing(4)

        # Header row: title + Refresh button.
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Required Models")
        title.setStyleSheet(
            "color: #FFFFFF; font-size: 10pt; font-weight: bold;"
        )
        header.addWidget(title)
        header.addStretch()
        refresh = QPushButton("Refresh")
        refresh.setCursor(Qt.PointingHandCursor)
        refresh.clicked.connect(self._on_models_inventory_refresh_clicked)
        header.addWidget(refresh)
        lay.addLayout(header)

        # Column headers.
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(2)
        grid.setContentsMargins(0, 4, 0, 0)
        col_hdr_style = "color: #AAAAAA; font-size: 8pt; font-weight: bold;"
        for col, text in enumerate(["File", "ONNX", "Backend", "Loaded", ""]):
            h = QLabel(text)
            h.setStyleSheet(col_hdr_style)
            grid.addWidget(h, 0, col)

        # Row dicts let refresh_models_inventory mutate only the
        # appropriate cells without rebuilding the whole table.
        self._models_onnx_labels: dict[str, QLabel] = {}
        self._models_loaded_labels: dict[str, QLabel] = {}
        self._models_unload_buttons: dict[str, QPushButton] = {}
        self._models_backend_buttons: dict[str, QPushButton] = {}

        # Backend buttons cover the four ORT-TRT-EP-capable models. The
        # toggle now flips between ORT TensorRT-EP and ORT CUDA-EP at
        # session-create time — no on-disk engine file required since
        # ORT-TRT-EP builds and caches engines itself.
        backend_attrs = {
            'swapper_model', 'swapper_512_model', 'swapper_256_model',
            'retinaface_model', 'recognition_model',
        }

        for i, (fname, role, required) in enumerate(MODEL_INVENTORY):
            row = i + 1  # row 0 is the header
            details = MODEL_INVENTORY_DETAILS.get(fname, {})
            models_attr = details.get("models_attr")

            name_label = QLabel(fname + (" *" if required else ""))
            name_label.setStyleSheet(
                "color: #DDDDDD; font-family: Consolas, 'Courier New', monospace;"
                " font-size: 9pt;"
            )
            # Let the (stretchy) filename column shrink below the full
            # monospace text width when the pane is narrow — otherwise the
            # long names make the Settings tab wider than the pane and the
            # right columns spill under the window frame. Ignored h-policy
            # means the label yields to the fixed columns and clips; the
            # tooltip below still surfaces the full name on hover.
            name_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            if required:
                name_label.setToolTip(f"{role} — required for baseline swap")
            else:
                name_label.setToolTip(role)

            onnx_label = QLabel("—")
            onnx_label.setStyleSheet("color: #888888; font-size: 9pt;")
            onnx_label.setMinimumWidth(28)
            onnx_label.setAlignment(Qt.AlignCenter)
            self._models_onnx_labels[fname] = onnx_label

            # Backend toggle — for the four ORT-TRT-EP-capable rows.
            if models_attr in backend_attrs:
                backend_btn = QPushButton("ONNX")
                backend_btn.setCursor(Qt.PointingHandCursor)
                backend_btn.setMaximumWidth(60)
                backend_btn.setToolTip(
                    "Click to switch the provider ORT picks at the next load.\n"
                    "TRT = TensorrtExecutionProvider (builds + caches engine\n"
                    "internally on first run). ONNX = CUDAExecutionProvider."
                )
                backend_btn.clicked.connect(
                    lambda _c=False, attr=models_attr: self._on_backend_toggle_clicked(attr)
                )
                self._models_backend_buttons[fname] = backend_btn
            else:
                backend_btn = None

            loaded_label = QLabel("—" if models_attr else "")
            loaded_label.setStyleSheet("color: #888888; font-size: 9pt;")
            loaded_label.setMinimumWidth(56)
            self._models_loaded_labels[fname] = loaded_label

            if models_attr:
                unload_btn = QPushButton("Unload")
                unload_btn.setCursor(Qt.PointingHandCursor)
                unload_btn.setEnabled(False)
                unload_btn.setMaximumWidth(60)
                unload_btn.clicked.connect(
                    lambda _c=False, attr=models_attr: self.model_unload_requested.emit(attr)
                )
                self._models_unload_buttons[fname] = unload_btn
            else:
                unload_btn = None

            grid.addWidget(name_label, row, 0)
            grid.addWidget(onnx_label, row, 1)
            if backend_btn is not None:
                grid.addWidget(backend_btn, row, 2)
            grid.addWidget(loaded_label, row, 3)
            if unload_btn is not None:
                grid.addWidget(unload_btn, row, 4)

        grid.setColumnStretch(0, 1)
        lay.addLayout(grid)

        legend = QLabel("* required for a baseline swap")
        legend.setStyleSheet("color: #777777; font-size: 8pt; font-style: italic;")
        lay.addWidget(legend)

        return frame

    def set_models_instance(self, models) -> None:
        """Give the pane a reference to Models so it can read live
        loaded state. Called once from MainWindow after Coordinator
        construction. None disables the Loaded column."""
        self._models = models
        self.refresh_models_inventory(
            getattr(self, "_models_folder_value", "./models")
        )

    def _on_models_inventory_refresh_clicked(self) -> None:
        self.refresh_models_inventory(
            getattr(self, "_models_folder_value", "./models")
        )

    def _on_backend_toggle_clicked(self, attr_name: str) -> None:
        """Flip the row's backend between TRT and ONNX. Reads the
        currently-effective backend from the button label text; emits
        model_backend_changed with the new value."""
        btn = next(
            (b for fname, b in self._models_backend_buttons.items()
             if MODEL_INVENTORY_DETAILS.get(fname, {}).get("models_attr") == attr_name),
            None,
        )
        current = btn.text() if btn is not None else "TRT"
        new_backend = "onnx" if current == "TRT" else "trt"
        self.model_backend_changed.emit(attr_name, new_backend)

    def refresh_models_inventory(self, folder: str) -> None:
        """Re-check ONNX presence, loaded state, and backend toggle
        for every row."""
        if not hasattr(self, "_models_onnx_labels"):
            return
        models = getattr(self, "_models", None)
        for fname in self._models_onnx_labels:
            details = MODEL_INVENTORY_DETAILS.get(fname, {})
            models_attr = details.get("models_attr")

            # ONNX presence.
            try:
                onnx_present = os.path.isfile(os.path.join(folder, fname))
            except (TypeError, ValueError):
                onnx_present = False
            self._set_presence(self._models_onnx_labels[fname], onnx_present)

            # Backend toggle — always actionable (ORT-TRT-EP builds
            # its own engine on first run, so no on-disk engine file
            # check is needed). The label shows what the next load
            # will pick.
            backend_btn = self._models_backend_buttons.get(fname)
            if backend_btn is not None:
                if models is not None and hasattr(models, "get_backend_preference"):
                    pref = models.get_backend_preference(models_attr)
                else:
                    pref = None
                effective = "ONNX" if pref == "onnx" else "TRT"
                backend_btn.setText(effective)
                backend_btn.setEnabled(onnx_present)

            # Loaded state. When loaded, distinguish TRT vs ONNX so the
            # user can see whether their toggle took effect.
            loaded_label = self._models_loaded_labels[fname]
            btn = self._models_unload_buttons.get(fname)
            if models_attr and models is not None and hasattr(models, "is_model_loaded"):
                loaded = bool(models.is_model_loaded(models_attr))
                if loaded:
                    uses_trt = self._reads_uses_trt_flag(models, models_attr)
                    if uses_trt is True:
                        loaded_label.setText("✓ TRT")
                    elif uses_trt is False:
                        loaded_label.setText("✓ ONNX")
                    else:
                        loaded_label.setText("✓ Loaded")
                    loaded_label.setStyleSheet(
                        "color: #5FD35F; font-size: 9pt; font-weight: bold;"
                    )
                else:
                    loaded_label.setText("—")
                    loaded_label.setStyleSheet("color: #888888; font-size: 9pt;")
                if btn is not None:
                    btn.setEnabled(loaded)
            else:
                # No tracked attribute or no Models instance yet — leave
                # blank rather than show a misleading "—".
                loaded_label.setText("" if not models_attr else "—")
                loaded_label.setStyleSheet("color: #888888; font-size: 9pt;")
                if btn is not None:
                    btn.setEnabled(False)

    @staticmethod
    def _reads_uses_trt_flag(models, models_attr: str):
        """Map a Models attribute name to the corresponding _uses_trt
        flag. Returns True / False / None when no flag exists."""
        flag_attr = {
            "swapper_model":       "_swapper_uses_trt",
            "swapper_512_model":   "_swapper_512_uses_trt",
            "swapper_256_model":   "_swapper_256_uses_trt",
            "retinaface_model":    "_retinaface_uses_trt",
            "recognition_model":   "_recognition_uses_trt",
        }.get(models_attr)
        if flag_attr is None:
            return None
        return bool(getattr(models, flag_attr, False))

    @staticmethod
    def _set_presence(label: QLabel, present: bool) -> None:
        if present:
            label.setText("✓ Present")
            label.setStyleSheet(
                "color: #5FD35F; font-size: 9pt; font-weight: bold;"
            )
        else:
            label.setText("✗ Missing")
            label.setStyleSheet(
                "color: #D35F5F; font-size: 9pt; font-weight: bold;"
            )

    def _build_section(self, title: str, names: list[str]) -> QFrame:
        frame = QFrame()
        frame.setProperty("panelTier", "3")
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Clickable header — QToolButton with a disclosure arrow that
        # flips between Down (expanded) and Right (collapsed).
        header = QToolButton()
        header.setText(title)
        header.setCheckable(True)
        header.setChecked(True)
        header.setArrowType(Qt.DownArrow)
        header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        header.setAutoRaise(True)
        header.setCursor(Qt.PointingHandCursor)
        header.setStyleSheet(
            "QToolButton {"
            "  color: #D0D0D0; font-size: 9pt; font-weight: bold;"
            "  padding: 4px 8px; border: 0; text-align: left;"
            "  background-color: transparent;"
            "}"
        )
        lay.addWidget(header)

        # Body holds the parameter widgets. Hidden when the header is
        # unchecked; setVisible(False) makes the layout collapse to 0.
        body = QFrame()
        body.setProperty("panelTier", "3")
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(8, 4, 8, 6)
        body_lay.setSpacing(4)
        for name in names:
            # HF cache buttons collapse into one right-justified "Cache:"
            # row instead of one button per line. HFRefineButton builds
            # the row (with both buttons); HFClearCacheButton is consumed
            # there, so skip it when the loop reaches it.
            if name == "HFClearCacheButton":
                continue
            if name == "HFRefineButton":
                body_lay.addWidget(self._build_grouped_button_row(
                    "Cache:", ["HFRefineButton", "HFClearCacheButton"], align="right",
                ))
                continue

            param = PARAMETER_BY_NAME.get(name)
            if param is None:
                continue
            # The HF mode selector is left-justified (label + buttons hug
            # the left edge) per the parameters layout.
            if name == "HighFidelityModeTextSel":
                widget = TextSelection(param, justify="left")
            else:
                widget = _widget_for(param)
            self.widgets[name] = widget
            # Buttons don't carry a value to mirror into self.values and
            # use clicked_with_arg(name, _) instead of value_changed.
            if param.kind == "button":
                widget.clicked_with_arg.connect(
                    lambda btn_name, _arg=None: self.button_clicked.emit(btn_name)
                )
            else:
                self.values[name] = param.default
                widget.value_changed.connect(self._on_value_changed)
            body_lay.addWidget(widget)
        lay.addWidget(body)

        self._section_headers[title] = header
        self._section_bodies[title] = body

        def _toggle(checked: bool, _body=body, _header=header, _title=title) -> None:
            _body.setVisible(checked)
            _header.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
            # `collapsed = not checked` so the persisted bool matches the
            # intuitive "is this section hidden?" sense.
            self.section_toggled.emit(_title, not checked)
        header.toggled.connect(_toggle)

        return frame

    def _build_grouped_button_row(
        self, label_text: str, button_names: list[str], *, align: str = "right",
    ) -> QFrame:
        """One row: a `label_text` label plus the named buttons on a single
        line. align='right' pushes label+buttons to the right edge (leading
        stretch); align='left' packs them left (trailing stretch). Buttons
        register into self.widgets and route clicks through button_clicked
        exactly like the per-row path, so host handlers are unchanged."""
        row = QFrame()
        row.setProperty("panelTier", "3")
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        if align == "right":
            lay.addStretch(1)

        label = QLabel(label_text)
        lay.addWidget(label)

        for name in button_names:
            param = PARAMETER_BY_NAME.get(name)
            if param is None:
                continue
            btn = _widget_for(param)
            self.widgets[name] = btn
            btn.clicked_with_arg.connect(
                lambda btn_name, _arg=None: self.button_clicked.emit(btn_name)
            )
            lay.addWidget(btn)

        if align == "left":
            lay.addStretch(1)

        return row

    def _on_value_changed(self, name: str, value: Any) -> None:
        self.values[name] = value
        self.params_changed.emit(dict(self.values))

    # ----- Public bulk operations -------------------------------------------------

    def apply_values(self, values: dict[str, Any], *, emit: bool = False) -> None:
        """Push a {name: value} dict into the widgets (used after JSON load).

        Reads the canonical value back from the widget (via `get()`) so
        integer-step sliders end up in `self.values` as `int`, not the
        original JSON-deserialized `float`."""
        for name, value in values.items():
            widget = self.widgets.get(name)
            if widget is None:
                continue
            widget.set(value, request_frame=False)
            getter = getattr(widget, 'get', None)
            self.values[name] = getter() if callable(getter) else value
        if emit:
            self.params_changed.emit(dict(self.values))

    def load_defaults(self, *, emit: bool = True) -> None:
        for name, widget in self.widgets.items():
            widget.load_default()
            # Buttons are momentary — they aren't part of the params
            # value snapshot, so don't write their default state into
            # self.values (which would then leak into params_changed).
            if PARAMETER_BY_NAME[name].kind != "button":
                self.values[name] = PARAMETER_BY_NAME[name].default
        if emit:
            self.params_changed.emit(dict(self.values))

    def apply_collapsed_state(self, state: dict[str, bool]) -> None:
        """Restore each section's collapsed/expanded state without firing
        section_toggled (avoids saving on initial setup)."""
        for title, header in self._section_headers.items():
            collapsed = bool(state.get(title, False))
            header.blockSignals(True)
            try:
                header.setChecked(not collapsed)
                header.setArrowType(Qt.RightArrow if collapsed else Qt.DownArrow)
            finally:
                header.blockSignals(False)
            self._section_bodies[title].setVisible(not collapsed)

    def collapsed_state(self) -> dict[str, bool]:
        return {
            title: not header.isChecked()
            for title, header in self._section_headers.items()
        }
