"""Center pane — toggles row, preview placeholder, timeline, media buttons.

Mirrors rope/GUI.py lines ~385-520. Layout:

    +-------------------------------------------------+
    | Audio | MaskView |  Preview Mode (TextSel)     |  toggle row
    +-------------------------------------------------+
    |                                                 |
    |              PREVIEW (QLabel placeholder        |  stretch=1
    |              — QOpenGLWidget in Phase D)        |
    |                                                 |
    +-------------------------------------------------+
    |  Timeline + frame entry + markers               |  fixed height
    +-------------------------------------------------+
    | SaveImage |  TLBeg TLLeft Record Play TLRight  |
    |           |  AddMrk DelMrk PrevMrk NextMrk      |  media buttons
    +-------------------------------------------------+
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from rope.qt.parameters import PARAMETER_BY_NAME
from rope.qt.widgets.button import IconButton
from rope.qt.widgets.embeddings_pane import EmbeddingsPane
from rope.qt.widgets.face_gallery import FaceGallery
from rope.qt.widgets.preview import PreviewWidget
from rope.qt.widgets.text_selection import TextSelection
from rope.qt.widgets.timeline import Timeline


def _tier(tier: int) -> QFrame:
    f = QFrame()
    f.setProperty("panelTier", str(tier))
    return f


class CenterPane(QFrame):
    """Aggregates the preview region + playback chrome.

    Public signals (forwarded to bus by main_window):
        toggle_audio()
        toggle_mask_view()
        preview_mode_changed(str)
        play_pressed()
        record_pressed()
        nudge_left() / nudge_right() / jump_to_start()
        add_marker() / del_marker() / prev_marker() / next_marker()
        frame_requested(int)
        scrub_started() / scrub_ended()
        save_image()
    """

    toggle_audio = Signal()
    toggle_mask_view = Signal()
    preview_mode_changed = Signal(str)
    play_pressed = Signal()
    record_pressed = Signal()
    nudge_left = Signal()
    nudge_right = Signal()
    jump_to_start = Signal()
    add_marker = Signal()
    del_marker = Signal()
    prev_marker = Signal()
    next_marker = Signal()
    frame_requested = Signal(int)
    scrub_started = Signal()
    scrub_ended = Signal()
    save_image = Signal()
    find_faces_pressed = Signal()
    clear_faces_pressed = Signal()
    toggle_swap_faces = Signal()
    preload_pressed = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        # Tier 3 (#28282E) matches the source-faces / target-videos /
        # embeddings panes so the center pane's chrome (toggle row,
        # timeline, media buttons, Found Faces) shares their tone.
        self.setProperty("panelTier", "3")

        # Mix of IconButton and plain QPushButton (Find/Clear Faces use
        # the latter so they match the Embeddings pane's Save/Refresh
        # chrome). QWidget covers both.
        self.buttons: dict[str, QWidget] = {}

        layout = QVBoxLayout(self)
        # Zero outer margins so the QSplitter handle runs flush to the
        # left/right pane borders, matching how the left-pane vertical
        # splitter (videos/faces) sits flush against the main splitter.
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Vertical splitter so the user can grow/shrink the embeddings
        # pane against the video preview. Top half contains everything
        # *except* embeddings; the preview inside it carries the
        # stretch=1, so when the user drags the handle down, embeddings
        # shrinks and the preview (the only stretchy item above)
        # absorbs the freed space. Found Faces / timeline / media rows
        # all have fixed heights, so they don't move when the handle
        # is dragged.
        self.center_splitter = QSplitter(Qt.Vertical)
        self.center_splitter.setChildrenCollapsible(False)
        # QSS sets the handle color; handle width must be set in code
        # because Qt ignores the QSS `width: 3px` on QSplitter::handle.
        self.center_splitter.setHandleWidth(3)

        top_widget = QWidget()
        top_widget.setProperty("panelTier", "3")
        top_lay = QVBoxLayout(top_widget)
        top_lay.setContentsMargins(0, 0, 0, 0)
        top_lay.setSpacing(4)
        top_lay.addWidget(self._build_toggle_row())
        top_lay.addWidget(self._build_preview(), stretch=1)
        # Timeline + media controls sit immediately under the preview so
        # transport/scrub stays glued to the video, with faces / embeddings
        # below the playback chrome.
        top_lay.addWidget(self._build_timeline_row())
        top_lay.addWidget(self._build_media_row())
        # Found Faces lives in a single QFrame (header + gallery in one
        # tier-3 surface) so the title row and the thumbnail strip share
        # one continuous background — same single-frame layout as the
        # Embeddings pane.
        self.found_faces_gallery = FaceGallery()
        top_lay.addWidget(self._build_found_faces_pane())

        self.embeddings_pane = EmbeddingsPane()

        self.center_splitter.addWidget(top_widget)
        self.center_splitter.addWidget(self.embeddings_pane)
        # Initial split: heavy weight on the top (video + chrome) so
        # embeddings starts compact. Users can drag to taste.
        self.center_splitter.setStretchFactor(0, 1)
        self.center_splitter.setStretchFactor(1, 0)

        layout.addWidget(self.center_splitter)

    # ---- Toggle row --------------------------------------------------------------

    def _build_toggle_row(self) -> QFrame:
        row = _tier(3); row.setFixedHeight(28)
        lay = QHBoxLayout(row); lay.setContentsMargins(4, 2, 4, 2); lay.setSpacing(6)

        # Preload Models sits at the far left of the toggle row (left of
        # Audio). Model preload is manual — clicking builds the pipeline
        # sessions for the current selections + thread count; the host
        # (main_window) flips this button's text/color as they load.
        preload_btn = QPushButton("Preload Models")
        preload_btn.setCursor(Qt.PointingHandCursor)
        preload_btn.setToolTip(
            "Build the swap-pipeline sessions (detector, recognizer, "
            "inswapper) for your current selections and thread count.\n"
            "Runs in the background — the window stays responsive while "
            "engines build. Turns green once the models are loaded."
        )
        preload_btn.clicked.connect(lambda *_: self.preload_pressed.emit())
        self.buttons["PreloadModelsButton"] = preload_btn
        lay.addWidget(preload_btn)

        self.buttons["AudioButton"] = IconButton(
            PARAMETER_BY_NAME["Audio"], callback=lambda *_: self.toggle_audio.emit()
        )
        lay.addWidget(self.buttons["AudioButton"])

        self.buttons["MaskViewButton"] = IconButton(
            PARAMETER_BY_NAME["MaskView"], callback=lambda *_: self.toggle_mask_view.emit()
        )
        lay.addWidget(self.buttons["MaskViewButton"])

        self.preview_mode = TextSelection(PARAMETER_BY_NAME["PreviewModeTextSel"])
        self.preview_mode.value_changed.connect(lambda _name, value: self.preview_mode_changed.emit(value))
        lay.addWidget(self.preview_mode, stretch=1)

        return row

    # ---- Preview placeholder -----------------------------------------------------

    def _build_preview(self) -> QWidget:
        self.preview = PreviewWidget()
        self.preview.setMinimumHeight(360)
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Click anywhere on the preview to toggle play/stop, matching the
        # Tk binding on tk.Label <ButtonRelease-1>.
        self.preview.clicked.connect(self.play_pressed)
        return self.preview

    # ---- Play-state helpers ------------------------------------------------------

    def set_play_state(self, playing: bool) -> None:
        """Reflect playback state on the Play button without firing its
        callback. Called by main_window on play_pressed (to flip the icon
        before VM acts) and on bus.stop_play (when VM auto-stops at EOF).
        """
        btn = self.buttons.get("TLPlayButton")
        if btn is not None:
            btn.set(bool(playing), request_frame=False)

    def set_record_state(self, recording: bool) -> None:
        btn = self.buttons.get("TLRecButton")
        if btn is not None:
            btn.set(bool(recording), request_frame=False)

    def set_preload_button_state(self, state: str) -> None:
        """Reflect model-preload progress on the Preload Models button.

        state:
            'idle'    — not preloaded: "Preload Models", default style, enabled.
            'loading' — build in progress: "Preloading…", disabled.
            'loaded'  — all selected sessions live: "Models Loaded", green.
        Only `color` is set inline so the button keeps its base look
        (background / border / padding) from the global stylesheet.
        """
        btn = self.buttons.get("PreloadModelsButton")
        if btn is None:
            return
        if state == "loading":
            btn.setText("Preloading…")
            btn.setEnabled(False)
            btn.setStyleSheet("")
        elif state == "loaded":
            btn.setText("Models Loaded")
            btn.setEnabled(True)
            btn.setStyleSheet("color: #5FD35F; font-weight: bold;")
        else:  # 'idle'
            btn.setText("Preload Models")
            btn.setEnabled(True)
            btn.setStyleSheet("")

    # ---- Faces row (Find / Clear / Swap) -----------------------------------------

    def _build_found_faces_pane(self) -> QFrame:
        # One tier-3 QFrame holding the "Found Faces" header (title +
        # Find / Clear buttons) and the FaceGallery directly below it
        # — same single-surface layout used by EmbeddingsPane, so
        # there's no visible seam between the title row and the
        # thumbnail strip. SwapFaces lives in the media row, not
        # here.
        frame = QFrame()
        frame.setProperty("panelTier", "3")
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        title = QLabel("Found Faces")
        title.setStyleSheet("font-weight: bold;")
        header.addWidget(title)
        header.addStretch()

        find_btn = QPushButton("Find Faces")
        find_btn.clicked.connect(lambda *_: self.find_faces_pressed.emit())
        self.buttons["FindFacesButton"] = find_btn
        header.addWidget(find_btn)

        clear_btn = QPushButton("Clear Faces")
        clear_btn.clicked.connect(lambda *_: self.clear_faces_pressed.emit())
        self.buttons["ClearFacesButton"] = clear_btn
        header.addWidget(clear_btn)

        outer.addLayout(header)
        outer.addWidget(self.found_faces_gallery, stretch=1)

        return frame

    # ---- Timeline ----------------------------------------------------------------

    def _build_timeline_row(self) -> QWidget:
        wrap = _tier(3)
        wrap.setFixedHeight(36)
        lay = QHBoxLayout(wrap); lay.setContentsMargins(4, 2, 4, 2); lay.setSpacing(0)
        self.timeline = Timeline()
        self.timeline.frame_requested.connect(self.frame_requested)
        self.timeline.scrub_started.connect(self.scrub_started)
        self.timeline.scrub_ended.connect(self.scrub_ended)
        lay.addWidget(self.timeline)
        return wrap

    # ---- Media-buttons row -------------------------------------------------------

    def _build_media_row(self) -> QFrame:
        row = _tier(3); row.setFixedHeight(44)
        outer = QHBoxLayout(row); outer.setContentsMargins(4, 4, 4, 4); outer.setSpacing(8)

        # Left side: SwapFaces toggle, then Save Image. SwapFaces moved
        # here from the faces row so it lives with the per-frame
        # transport controls — its on/off state is what gates swap
        # rendering during play.
        self.buttons["SwapFacesButton"] = IconButton(
            PARAMETER_BY_NAME["SwapFaces"], callback=lambda *_: self.toggle_swap_faces.emit(),
        )
        outer.addWidget(self.buttons["SwapFacesButton"])

        self.buttons["SaveImageButton"] = IconButton(
            PARAMETER_BY_NAME["SaveImageButton"], callback=lambda *_: self.save_image.emit()
        )
        outer.addWidget(self.buttons["SaveImageButton"])

        outer.addStretch()

        # Center: transport controls
        for btn_name, signal in (
            ("TLBegButton", self.jump_to_start),
            ("TLLeftButton", self.nudge_left),
            ("TLRecButton", self.record_pressed),
            ("TLPlayButton", self.play_pressed),
            ("TLRightButton", self.nudge_right),
        ):
            param_name = btn_name.replace("Button", "")
            # Map legacy names to dataclass names
            param_name = {
                "TLBeg": "TLBeginning", "TLLeft": "TLLeft",
                "TLRec": "Record", "TLPlay": "Play", "TLRight": "TLRight",
            }[param_name]
            btn = IconButton(
                PARAMETER_BY_NAME[param_name], callback=lambda *_, s=signal: s.emit()
            )
            self.buttons[btn_name] = btn
            outer.addWidget(btn)

        outer.addStretch()

        # Right: marker controls
        for btn_name, param_name, signal in (
            ("AddMarkerButton", "AddMarkerButton", self.add_marker),
            ("DelMarkerButton", "DelMarkerButton", self.del_marker),
            ("PrevMarkerButton", "PrevMarkerButton", self.prev_marker),
            ("NextMarkerButton", "NextMarkerButton", self.next_marker),
        ):
            btn = IconButton(
                PARAMETER_BY_NAME[param_name], callback=lambda *_, s=signal: s.emit()
            )
            self.buttons[btn_name] = btn
            outer.addWidget(btn)

        return row

    # ---- Public API used by main_window ------------------------------------------

    def attach_info_label(self, info_label) -> None:
        # self.buttons now mixes IconButton (which exposes
        # add_info_frame for the hover-info system) and plain
        # QPushButton (Find/Clear Faces). Skip plain buttons — they
        # don't participate in the info-label flow.
        for btn in self.buttons.values():
            attach = getattr(btn, "add_info_frame", None)
            if callable(attach):
                attach(info_label)
        self.preview_mode.add_info_frame(info_label)
