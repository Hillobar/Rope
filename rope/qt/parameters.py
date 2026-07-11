"""Typed parameter schema for the Qt port.

DEFAULT_DATA (in rope/qt/_default_data.py) is the historic dict from the
old Tk GUI — keyed with stringly-typed suffixes (Name+'Amount',
Name+'State', Name+'Mode', Name+'IconOn', ...). This module reflects
that data into typed dataclasses so widgets (rope/qt/widgets/*) can ask
for `.default`, `.min`, `.max`, `.modes`, etc. without parsing suffixes.

The on-disk format for `saved_parameters.json` is unchanged — it stays a
flat {widget_name: value} mapping, exactly what the Tk
GUI.parameter_io() wrote.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

from rope.qt._default_data import DEFAULT_DATA


ControlScope = Literal["control", "parameter", "merge"]
ButtonDisplay = Literal["icon", "text", "both"]


@dataclass(frozen=True)
class SliderParam:
    name: str
    label: str
    scope: ControlScope = "parameter"
    kind: Literal["slider"] = field(default="slider", init=False)

    @property
    def default(self) -> float: return DEFAULT_DATA[self.name + "Amount"]
    @property
    def min(self) -> float: return DEFAULT_DATA[self.name + "Min"]
    @property
    def max(self) -> float: return DEFAULT_DATA[self.name + "Max"]
    @property
    def inc(self) -> float: return DEFAULT_DATA[self.name + "Inc"]
    @property
    def info_text(self) -> str: return DEFAULT_DATA.get(self.name + "InfoText", "")


@dataclass(frozen=True)
class SwitchParam:
    name: str
    label: str
    scope: ControlScope = "parameter"
    kind: Literal["switch"] = field(default="switch", init=False)

    @property
    def default(self) -> bool: return bool(DEFAULT_DATA.get(self.name + "State", False))
    @property
    def info_text(self) -> str: return DEFAULT_DATA.get(self.name + "InfoText", "")


@dataclass(frozen=True)
class SelectParam:
    name: str
    label: str
    scope: ControlScope = "parameter"
    kind: Literal["select"] = field(default="select", init=False)

    @property
    def default(self) -> str: return DEFAULT_DATA[self.name + "Mode"]
    @property
    def modes(self) -> list[str]: return list(DEFAULT_DATA[self.name + "Modes"])
    @property
    def info_text(self) -> str: return DEFAULT_DATA.get(self.name + "InfoText", "")


@dataclass(frozen=True)
class EntryParam:
    name: str
    label: str
    scope: ControlScope = "parameter"
    kind: Literal["entry"] = field(default="entry", init=False)

    @property
    def default(self) -> str: return DEFAULT_DATA.get(self.name, "")
    @property
    def info_text(self) -> str: return DEFAULT_DATA.get(self.name + "InfoText", "")


@dataclass(frozen=True)
class ButtonParam:
    name: str
    label: str  # tooltip / accessibility label; visible text is .text below
    scope: ControlScope = "control"
    width: int = 125
    height: int = 20
    kind: Literal["button"] = field(default="button", init=False)

    @property
    def text(self) -> str: return DEFAULT_DATA.get(self.name + "Text", "")
    @property
    def display(self) -> ButtonDisplay: return DEFAULT_DATA.get(self.name + "Display", "both")
    @property
    def icon_on(self) -> str | None: return DEFAULT_DATA.get(self.name + "IconOn")
    @property
    def icon_off(self) -> str | None: return DEFAULT_DATA.get(self.name + "IconOff")
    @property
    def icon_hover(self) -> str | None: return DEFAULT_DATA.get(self.name + "IconHover")
    @property
    def default(self) -> bool: return bool(DEFAULT_DATA.get(self.name + "State", False))
    @property
    def info_text(self) -> str: return DEFAULT_DATA.get(self.name + "InfoText", "")


Parameter = SliderParam | SwitchParam | SelectParam | EntryParam | ButtonParam


# Canonical widget list, transcribed from rope/GUI.py construction.
# Each entry: a Param dataclass that knows how to resolve its own schema
# fields by reading DEFAULT_DATA through properties.
PARAMETERS: list[Parameter] = [
    # ---- Top-bar buttons (scope=control)
    ButtonParam("StartRope", "Start Rope", width=200),
    ButtonParam("OutputFolder", "Output Folder", width=190),
    ButtonParam("EmbedFile", "Embedding File", width=190),
    ButtonParam("ClearVramButton", "Clear VRAM", width=85),
    ButtonParam("BenchmarkButton", "Benchmark", width=100),
    ButtonParam("BenchmarkHeadlessButton", "Benchmark (Headless)", width=140),
    # ---- Left-pane folder buttons
    ButtonParam("LoadTVideos", "Select Target Videos Folder", width=170),
    ButtonParam("LoadSFaces", "Select Source Faces Folder", width=170),
    # ---- Center-pane media controls
    ButtonParam("Audio", "Enable Audio", width=100),
    ButtonParam("MaskView", "Show Mask", width=100),
    ButtonParam("SaveImageButton", "Save Image", width=100),
    ButtonParam("AutoSwapButton", "Auto Swap", width=100),
    ButtonParam("TLBeginning", "Timeline Start", width=20),
    ButtonParam("TLLeft", "Nudge Left", width=20),
    ButtonParam("Record", "Record", width=20),
    ButtonParam("Play", "Play", width=20),
    ButtonParam("TLRight", "Nudge Right", width=20),
    ButtonParam("AddMarkerButton", "Add Marker", width=20),
    ButtonParam("DelMarkerButton", "Delete Marker", width=20),
    ButtonParam("PrevMarkerButton", "Previous Marker", width=20),
    ButtonParam("NextMarkerButton", "Next Marker", width=20),
    # ---- Found faces row
    ButtonParam("FindFaces", "Find Faces", width=112, height=33),
    ButtonParam("ClearFaces", "Clear Faces", width=112, height=33),
    ButtonParam("SwapFaces", "Swap Faces", width=112, height=33),
    ButtonParam("DelEmbed", "Delete Embedding", width=112, height=33),
    # ---- Parameter IO row
    ButtonParam("SaveParamsButton", "Save Parameters", width=100),
    ButtonParam("LoadParamsButton", "Load Parameters", width=100),
    ButtonParam("DefaultParamsButton", "Load Defaults", width=100),
    # ---- Capture / misc
    ButtonParam("FindSimilarFacesButton", "Find Similar", width=120),
    ButtonParam("CapturePlayButton", "Capture Play/Pause", width=120),
    ButtonParam("CaptureReopenButton", "Reopen Viewfinder", width=140),

    # ---- Preview-mode selector
    SelectParam("PreviewModeTextSel", "Mode:", scope="control"),

    # ---- Restorer
    SwitchParam("RestorerSwitch", "Restorer"),
    SelectParam("RestorerTypeTextSel", "Restorer Type"),
    SelectParam("RestorerDetTypeTextSel", "Detection Alignment"),
    SliderParam("RestorerSlider", "Blend"),

    # ---- Similarity threshold
    SliderParam("ThresholdSlider", "Similarity Threshold"),

    # ---- Orientation
    SwitchParam("OrientSwitch", "Orientation"),
    SwitchParam("OrientAutoSwitch", "Auto-detect Orientation"),
    SliderParam("OrientSlider", "Angle"),

    # ---- Strength
    SwitchParam("StrengthSwitch", "Strength"),
    SliderParam("StrengthSlider", "Amount"),
    SwitchParam("ColorMatchSwitch", "Color Match (LAB)"),

    # ---- Likeness / fidelity (input-embedding tuning + 2-pass refinement)
    SliderParam("LikenessSlider", "Likeness"),
    SliderParam("EmbExtrapSlider", "Distinctiveness"),
    SwitchParam("HighFidelitySwitch", "High Fidelity (2-pass)"),
    SliderParam("HighFidelityAlphaSlider", "HF Correction"),
    SelectParam("HighFidelityModeTextSel", "Mode:"),
    ButtonParam("HFRefineButton", "HF Refine Cache", scope="parameter", width=70, height=22),
    ButtonParam("HFClearCacheButton", "HF Clear Cache", scope="parameter", width=70, height=22),

    # ---- Border / blend
    SliderParam("BorderTopSlider", "Top Border Distance"),
    SliderParam("BorderSidesSlider", "Sides Border Distance"),
    SliderParam("BorderBottomSlider", "Bottom Border Distance"),
    SliderParam("BorderBlurSlider", "Border Blend"),
    SliderParam("BlendSlider", "Overall Mask Blend"),

    # ---- Differencer / occluder / parsers
    SwitchParam("DiffSwitch", "Differencing"),
    SliderParam("DiffSlider", "Difference Amount"),
    SwitchParam("OccluderSwitch", "Occluder"),
    SliderParam("OccluderSlider", "Occluder Size"),
    SwitchParam("DFLXSegSwitch", "DFL XSeg Mask"),
    SliderParam("DFLXSegSizeSlider", "XSeg Size"),
    SliderParam("DFLXSegBlurSlider", "XSeg Blur"),
    SwitchParam("FaceParserSwitch", "Face Parser"),
    SliderParam("FaceParserSlider", "Background"),
    SliderParam("MouthParserSlider", "Mouth"),

    # ---- Color
    SwitchParam("ColorSwitch", "Color Adjustments"),
    SliderParam("ColorRedSlider", "Red"),
    SliderParam("ColorGreenSlider", "Green"),
    SliderParam("ColorBlueSlider", "Blue"),
    SliderParam("ColorGammaSlider", "Gamma"),
    SliderParam("ColorContrastSlider", "Contrast"),
    SliderParam("ColorSaturationSlider", "Saturation"),

    # ---- Face adjustments
    SwitchParam("FaceAdjSwitch", "Input Face Adjustments"),
    SliderParam("KPSXSlider", "KPS - X"),
    SliderParam("KPSYSlider", "KPS - Y"),
    SliderParam("KPSScaleSlider", "KPS - Scale"),
    SliderParam("FaceScaleSlider", "Face Scale"),

    # ---- Find-similar threshold
    SliderParam("FindSimilarThresholdSlider", "Find Similar Threshold"),

    # ---- Threading / detection
    SliderParam("ThreadsSlider", "Threads"),
    SelectParam("ModelSessionsTextSel", "Model Sessions"),
    SelectParam("DetectTypeTextSel", "Detection Type"),
    SelectParam("DetectInputSizeTextSel", "Detect Input Size"),
    SliderParam("DetectScoreSlider", "Detect Score"),
    SelectParam("RecordTypeTextSel", "Record Type"),
    SliderParam("VideoQualSlider", "FFMPEG Quality"),
    SelectParam("MergeTextSel", "Merge Math", scope="merge"),
    SelectParam("SwapperTypeTextSel", "Swapper Resolution"),

    # ---- Capture
    SliderParam("CaptureFPSSlider", "Capture FPS", scope="control"),
    SliderParam("CaptureBorderSlider", "Capture Border", scope="control"),
    SwitchParam("CaptureLockRegionSwitch", "Lock Capture Region", scope="control"),
    SwitchParam("CaptureAlwaysOnTopSwitch", "Always On Top", scope="control"),
]


PARAMETER_BY_NAME: dict[str, Parameter] = {p.name: p for p in PARAMETERS}


def parameters_by_kind(kind: str) -> Iterable[Parameter]:
    return (p for p in PARAMETERS if p.kind == kind)


def parameters_by_scope(scope: ControlScope) -> Iterable[Parameter]:
    return (p for p in PARAMETERS if p.scope == scope)


def default_values(scope: ControlScope | None = None, *, include_buttons: bool = False) -> dict[str, Any]:
    """Return a {name: default_value} snapshot, optionally filtered by scope.

    By default buttons are excluded — they have no "value" in the
    parameter sense. The Tk GUI's control dict, however, treated button
    state (Audio toggle, MaskView toggle, SwapFaces toggle) as part of
    `vm.control`, and VideoManager does `if not self.control['SwapFacesButton']`
    style lookups. Pass include_buttons=True when seeding that dict.
    """
    out: dict[str, Any] = {}
    for p in PARAMETERS:
        if scope is not None and p.scope != scope:
            continue
        if p.kind == "button" and not include_buttons:
            continue
        out[p.name] = p.default
    return out


# Legacy widget-registry names that VideoManager keys off of.
# The Tk GUI's self.widget registry named these buttons with a 'Button'
# suffix (self.widget['SwapFacesButton'] wrapping the 'SwapFaces' ButtonParam),
# and self.control was keyed by widget name. VideoManager.swap_core et al
# do `self.control['MaskViewButton']` etc., so the schema-default key needs
# aliasing in any dict that gets handed to VM.
_CONTROL_KEY_ALIASES: tuple[tuple[str, str], ...] = (
    ("SwapFaces", "SwapFacesButton"),
    ("Audio", "AudioButton"),
    ("MaskView", "MaskViewButton"),
    ("AutoSwapButton", "AutoSwapButton"),
)


def seed_control_dict() -> dict[str, Any]:
    """Build the control snapshot the way the Tk GUI's widget registry did.

    Used by both the Coordinator (to seed vm.control on construction so
    the very first frame request doesn't TypeError) and MainWindow (so
    its self._control has the legacy alias keys baked in — every
    bus.control_changed.emit(self._control) carries them through).
    """
    out = default_values(scope="control", include_buttons=True)
    for schema_name, vm_key in _CONTROL_KEY_ALIASES:
        if schema_name in out and vm_key not in out:
            out[vm_key] = out[schema_name]
    return out
