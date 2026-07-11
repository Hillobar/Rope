import os
import sys
from pathlib import Path

# `import rope` triggers rope/__init__.py → rope/_native_dlls.py, which
# (a) registers TRT runtime DLL directories and (b) sets
# ORT_LOG_SEVERITY_LEVEL=3 BEFORE ORT is imported anywhere. We keep the
# belt-and-braces set_default_logger_severity(3) call below for the
# case where ORT was imported by some other side-effect import first.

from PySide6.QtWidgets import QApplication

from rope.qt.bus import bus
from rope.qt.coordinator import Coordinator
from rope.qt.main_window import MainWindow


def _load_stylesheet(app: QApplication) -> None:
    qss_path = Path(__file__).with_name("rope.qss")
    if qss_path.is_file():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))


def run(skip_backend: bool = False) -> int:
    """Launch the Qt version of Rope-Pearl.

    skip_backend=True instantiates only the window/Bus/Coordinator skeleton
    without loading Models/VideoManager. Useful for Phase A smoke testing
    where loading ONNX models is slow and unnecessary.
    """
    app = QApplication.instance() or QApplication(sys.argv)
    _load_stylesheet(app)

    if skip_backend:
        models, vm = None, None
    else:
        # Belt-and-braces — env var handles fresh ORT init, this handles
        # the case where ORT was loaded earlier by another import.
        try:
            import onnxruntime
            onnxruntime.set_default_logger_severity(3)
        except (ImportError, AttributeError):
            pass
        import rope.Models as Models
        import rope.VideoManager as VM
        models = Models.Models()
        vm = VM.VideoManager(models)

    window = MainWindow()
    # Apply the persisted Settings.models_folder before any model is
    # lazily loaded. set_models_folder is a no-op when the path matches
    # the default, so first-run startup pays nothing.
    saved_folder = getattr(window.settings, "models_folder", None)
    if saved_folder:
        models.set_models_folder(saved_folder)
    coordinator = Coordinator(models=models, vm=vm)

    # Phase A smoke probe: prove signals connect end-to-end. Removed once
    # widgets land in Phase B/C and emit real signals.
    bus.frame_ready.connect(lambda _frame, requested: window.statusBar().showMessage(
        f"frame_ready (requested={requested})", 2000
    ))
    bus.stop_play.connect(lambda: window.statusBar().showMessage("stop_play", 2000))
    bus.slider_length_changed.connect(
        lambda n: window.statusBar().showMessage(f"slider_length_changed={n}", 2000)
    )

    window.show()
    # Keep a reference so the coordinator isn't GC'd
    window._coordinator = coordinator
    # Hand the Models instance to the Settings tab so it can render the
    # Loaded column. Skipped under skip_backend (smoke test) since
    # models is None there.
    if models is not None and hasattr(window, "_params_pane"):
        # Apply any persisted per-model backend overrides before the
        # first lazy load. unload=False because nothing is loaded yet,
        # and we don't want a stray vram_updated emit on startup.
        prefs = getattr(window.settings, "model_backends", None) or {}
        for attr_name, backend in prefs.items():
            try:
                models.set_backend_preference(attr_name, backend, unload=False)
            except Exception:
                pass
        window._params_pane.set_models_instance(models)
        # Re-apply the loaded saved_parameters now that `models` is
        # reachable via window._coordinator. _load_saved_parameters ran
        # during MainWindow.__init__ — before the coordinator was
        # assigned — so its emit's _on_params_changed call saw
        # _get_models() return None and skipped the Models-side
        # propagation (set_model_session_mode, update_load_expectation).
        # Re-calling here ensures ModelSessionsTextSel and similar
        # Models-affecting params from saved_parameters.json take effect.
        try:
            window._on_params_changed(dict(window._params_pane.values))
        except Exception as e:
            print(f'[app] late re-apply of saved params failed: {e}')
        # Model preload is no longer automatic at startup — it now happens
        # only when the user clicks the "Preload Models" button (left of
        # Enable Audio in the toggle row; MainWindow._on_preload_models),
        # so the app starts instantly and engines build against whatever
        # backend / thread count the user has actually selected.

    # Show the startup splash in the preview. Staged last (after the
    # late saved-params re-apply above) so any startup frame request
    # can't clobber it; the first real media frame replaces it later.
    window.show_splash()

    return app.exec()
