import json
import math
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any

from dearpygui import dearpygui

from dbd.inference.AI_model import DEFAULT_FRONTIER_HIT_DELAY, DEFAULT_FRONTIER_THRESHOLD, DEFAULT_THRESHOLD, prediction_dict
from dbd.inference.events import EventQueue, WorkerEvent
from dbd.inference.monitoring import COMMUNITY_MODEL_PATH, run_monitoring_with_callbacks
from dbd.utils.configurations import RunConfiguration
from dbd.utils.monitoring_mss import Monitoring_mss

COMMUNITY_CAPTURE_NAME = "MSS"
DEFAULT_VIEWPORT_WIDTH = 1400
DEFAULT_VIEWPORT_HEIGHT = 950
UI_FONT_SIZE = 16
HEADING_FONT_SIZE = 23
FONTS_FOLDER = Path("fonts").resolve()
INFERENCE_SCORE_GROUPS = (
    ("none", "None", (0,)),
    ("repair_heal", "Repair-heal", (1, 2, 3)),
    ("full_white", "Full white", (4, 5)),
    ("full_black", "Full black", (6, 7)),
    ("wiggle", "Wiggle", (8, 9)),
)


@dataclass(frozen=True, slots=True)
class AppSettings:
    """Optional UI presets loaded before a run is configured."""

    monitor_id: int | None = None
    thresh: int = DEFAULT_THRESHOLD
    thresh_frontier: int = DEFAULT_FRONTIER_THRESHOLD
    delay_frontier: int = DEFAULT_FRONTIER_HIT_DELAY


def _safe_score(score: Any) -> float:
    """Return a finite model score clamped to the progress bar range."""
    try:
        score_value = float(score)
    except (TypeError, ValueError):
        return 0.0
    return min(max(score_value, 0.0), 1.0) if math.isfinite(score_value) else 0.0


def _score_at(scores: Any, class_id: int) -> float:
    try:
        return _safe_score(scores[class_id])
    except (IndexError, TypeError):
        return 0.0


def get_inference_score_values(scores: Any) -> dict[str, float]:
    """Group the ten model probabilities into the five displayed score bars."""
    return {key: round(min(sum(_score_at(scores, class_id) for class_id in class_ids), 1.0), 10) for key, _label, class_ids in INFERENCE_SCORE_GROUPS}


def get_monitor_choices() -> list[tuple[str, int]]:
    """Get monitor labels and ids from the fixed MSS capture backend."""
    try:
        return list(Monitoring_mss.get_monitors_info())
    except Exception:
        return []


def load_settings(config_path: Path = Path("settings.json")) -> AppSettings:
    """Load known settings from JSON, falling back to defaults."""
    defaults = AppSettings()

    if config_path.exists():
        try:
            with config_path.open() as f:
                loaded_settings = json.load(f)
            if isinstance(loaded_settings, dict):
                known_names = {field.name for field in fields(AppSettings)}
                known_settings = {key: value for key, value in loaded_settings.items() if key in known_names}
                return AppSettings(**known_settings)
        except (json.JSONDecodeError, OSError):
            pass

    return defaults


class AutoSkillCheckApp:
    """Dear PyGui application for configuring and displaying skill-check monitoring."""

    SETTINGS_PANEL_WIDTH = 650
    HEADER_HEIGHT = 62
    LIVE_METRICS_TABLE_HEIGHT = 300
    LIVE_METRIC_CARD_HEIGHT = 280
    CONFIGURATION_HEIGHT = 160
    SELECTOR_WIDTH = 292
    ACTION_BUTTON_WIDTH = 290
    ACTION_BUTTON_HEIGHT = 44
    MAX_EVENTS_PER_POLL = 100
    TELEMETRY_EVENT_TYPES = frozenset({"prediction", "fps"})
    ORDERED_EVENT_TYPES = frozenset({"startup", "running", "hit", "log", "finished"})

    WINDOW_TAG = "main_window"
    TITLE_TAG = "app_title"
    SETUP_TITLE_TAG = "setup_title"
    LIVE_TITLE_TAG = "live_title"
    STATUS_DOT_TAG = "status_dot"
    MONITOR_TAG = "monitor"
    THRESHOLD_TAG = "threshold"
    FRONTIER_THRESHOLD_TAG = "frontier_threshold"
    FRONTIER_DELAY_TAG = "frontier_delay"
    RUN_TAG = "run_button"
    STOP_TAG = "stop_button"
    STATUS_TAG = "status"
    STATUS_WAITING = "WAITING"
    STATUS_STARTING = "STARTING"
    STATUS_RUNNING = "RUNNING"
    STATUS_STOPPING = "STOPPING"
    STATUS_STOPPED = "STOPPED"
    DETECTION_TAG = "detection"
    INFERENCE_PANEL_TAG = "inference_panel"
    DETECTION_PANEL_TAG = "detection_panel"
    FPS_TAG = "fps"
    FPS_MARK_TAG = "fps_mark"
    FPS_BAR_TAG = "fps_bar"
    LOG_TAG = "event_log"
    SIGNAL_LOGO_TAG = "signal_logo"
    DETECTION_LOGO_TAG = "detection_logo"
    CONFIG_MODEL_TAG = "configuration_model"
    CONFIG_DEVICE_TAG = "configuration_device"
    CONFIG_CAPTURE_TAG = "configuration_capture"
    CONFIG_THRESH_TAG = "configuration_thresh"
    EMPTY_VALUE = "-"

    TEXT = (230, 242, 246, 255)
    MUTED = (145, 169, 181, 255)
    ACCENT = (55, 205, 232, 255)
    POSITIVE = (61, 220, 123, 255)
    WARNING = (247, 190, 74, 255)
    NEGATIVE = (255, 105, 125, 255)

    def __init__(
        self,
        dearpygui: Any,
        presets: AppSettings | None = None,
    ) -> None:
        self.dpg = dearpygui
        self.presets = presets or AppSettings()
        self.heading_font: Any = None
        self.progress_themes: dict[tuple[int, int, int, int], Any] = {}
        self.worker: Thread | None = None
        self.event_queue: EventQueue | None = None
        self.stop_event: Event | None = None
        self.worker_finished = False

    def _preset(self, name: str, default: Any = None) -> Any:
        value = getattr(self.presets, name, None)
        return default if value is None else value

    def _build_fonts(self, fonts_folder: Path = FONTS_FOLDER) -> None:
        """Load the bundled UI font and its bold heading face when available."""
        regular_path = fonts_folder / "CascadiaCode" / "CaskaydiaCoveNerdFontMono-Regular.ttf"
        heading_path = fonts_folder / "CascadiaCode" / "CaskaydiaCoveNerdFontMono-Bold.ttf"
        if not regular_path.is_file():
            return

        dpg = self.dpg
        with dpg.font_registry():
            ui_font = dpg.add_font(str(regular_path), UI_FONT_SIZE)
            if heading_path.is_file():
                self.heading_font = dpg.add_font(str(heading_path), HEADING_FONT_SIZE)
        dpg.bind_font(ui_font)

    def _build_theme(self) -> None:
        dpg = self.dpg
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvAll):
                for color, value in (
                    (dpg.mvThemeCol_WindowBg, (8, 13, 19, 255)),
                    (dpg.mvThemeCol_ChildBg, (14, 24, 33, 255)),
                    (dpg.mvThemeCol_PopupBg, (18, 31, 42, 255)),
                    (dpg.mvThemeCol_FrameBg, (22, 39, 52, 255)),
                    (dpg.mvThemeCol_FrameBgHovered, (29, 56, 70, 255)),
                    (dpg.mvThemeCol_FrameBgActive, (35, 76, 92, 255)),
                    (dpg.mvThemeCol_Text, self.TEXT),
                    (dpg.mvThemeCol_TextDisabled, self.MUTED),
                    (dpg.mvThemeCol_Border, (42, 101, 122, 255)),
                    (dpg.mvThemeCol_Separator, (35, 75, 89, 255)),
                    (dpg.mvThemeCol_Header, (23, 53, 67, 255)),
                    (dpg.mvThemeCol_HeaderHovered, (30, 72, 88, 255)),
                    (dpg.mvThemeCol_HeaderActive, (36, 91, 108, 255)),
                    (dpg.mvThemeCol_CheckMark, self.ACCENT),
                    (dpg.mvThemeCol_SliderGrab, self.ACCENT),
                    (dpg.mvThemeCol_SliderGrabActive, (104, 229, 247, 255)),
                    (dpg.mvThemeCol_ScrollbarBg, (10, 20, 28, 255)),
                    (dpg.mvThemeCol_ScrollbarGrab, (55, 84, 98, 255)),
                    (dpg.mvThemeCol_ScrollbarGrabHovered, (74, 119, 135, 255)),
                    (dpg.mvThemeCol_ScrollbarGrabActive, self.ACCENT),
                ):
                    dpg.add_theme_color(color, value)
                for style, values in (
                    (dpg.mvStyleVar_WindowRounding, (9,)),
                    (dpg.mvStyleVar_ChildRounding, (7,)),
                    (dpg.mvStyleVar_FrameRounding, (5,)),
                    (dpg.mvStyleVar_GrabRounding, (5,)),
                    (dpg.mvStyleVar_WindowPadding, (14, 12)),
                    (dpg.mvStyleVar_FramePadding, (8, 5)),
                    (dpg.mvStyleVar_ItemSpacing, (8, 7)),
                    (dpg.mvStyleVar_ItemInnerSpacing, (7, 4)),
                    (dpg.mvStyleVar_WindowBorderSize, (0,)),
                    (dpg.mvStyleVar_ChildBorderSize, (1,)),
                    (dpg.mvStyleVar_FrameBorderSize, (0,)),
                ):
                    dpg.add_theme_style(style, *values)
            with dpg.theme_component(dpg.mvButton):
                for color, value in (
                    (dpg.mvThemeCol_Button, (20, 117, 143, 255)),
                    (dpg.mvThemeCol_ButtonHovered, (29, 157, 183, 255)),
                    (dpg.mvThemeCol_ButtonActive, (17, 91, 113, 255)),
                ):
                    dpg.add_theme_color(color, value)
            with dpg.theme_component(dpg.mvInputText):
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (12, 22, 30, 255))
        dpg.bind_theme(theme)
        self.progress_themes = {}
        for color in (self.NEGATIVE, (255, 153, 102, 255), self.WARNING, self.POSITIVE, self.ACCENT, self.MUTED):
            with dpg.theme() as progress_theme, dpg.theme_component(dpg.mvProgressBar):
                dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, color)
            self.progress_themes[color] = progress_theme

    def _set_status(self, status: str) -> None:
        status_color = {
            self.STATUS_WAITING: self.WARNING,
            self.STATUS_STARTING: self.ACCENT,
            self.STATUS_RUNNING: self.POSITIVE,
            self.STATUS_STOPPING: self.WARNING,
            self.STATUS_STOPPED: self.NEGATIVE,
        }.get(status, self.MUTED)
        self.dpg.set_value(self.STATUS_TAG, status)
        self.dpg.configure_item(self.STATUS_TAG, color=status_color)
        self.dpg.configure_item(self.STATUS_DOT_TAG, color=status_color)
        self._update_buttons()

    def _set_monitor_choices(self, preferred_monitor_id: int | None = None) -> None:
        choices = get_monitor_choices()
        monitor_ids = dict(choices)
        labels = list(monitor_ids)
        self.dpg.configure_item(self.MONITOR_TAG, items=labels, enabled=bool(labels), user_data=monitor_ids)
        if labels:
            selected_label = next(
                (label for label, monitor_id in choices if monitor_id == preferred_monitor_id),
                labels[0],
            )
            self.dpg.set_value(self.MONITOR_TAG, selected_label)
            self._set_status(self.STATUS_WAITING)
            if preferred_monitor_id is not None and monitor_ids[selected_label] != preferred_monitor_id:
                self._append_log(f"Preset monitor {preferred_monitor_id} was not found; using {selected_label}.", "warning")
        else:
            self.dpg.set_value(self.MONITOR_TAG, "No monitors detected")
            self._set_status(self.STATUS_WAITING)

    def _scroll_log_to_bottom(self, *_args: Any) -> None:
        set_y_scroll = getattr(self.dpg, "set_y_scroll", None)
        if callable(set_y_scroll):
            set_y_scroll(self.LOG_TAG, -1)

    def _append_log(self, message: str, level: str = "info") -> None:
        icons = {"success": "✓", "warning": "!", "error": "×", "info": "•"}
        colors = {"success": self.POSITIVE, "warning": self.WARNING, "error": self.NEGATIVE, "info": self.MUTED}
        level = level if level in colors else "info"
        line = f"{icons[level]}  {message}"

        does_item_exist = getattr(self.dpg, "does_item_exist", None)
        add_text = getattr(self.dpg, "add_text", None)
        if callable(does_item_exist) and does_item_exist(self.LOG_TAG) and callable(add_text):
            add_text(
                line,
                parent=self.LOG_TAG,
                color=colors[level],
            )
            self._scroll_log_to_bottom()
        else:
            self.dpg.set_value(self.LOG_TAG, line)

    def _update_buttons(self) -> None:
        status = self.dpg.get_value(self.STATUS_TAG)
        self.dpg.configure_item(
            self.RUN_TAG,
            enabled=status in {self.STATUS_WAITING, self.STATUS_STOPPED},
        )
        self.dpg.configure_item(
            self.STOP_TAG,
            enabled=status in {self.STATUS_STARTING, self.STATUS_RUNNING},
        )

    def _start_worker(
        self,
        target: Callable[..., None],
        worker_kwargs: dict[str, Any],
        started_message: str,
    ) -> bool:
        """Start one background worker with fresh synchronization objects."""
        self._reap_worker()
        if self.worker is not None:
            if self.worker_finished:
                self._append_log("Previous monitoring worker is still shutting down; please retry.", "info")
                self._set_status(self.STATUS_STOPPED)
                return False
            self._append_log("Monitoring is already running.", "error")
            if self.worker.is_alive():
                self._set_status(self.STATUS_RUNNING)
            return False

        event_queue: EventQueue = Queue()
        stop_event = Event()
        worker = Thread(
            target=target,
            kwargs={
                **worker_kwargs,
                "event_queue": event_queue,
                "stop_event": stop_event,
            },
            daemon=True,
        )

        self.worker_finished = False
        self._set_status(self.STATUS_STARTING)
        try:
            worker.start()
        except RuntimeError as error:
            self._append_log(f"Failed to start background worker: {error}", "error")
            self._set_status(self.STATUS_WAITING)
            return False

        self.event_queue = event_queue
        self.stop_event = stop_event
        self.worker = worker
        self._append_log(started_message, "success")
        return True

    def _request_worker_stop(self) -> bool:
        """Signal the active worker without blocking the render thread."""
        if self.worker is None or not self.worker.is_alive() or self.stop_event is None:
            return False
        self.stop_event.set()
        return True

    def _reap_worker(self) -> None:
        """Forget a completed worker after all of its events are consumed."""
        if self.worker is None or self.worker.is_alive():
            return
        if self.event_queue is not None and not self.event_queue.empty():
            return
        stopped_unexpectedly = not self.worker_finished
        self.worker = None
        self.event_queue = None
        self.stop_event = None
        if stopped_unexpectedly:
            self.worker_finished = True
            self._set_status(self.STATUS_STOPPED)
            self._append_log("Monitoring worker stopped unexpectedly.", "error")

    def _shutdown_worker(self) -> None:
        """Stop and briefly join the worker while the application closes."""
        if self.worker is None:
            return
        if self.worker.is_alive() and self.stop_event is not None:
            self.stop_event.set()
        self.worker.join(timeout=5)
        if not self.worker.is_alive():
            self.worker = None
            self.event_queue = None
            self.stop_event = None

    def _read_run_configuration(self) -> RunConfiguration | None:
        """Validate the selected controls and return one run configuration."""
        if not COMMUNITY_MODEL_PATH.is_file():
            self._append_log(f"Community AI model not found: {COMMUNITY_MODEL_PATH}", "error")
            self._set_status(self.STATUS_WAITING)
            return None

        monitor_label = self.dpg.get_value(self.MONITOR_TAG)
        monitor_ids = self.dpg.get_item_user_data(self.MONITOR_TAG) or {}
        monitor_id = monitor_ids.get(monitor_label)
        if monitor_id is None:
            self._append_log("Select an available monitor before starting.", "error")
            self._set_status(self.STATUS_WAITING)
            return None

        return RunConfiguration(
            monitor_id=monitor_id,
            threshold=int(self.dpg.get_value(self.THRESHOLD_TAG)),
            frontier_threshold=int(self.dpg.get_value(self.FRONTIER_THRESHOLD_TAG)),
            frontier_delay=int(self.dpg.get_value(self.FRONTIER_DELAY_TAG)),
        )

    def _run(self, *_args: Any) -> None:
        if self.dpg.get_value(self.STATUS_TAG) not in {self.STATUS_WAITING, self.STATUS_STOPPED}:
            return

        configuration = self._read_run_configuration()
        if configuration is None:
            return

        self._start_worker(
            run_monitoring_with_callbacks,
            {"configuration": configuration},
            "Monitoring started.",
        )

    def _stop(self, *_args: Any) -> None:
        status = self.dpg.get_value(self.STATUS_TAG)
        if status == self.STATUS_STOPPING:
            return
        if status not in {self.STATUS_STARTING, self.STATUS_RUNNING}:
            return

        self._set_status(self.STATUS_STOPPING)
        if self._request_worker_stop():
            self._append_log("Monitoring stop requested.", "info")
        else:
            self._append_log("Monitoring worker has already stopped; waiting for final events.", "info")

    def _update_fps_display(self, fps: Any) -> None:
        try:
            fps_value = float(fps)
        except (TypeError, ValueError):
            fps_value = 0.0
        if not math.isfinite(fps_value) or fps_value < 0:
            fps_value = 0.0

        fps_color = next(
            (
                color
                for limit, color in (
                    (40, self.NEGATIVE),
                    (60, (255, 153, 102, 255)),
                    (80, self.WARNING),
                    (110, self.POSITIVE),
                )
                if fps_value < limit
            ),
            self.ACCENT,
        )
        self.dpg.set_value(self.FPS_TAG, f"{fps_value:.0f} FPS")
        self.dpg.set_value(self.FPS_BAR_TAG, min(fps_value, 120) / 120)
        self.dpg.configure_item(self.FPS_BAR_TAG, overlay=f"{fps_value:.0f} FPS")
        self.dpg.configure_item(self.FPS_TAG, color=fps_color)
        self.dpg.configure_item(self.FPS_MARK_TAG, color=fps_color)

        fps_theme = self.progress_themes.get(fps_color)
        bind_item_theme = getattr(self.dpg, "bind_item_theme", None)
        if fps_theme is not None and callable(bind_item_theme):
            bind_item_theme(self.FPS_BAR_TAG, fps_theme)

    def _prediction_color(self, description: Any) -> tuple[int, int, int, int]:
        description = str(description).lower()
        if description == "none":
            return self.MUTED
        if "out" in description:
            return self.NEGATIVE
        if "great" in description or "frontier" in description:
            return self.POSITIVE
        return self.MUTED

    def _update_inference_score_display(self, score_data: Any) -> None:
        scores = score_data.get("scores") if isinstance(score_data, dict) else None
        values = get_inference_score_values(scores) if scores is not None else {key: 0.0 for key, _label, _class_ids in INFERENCE_SCORE_GROUPS}

        bind_item_theme = getattr(self.dpg, "bind_item_theme", None)
        for key, _label, class_ids in INFERENCE_SCORE_GROUPS:
            value = values[key]
            color = self.MUTED
            if scores is not None:
                best_class_id = max(class_ids, key=lambda candidate: _score_at(scores, candidate))
                color = self._prediction_color(prediction_dict[best_class_id]["desc"])
            bar_tag = f"inference_score_{key}"
            self.dpg.set_value(bar_tag, value)
            self.dpg.configure_item(bar_tag, overlay=f"{value:.0%}")
            self.dpg.configure_item(f"inference_score_label_{key}", color=color)
            score_theme = self.progress_themes.get(color)
            if score_theme is not None and callable(bind_item_theme):
                bind_item_theme(bar_tag, score_theme)

    def _handle_startup(self, event: WorkerEvent) -> None:
        config = event.get("data")
        if not isinstance(config, dict):
            return
        if self.dpg.get_value(self.STATUS_TAG) not in {self.STATUS_RUNNING, self.STATUS_STOPPING, self.STATUS_STOPPED}:
            self._set_status(self.STATUS_STARTING)

        device = config.get("device", self.EMPTY_VALUE)
        if config.get("threads") is not None:
            device = f"{device} · {config['threads']} threads"
        monitoring = str(config.get("monitoring", self.EMPTY_VALUE)).upper()
        capture = f"{monitoring} · Monitor {config.get('monitor_id', self.EMPTY_VALUE)}"
        thresholds = (
            f"Default: {config.get('threshold', self.EMPTY_VALUE)}% · "
            f"Gens: {config.get('frontier_threshold', self.EMPTY_VALUE)}% + "
            f"{config.get('frontier_delay', self.EMPTY_VALUE)} ms"
        )
        configuration = {
            self.CONFIG_MODEL_TAG: str(config.get("model_name", self.EMPTY_VALUE)),
            self.CONFIG_DEVICE_TAG: str(device),
            self.CONFIG_CAPTURE_TAG: capture,
            self.CONFIG_THRESH_TAG: thresholds,
        }
        for tag, value in configuration.items():
            self.dpg.set_value(tag, value)
            self.dpg.configure_item(tag, color=self.TEXT)

    def _handle_prediction(self, event: WorkerEvent) -> None:
        data = event.get("data")
        if not isinstance(data, dict):
            return
        scores = data.get("scores")
        if not isinstance(scores, (list, tuple)) or len(scores) < 10:
            return
        prediction = str(data.get("description", data.get("desc", self.EMPTY_VALUE)))
        self.dpg.set_value(self.DETECTION_TAG, prediction)
        self.dpg.configure_item(self.DETECTION_TAG, color=self._prediction_color(prediction))
        self._update_inference_score_display(data)
        self._set_status(self.STATUS_RUNNING)

    def _handle_hit(self, event: WorkerEvent) -> None:
        hit = event.get("data")
        if not isinstance(hit, dict):
            return
        desc = str(hit.get("desc", "Skill check"))
        confidence = hit.get("confidence", 0)
        self._append_log(f"{desc.upper()} hit at {confidence}% confidence", "info")

    def _handle_finished(self, _event: WorkerEvent) -> None:
        was_stopping = self.dpg.get_value(self.STATUS_TAG) == self.STATUS_STOPPING
        self.worker_finished = True
        self._set_status(self.STATUS_STOPPED)
        if was_stopping:
            self._append_log("Monitoring stopped.", "success")

    def _dispatch_event(self, event: WorkerEvent) -> None:
        event_type = event.get("type")
        if not isinstance(event_type, str):
            return
        if self.worker_finished and event_type in self.TELEMETRY_EVENT_TYPES:
            return
        if event_type == "startup":
            self._handle_startup(event)
        elif event_type == "running":
            self._set_status(self.STATUS_RUNNING)
        elif event_type == "prediction":
            self._handle_prediction(event)
        elif event_type == "fps":
            self._update_fps_display(event.get("data"))
        elif event_type == "hit":
            self._handle_hit(event)
        elif event_type == "finished":
            self._handle_finished(event)
        elif event_type == "log":
            self._append_log(str(event.get("data")), str(event.get("level", "info")))

    def _drain_events(self) -> list[WorkerEvent]:
        """Read and coalesce a bounded batch of worker events."""
        if self.event_queue is None:
            return []

        drained: list[WorkerEvent] = []
        pending_telemetry: dict[str, WorkerEvent] = {}

        def flush_telemetry() -> None:
            for event_type in ("prediction", "fps"):
                event = pending_telemetry.pop(event_type, None)
                if event is not None:
                    drained.append(event)

        for _event_count in range(self.MAX_EVENTS_PER_POLL):
            try:
                event = self.event_queue.get_nowait()
            except (Empty, EOFError, OSError):
                break
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            if not isinstance(event_type, str):
                continue
            if event_type in self.TELEMETRY_EVENT_TYPES:
                pending_telemetry[event_type] = event
            elif event_type in self.ORDERED_EVENT_TYPES:
                flush_telemetry()
                drained.append(event)

        flush_telemetry()
        return drained

    def _poll_events(self, *_args: Any) -> None:
        for event in self._drain_events():
            self._dispatch_event(event)
        self._reap_worker()

    def _add_help_marker(self, message: str) -> None:
        dpg = self.dpg
        help_button = dpg.add_button(label="?", small=True)
        with dpg.tooltip(help_button):
            dpg.add_text(message, wrap=420)

    def _add_labeled_slider(
        self,
        label: str,
        limits: tuple[int, int],
        default: int,
        tag: str,
        help_text: str,
    ) -> None:
        minimum, maximum = limits
        with self.dpg.group(horizontal=True):
            self.dpg.add_text(label, color=self.TEXT)
            self._add_help_marker(help_text)
        self.dpg.add_slider_int(
            min_value=minimum,
            max_value=maximum,
            default_value=default,
            tag=tag,
            width=-1,
        )

    def _build_header(self) -> None:
        dpg = self.dpg
        with dpg.child_window(width=-1, height=self.HEADER_HEIGHT, border=True), dpg.group(horizontal=True):
            dpg.add_text("DBD AUTO SKILL CHECK — COMMUNITY", tag=self.TITLE_TAG, color=self.ACCENT)
            dpg.add_spacer(width=-1)
            dpg.add_text(
                "Reuse or redistribution: credit https://github.com/Manuteaa/dbd_autoSkillCheck.\n"
                "Follow GPLv3: include the license and provide corresponding source.",
                color=self.TEXT,
            )

    def _build_capture_settings(self) -> None:
        dpg = self.dpg
        dpg.add_spacer(height=6)
        dpg.add_separator()
        dpg.add_text("SCREEN CAPTURE", color=self.ACCENT)
        with dpg.group(horizontal=True):
            with dpg.group():
                with dpg.group(horizontal=True):
                    dpg.add_text("SOURCE", color=self.MUTED)
                    self._add_help_marker(
                        "The community edition uses MSS to capture a 224x224 crop from the center of the selected display."
                        "\n\nOn Linux, launch the application from an X11/XWayland session where DISPLAY is set.",
                    )
                dpg.add_text(COMMUNITY_CAPTURE_NAME, color=self.TEXT)
            dpg.add_spacer(width=10)
            with dpg.group():
                dpg.add_text("DISPLAY", color=self.MUTED)
                dpg.add_combo(items=[], tag=self.MONITOR_TAG, width=self.SELECTOR_WIDTH)

    def _build_additional_ai_settings(self) -> None:
        dpg = self.dpg
        dpg.add_spacer(height=6)
        dpg.add_separator()
        with dpg.collapsing_header(label="Additional AI Settings", default_open=False):
            for label, minimum, maximum, default, tag, help_text in (
                (
                    "Default SC confidence threshold (%)",
                    51,
                    100,
                    self._preset("thresh", DEFAULT_THRESHOLD),
                    self.THRESHOLD_TAG,
                    "Minimum confidence required before automatically hitting skill checks other than repair/heal. "
                    "Increase the value if the AI activates too early.",
                ),
                (
                    "Repair-heal SC confidence threshold (%)",
                    51,
                    100,
                    self._preset("thresh_frontier", DEFAULT_FRONTIER_THRESHOLD),
                    self.FRONTIER_THRESHOLD_TAG,
                    "Minimum confidence required before automatically hitting repair/heal skill checks. Increase the value if the AI activates too early.",
                ),
                (
                    "Repair-heal SC hit delay (ms)",
                    0,
                    50,
                    self._preset("delay_frontier", DEFAULT_FRONTIER_HIT_DELAY),
                    self.FRONTIER_DELAY_TAG,
                    "Forced wait time before hitting SPACE (repair/heal skill checks only). Increase the value if the AI hits too early.",
                ),
            ):
                self._add_labeled_slider(label, (minimum, maximum), default, tag, help_text)

    def _build_action_buttons(self) -> None:
        dpg = self.dpg
        dpg.add_spacer(height=8)
        dpg.add_separator()
        with dpg.group(horizontal=True):
            for index, (label, tag, callback, enabled) in enumerate(
                (
                    ("▶  START MONITORING", self.RUN_TAG, self._run, True),
                    ("■  STOP MONITORING", self.STOP_TAG, self._stop, False),
                ),
            ):
                if index:
                    dpg.add_spacer(width=12)
                dpg.add_button(
                    label=label,
                    tag=tag,
                    callback=callback,
                    width=self.ACTION_BUTTON_WIDTH,
                    height=self.ACTION_BUTTON_HEIGHT,
                    enabled=enabled,
                )

    def _build_settings_panel(self) -> None:
        dpg = self.dpg
        with dpg.child_window(
            width=self.SETTINGS_PANEL_WIDTH,
            height=-1,
            border=True,
            horizontal_scrollbar=False,
        ):
            with dpg.group(horizontal=True):
                dpg.add_text("SETTINGS", tag=self.SETUP_TITLE_TAG, color=self.ACCENT)
                self._add_help_marker(
                    "Default values are loaded from settings.json when the application starts.\n"
                    "Edit that file before launch to keep your preferred settings. "
                )
            dpg.add_text(
                "This is the limited community version of DBD Auto Skill Check.",
                color=self.WARNING,
                wrap=600,
            )
            dpg.add_text(
                "The complete version is available through the Discord server after accepting the fair-use agreement.",
                color=self.TEXT,
                wrap=600,
            )
            dpg.add_text("https://discord.gg/3mewehHHpZ", color=self.ACCENT)
            dpg.add_spacer(height=8)
            self._build_capture_settings()
            self._build_additional_ai_settings()
            self._build_action_buttons()

    def _build_inference_card(self) -> None:
        dpg = self.dpg
        with dpg.child_window(
            tag=self.INFERENCE_PANEL_TAG,
            width=-1,
            height=self.LIVE_METRIC_CARD_HEIGHT,
            border=True,
            horizontal_scrollbar=False,
        ):
            dpg.add_text("AI INFERENCE PERFORMANCE", tag=self.SIGNAL_LOGO_TAG, color=self.ACCENT)
            dpg.add_spacer(height=8)
            with dpg.group(horizontal=True):
                dpg.add_text("●", tag=self.STATUS_DOT_TAG, color=self.WARNING)
                dpg.add_text("MONITOR STATUS", color=self.MUTED)
            dpg.add_text(self.STATUS_WAITING, tag=self.STATUS_TAG, color=self.WARNING)
            dpg.add_spacer(height=14)
            with dpg.group(horizontal=True):
                dpg.add_text("↯", tag=self.FPS_MARK_TAG, color=self.ACCENT)
                dpg.add_text("CURRENT INFERENCE FPS", color=self.MUTED)
            dpg.add_text("0 FPS", tag=self.FPS_TAG, color=self.ACCENT)
            dpg.add_progress_bar(default_value=0.0, tag=self.FPS_BAR_TAG, width=-1, height=20, overlay="0 FPS")

    def _build_detection_card(self) -> None:
        dpg = self.dpg
        with dpg.child_window(
            tag=self.DETECTION_PANEL_TAG,
            width=-1,
            height=self.LIVE_METRIC_CARD_HEIGHT,
            border=True,
            horizontal_scrollbar=False,
        ):
            with dpg.group(horizontal=True):
                dpg.add_text("AI DETECTION", tag=self.DETECTION_LOGO_TAG, color=self.ACCENT)
                self._add_help_marker(
                    "For best results, run the game and AI at 120 FPS or higher and set your monitor to at least 120 Hz.\n\n"
                    "Fine-tune the AI settings if needed. If you have trouble, visit the Discord server for help.",
                )
            dpg.add_spacer(height=8)
            dpg.add_text(self.EMPTY_VALUE, tag=self.DETECTION_TAG, color=self.MUTED, wrap=420)
            with dpg.table(
                header_row=False,
                policy=dpg.mvTable_SizingStretchProp,
                width=-1,
                inner_width=8,
                borders_innerH=False,
                borders_outerH=False,
                borders_innerV=False,
                borders_outerV=False,
            ):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=120)
                dpg.add_table_column()
                for key, label, _class_ids in INFERENCE_SCORE_GROUPS:
                    with dpg.table_row():
                        dpg.add_text(label, tag=f"inference_score_label_{key}", color=self.MUTED)
                        dpg.add_progress_bar(
                            default_value=0.0,
                            tag=f"inference_score_{key}",
                            width=-1,
                            height=16,
                            overlay="0%",
                        )

    def _build_configuration_card(self) -> None:
        dpg = self.dpg
        dpg.add_text("ACTIVE CONFIGURATION", color=self.ACCENT)
        with (
            dpg.child_window(height=self.CONFIGURATION_HEIGHT, border=True, horizontal_scrollbar=False),
            dpg.table(
                header_row=False,
                policy=dpg.mvTable_SizingStretchSame,
                width=-1,
                inner_width=8,
                borders_innerH=False,
                borders_outerH=False,
                borders_innerV=False,
                borders_outerV=False,
            ),
        ):
            dpg.add_table_column()
            dpg.add_table_column()
            for row in (
                (("◈ MODEL", self.CONFIG_MODEL_TAG), ("⚙ DEVICE", self.CONFIG_DEVICE_TAG)),
                (("▣ CAPTURE", self.CONFIG_CAPTURE_TAG), ("◌ AI THRESHOLDS", self.CONFIG_THRESH_TAG)),
            ):
                with dpg.table_row():
                    for title, tag in row:
                        with dpg.group():
                            dpg.add_text(title, color=self.ACCENT)
                            dpg.add_text(self.EMPTY_VALUE, tag=tag, color=self.MUTED, wrap=290)

    def _build_activity_log(self) -> None:
        self.dpg.add_text("RECENT ACTIVITIES", color=self.MUTED)
        with self.dpg.child_window(tag=self.LOG_TAG, height=-1, border=False, horizontal_scrollbar=False):
            pass

    def _build_live_panel(self) -> None:
        dpg = self.dpg
        with dpg.child_window(width=-1, height=-1, border=True, horizontal_scrollbar=False):
            dpg.add_text("LIVE MONITOR", tag=self.LIVE_TITLE_TAG, color=self.ACCENT)
            with dpg.table(
                header_row=False,
                policy=dpg.mvTable_SizingStretchSame,
                width=-1,
                height=self.LIVE_METRICS_TABLE_HEIGHT,
                inner_width=8,
                borders_innerH=False,
                borders_outerH=False,
                borders_innerV=False,
                borders_outerV=False,
            ):
                dpg.add_table_column()
                dpg.add_table_column()
                with dpg.table_row():
                    self._build_inference_card()
                    self._build_detection_card()
            dpg.add_spacer(height=10)
            self._build_configuration_card()
            dpg.add_spacer(height=6)
            self._build_activity_log()

    def _bind_heading_fonts(self) -> None:
        if self.heading_font is None:
            return
        does_item_exist = getattr(self.dpg, "does_item_exist", None)
        for tag in (
            self.TITLE_TAG,
            self.SETUP_TITLE_TAG,
            self.LIVE_TITLE_TAG,
            self.SIGNAL_LOGO_TAG,
            self.DETECTION_LOGO_TAG,
            self.STATUS_TAG,
            self.FPS_TAG,
            self.DETECTION_TAG,
        ):
            if callable(does_item_exist) and not does_item_exist(tag):
                continue
            self.dpg.bind_item_font(tag, self.heading_font)

    def _build_window(self) -> None:
        with self.dpg.window(tag=self.WINDOW_TAG, label="DBD ASC", no_collapse=True):
            self._build_header()
            self.dpg.add_separator()
            with self.dpg.group(horizontal=True):
                self._build_settings_panel()
                self._build_live_panel()

        self._bind_heading_fonts()
        self._set_monitor_choices(preferred_monitor_id=self._preset("monitor_id"))
        self._update_fps_display(0)
        self._update_inference_score_display(None)

    def run(self, *_args: Any) -> None:
        dpg = self.dpg
        dpg.create_context()
        try:
            self._build_fonts()
            self._build_theme()
            self._build_window()
            dpg.create_viewport(
                title="DBD ASC COMMUNITY",
                width=DEFAULT_VIEWPORT_WIDTH,
                height=DEFAULT_VIEWPORT_HEIGHT,
                resizable=True,
            )
            dpg.setup_dearpygui()
            dpg.show_viewport()
            dpg.set_primary_window(self.WINDOW_TAG, True)
            while dpg.is_dearpygui_running():
                self._poll_events()
                dpg.render_dearpygui_frame()
        finally:
            self._shutdown_worker()
            dpg.destroy_context()


def main() -> None:
    """Start the Dear PyGui desktop application."""
    presets = load_settings()
    AutoSkillCheckApp(dearpygui, presets=presets).run()


if __name__ == "__main__":
    main()
