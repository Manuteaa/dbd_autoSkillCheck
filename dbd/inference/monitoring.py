import copy
import time
from pathlib import Path
from threading import Event
from typing import Any

from dbd.inference.AI_model import AIModel, prediction_dict
from dbd.inference.events import EventCallback, EventQueue, make_event_emitter
from dbd.utils.configurations import RunConfiguration
from dbd.utils.directkeys import SPACE, PressKey, ReleaseKey
from dbd.utils.monitoring_mss import Monitoring_mss

COMMUNITY_MODEL_PATH = Path("models/model_cpu.onnx").resolve()
COMMUNITY_CPU_THREADS = 4
UI_UPDATE_INTERVAL = 1 / 20


def real_time_monitoring(
    pred_dict: dict,
    monitor_id: int,
    emit_cb: EventCallback,
    stop_event: Any = None,
):
    if not COMMUNITY_MODEL_PATH.exists():
        emit_cb("log", f"AI model file not found: {COMMUNITY_MODEL_PATH}", "error")
        return

    monitoring_type = "MSS"

    monitoring = Monitoring_mss(monitor_id=monitor_id, crop_size=224)
    try:
        ai_model = AIModel(
            str(COMMUNITY_MODEL_PATH),
            nb_cpu_threads=COMMUNITY_CPU_THREADS,
            monitoring=monitoring,
        )
    except Exception as e:
        emit_cb("log", f"Failed to load AI model: {e}", "error")
        return

    execution_provider = ai_model.check_provider()
    execution_provider = execution_provider.replace("ExecutionProvider", "")

    # Publish startup configuration for the active interface.
    emit_cb(
        "startup",
        {
            "model_name": COMMUNITY_MODEL_PATH.name,
            "device": execution_provider,
            "threads": COMMUNITY_CPU_THREADS,
            "monitoring": monitoring_type,
            "monitor_id": monitor_id,
            "threshold": int(pred_dict[1]["threshold"] * 100),
            "frontier_threshold": int(pred_dict[2]["threshold"] * 100),
            "frontier_delay": pred_dict[2]["hit_delay"],
        },
        None,
    )

    # Variables
    t0 = time.monotonic()
    last_ui_update = 0.0
    nb_frames = 0
    try:
        emit_cb("running", None, None)

        while stop_event is None or not stop_event.is_set():
            frame_np = ai_model.grab_screenshot()
            nb_frames += 1

            pred, probs = ai_model.predict(frame_np)
            should_hit = pred_dict[pred]["hit"]
            confidence = int(probs[pred] * 100)
            now = time.monotonic()
            if now - last_ui_update >= UI_UPDATE_INTERVAL:
                emit_cb(
                    "prediction",
                    {
                        "class_id": pred,
                        "description": pred_dict[pred]["desc"],
                        "confidence": confidence,
                        "scores": list(probs),
                    },
                    None,
                )
                last_ui_update = now

            if should_hit:
                # avoid some false positive
                if probs[pred] < pred_dict[pred]["threshold"]:
                    continue

                # apply hit delay
                if pred_dict[pred]["hit_delay"] > 0:
                    time.sleep(pred_dict[pred]["hit_delay"] / 1000.0)

                PressKey(SPACE)
                time.sleep(0.005)
                ReleaseKey(SPACE)

                emit_cb(
                    "hit",
                    {"desc": pred_dict[pred]["desc"], "confidence": confidence},
                    None,
                )

                time.sleep(0.5)  # avoid hitting the same skill check multiple times
                t0 = time.monotonic()
                nb_frames = 0
                continue

            # Compute fps
            t_diff = time.monotonic() - t0
            if t_diff > 1.0:
                fps = round(nb_frames / t_diff, 1)
                emit_cb("fps", int(fps), None)

                t0 = time.monotonic()
                nb_frames = 0
    except KeyboardInterrupt:
        emit_cb("log", "Monitoring stopped by user.", "info")
    except Exception as e:
        emit_cb("log", f"Critical error during monitoring: {e}", "error")
    finally:
        emit_cb("fps", 0, None)
        ai_model.cleanup()


def run_monitoring_with_callbacks(
    *,
    configuration: RunConfiguration,
    event_queue: EventQueue,
    stop_event: Event,
) -> None:
    """Run monitoring and publish UI-safe events to a thread-safe queue."""
    pred_dict = copy.deepcopy(prediction_dict)
    for value in pred_dict.values():
        if value["hit"]:
            value["threshold"] = configuration.threshold / 100.0

    pred_dict[2]["threshold"] = configuration.frontier_threshold / 100.0
    pred_dict[2]["hit_delay"] = configuration.frontier_delay
    emit = make_event_emitter(event_queue)

    try:
        real_time_monitoring(
            pred_dict,
            monitor_id=configuration.monitor_id,
            emit_cb=emit,
            stop_event=stop_event,
        )
    except Exception as error:
        emit("log", f"Monitoring worker failed: {error}", "error")
    finally:
        emit("finished")
