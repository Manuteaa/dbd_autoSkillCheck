import os
from time import time, sleep

from gradio import (
    Dropdown, Radio, Number, Image, Label, Button, Slider,
    skip, Info, Warning, Error, Blocks, Row, Column, Markdown,
)

from dbd.AI_model import AI_model
from dbd.utils.directkeys import PressKey, ReleaseKey, SPACE
from dbd.utils.monitor import get_monitors, get_monitor_attributes, get_frame


ai_model = None
def cleanup():
    global ai_model
    if ai_model is not None:
        del ai_model
        ai_model = None
    return 0.


def monitor(ai_model_path, device, monitor_id, hit_ante, nb_cpu_threads):
    if ai_model_path is None or not os.path.exists(ai_model_path):
        raise Error("Invalid AI model file", duration=0)

    if device is None:
        raise Error("Invalid device option")

    if monitor_id is None:
        raise Error("Invalid monitor option")

    use_gpu = (device == devices[1])

    try:
        global ai_model
        ai_model = AI_model(ai_model_path, use_gpu, nb_cpu_threads, monitor_id)
        execution_provider = ai_model.check_provider()
    except Exception as e:
        raise Error("Error when loading AI model: {}".format(e), duration=0)

    if execution_provider == "CUDAExecutionProvider":
        Info("Running AI model on GPU (success, CUDA)")
    elif execution_provider == "DmlExecutionProvider":
        Info("Running AI model on GPU (success, DirectML)")
    elif execution_provider == "TensorRT":
        Info("Running AI model on GPU (success, TensorRT)")
    else:
        Info(f"Running AI model on CPU (success, {nb_cpu_threads} threads)")
        if use_gpu:
            Warning("Could not run AI model on GPU device. Check python console logs to debug.")

    # Variables
    t0 = time()
    nb_frames = 0

    try:
        while True:
            screenshot = ai_model.grab_screenshot()
            image_pil = ai_model.screenshot_to_pil(screenshot)
            image_np = ai_model.pil_to_numpy(image_pil)
            nb_frames += 1

            pred, desc, probs, should_hit = ai_model.predict(image_np)

            if should_hit:
                # ante-frontier hit delay
                if pred == 2 and hit_ante > 0:
                    sleep(hit_ante * 0.001)

                PressKey(SPACE)
                ReleaseKey(SPACE)

                yield skip(), image_pil, probs

                sleep(0.5)  # avoid hitting the same skill check multiple times
                t0 = time()
                nb_frames = 0
                continue

            # Compute fps
            t_diff = time() - t0
            if t_diff > 1.0:
                fps = round(nb_frames / t_diff, 1)
                yield fps, skip(), skip()

                t0 = time()
                nb_frames = 0

    except Exception as e:
        pass
    finally:
        print("Monitoring stopped.")


if __name__ == "__main__":
    debug_folder = "saved_images"
    models_folder = "models"

    fps_info = "Number of frames per second the AI model analyses the monitored frame. Check The GitHub FAQ for more details and requirements."
    devices = ["CPU (default)", "GPU"]
    cpu_choices = [("Very low", 1), ("Low", 2), ("Normal", 4), ("High", 8), ("Very high", 12)]

    # Find available AI models
    model_files = [(f, f'{models_folder}/{f}') for f in os.listdir(f"{models_folder}/") if f.endswith(".onnx") or f.endswith(".trt")]
    if len(model_files) == 0:
        raise Error(f"No AI model found in {models_folder}/", duration=0)

    # Monitor selection
    monitor_choices = get_monitors()
    def switch_monitor(monitor_id):
        monitor = get_monitor_attributes(monitor_id, crop_size=320)
        return get_frame(monitor)

    with (Blocks(title="Auto skill check") as webui):
        Markdown("<h1 style='text-align: center;'>DBD Auto skill check</h1>", elem_id="title")
        Markdown("https://github.com/Manuteaa/dbd_autoSkillCheck")

        with Row():
            with Column(variant="panel"):
                with Column(variant="panel"):
                    Markdown("AI inference settings")
                    ai_model_path = Dropdown(choices=model_files, value=model_files[0][1], label="Name the AI model to use (ONNX or TensorRT Engine)")
                    device = Radio(choices=devices, value=devices[0], label="Device the AI model will use")
                    monitor_id = Dropdown(choices=monitor_choices, value=monitor_choices[0][1], label="Monitor to use")
                with Column(variant="panel"):
                    Markdown("AI Features options")
                    hit_ante = Slider(minimum=0, maximum=50, step=5, value=20, label="Ante-frontier hit delay in ms")
                    cpu_stress = Radio(
                        label="CPU workload for AI model inference (increase to improve AI model FPS or decrease to reduce CPU stress)",
                        choices=cpu_choices,
                        value=cpu_choices[2][1],
                    )
                with Column():
                    run_button = Button("RUN", variant="primary")
                    stop_button = Button("STOP", variant="stop")

            with Column(variant="panel"):
                fps = Number(label="AI model FPS", info=fps_info, interactive=False)
                image_pil = Image(label="Last hit skill check frame", height=224, interactive=False)
                probs = Label(label="Skill check recognition")

        monitoring = run_button.click(
            fn=monitor, 
            inputs=[ai_model_path, device, monitor_id, hit_ante, cpu_stress],
            outputs=[fps, image_pil, probs]
        )

        stop_button.click(fn=cleanup, inputs=None, outputs=fps)
        monitor_id.select(fn=switch_monitor, inputs=monitor_id, outputs=image_pil)

    try:
        webui.launch()
    except:
        print("User stopped the web UI. Please wait to cleanup resources...")
    finally:
        cleanup()
