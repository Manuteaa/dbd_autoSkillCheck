import os
from pathlib import Path
from time import time, sleep

from dbd.AI_model import AI_model
from dbd.utils.directkeys import PressKey, ReleaseKey, SPACE

from gradio import (
    Dropdown, Radio, Number, Image, Label, Button, Slider,
    skip, Info, Warning, Error, Blocks, Row, Column, Markdown
)


def monitor(onnx_ai_model, device, debug_option, hit_ante, cpu_stress):
    if onnx_ai_model is None or not os.path.exists(onnx_ai_model) or ".onnx" not in onnx_ai_model:
        raise Error("Invalid onnx file", duration=0)

    if device is None:
        raise Error("Invalid device option")

    if debug_option is None:
        raise Error("Invalid debug option")

    if cpu_stress == "min":
        nb_cpu_threads = 1
    elif cpu_stress == "low":
        nb_cpu_threads = 2
    elif cpu_stress == "normal":
        nb_cpu_threads = 4
    else:
        nb_cpu_threads = None

    # AI model
    use_gpu = (device == devices[1])
    ai_model = AI_model(onnx_ai_model, use_gpu, nb_cpu_threads)

    execution_provider = ai_model.check_provider()
    if execution_provider == "CUDAExecutionProvider":
            Info("Running AI model on GPU (success, CUDA)")
    elif execution_provider == "DmlExecutionProvider":
            Info("Running AI model on GPU (success, DirectML)")
    else:
        Info("Running AI model on CPU")
        if device == devices[1]:
            Warning("Could not run AI model on GPU device. Check python console logs to debug.")

    # Create debug folders
    if debug_option == debug_options[2] or debug_option == debug_options[3]:
        Path(debug_folder).mkdir(exist_ok=True)
        for folder_idx in range(len(ai_model.pred_dict)):
            Path(os.path.join(debug_folder, str(folder_idx))).mkdir(exist_ok=True)

    # Variables
    t0 = time()
    nb_frames = 0
    nb_hits = 0

    while True:
        screenshot = ai_model.grab_screenshot()
        image_pil = ai_model.screenshot_to_pil(screenshot)
        image_np = ai_model.pil_to_numpy(image_pil)
        nb_frames += 1

        pred, desc, probs, should_hit = ai_model.predict(image_np)

        if pred != 0 and debug_option == debug_options[3]:
            path = os.path.join(debug_folder, str(pred), "{}.png".format(nb_hits))
            image_pil.save(path)
            nb_hits += 1

        if should_hit:

            # ante-frontier hit delay
            if pred == 2 and hit_ante > 0:
                sleep(hit_ante * 0.001)

            PressKey(SPACE)
            ReleaseKey(SPACE)

            yield skip(), image_pil, probs

            if debug_option == debug_options[2]:
                path = os.path.join(debug_folder, str(pred), "hit_{}.png".format(nb_hits))
                image_pil.save(path)
                nb_hits += 1

            sleep(0.5)  # avoid hitting the same skill check multiple times
            t0 = time()
            nb_frames = 0
            continue

        # Compute fps
        t_diff = time() - t0
        if t_diff > 1.0:
            fps = round(nb_frames / t_diff, 1)

            if debug_option == debug_options[1]:
                yield fps, image_pil, skip()
            else:
                yield fps, skip(), skip()

            t0 = time()
            nb_frames = 0

    print("HERE")


if __name__ == "__main__":
    debug_folder = "saved_images"

    debug_options = ["None (default)",
                     "Display the monitored frame (a 224x224 center-cropped image, displayed at 1fps) instead of last hit skill check frame. Useful to check the monitored screen",
                     "Save hit skill check frames in {}/".format(debug_folder),
                     "Save all skill check frames in {}/ (will impact fps)".format(debug_folder)]

    fps_info = "Number of frames per second the AI model analyses the monitored frame. Check The GitHub FAQ for more details and requirements."
    devices = ["CPU (default)", "GPU"]

    with (Blocks(title="DBD Auto skill check") as webui):
        Markdown("<h1 style='text-align: center;'>DBD Auto skill check</h1>", elem_id="title")
        Markdown("https://github.com/Manuteaa/dbd_autoSkillCheck")

        with Row():
            with Column(variant="panel"):
                with Column(variant="panel"):
                    Markdown("AI inference settings")
                    onnx_ai_model = Dropdown(choices=["model.onnx"], value="model.onnx", label="Filepath of the ONNX model (trained AI model)")
                    device = Radio(choices=devices, value=devices[0], label="Device the AI onnx model will use")
                with Column(variant="panel"):
                    Markdown("Debug options - for debugging or analytics")
                    debug_option = Dropdown(choices=debug_options, value=debug_options[0], label="Debugging selection")
                with Column(variant="panel"):
                    Markdown("Features options")
                    hit_ante = Slider(minimum=0, maximum=50, step=5, value=20, label="Ante-frontier hit delay in ms")
                    cpu_stress = Radio(label="CPU workload for AI model inference (increase to improve AI model FPS or decrease to reduce CPU stress)", choices=["min", "low", "normal", "max"], value="low")
                with Column():
                    run_button = Button("RUN", variant="primary")
                    stop_button = Button("STOP", variant="stop")

            with Column(variant="panel"):
                fps = Number(label="AI model FPS", info=fps_info, interactive=False)
                image_pil = Image(label="Last hit skill check frame", height=224, interactive=False)
                probs = Label(label="Skill check recognition")

        monitoring = run_button.click(fn=monitor, inputs=[onnx_ai_model, device, debug_option, hit_ante, cpu_stress], outputs=[fps, image_pil, probs])
        stop_button.click(fn=None, inputs=None, outputs=None, cancels=[monitoring])

    webui.launch()
