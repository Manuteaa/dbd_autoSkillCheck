import os
from pathlib import Path
from time import time, sleep

from gradio import (
    Interface, Dropdown, Radio, Checkbox, Number, Image, Label,
    update,
    Info, Warning, Error
)

from dbd.AI_model import AI_model
from dbd.utils.directkeys import PressKey, ReleaseKey, SPACE


def monitor(onnx_ai_model, device, debug_option, hit_ante, cap_fps):
    if onnx_ai_model is None or not os.path.exists(onnx_ai_model) or ".onnx" not in onnx_ai_model:
        raise Error("Invalid onnx file", duration=0)

    if device is None:
        raise Error("Invalid device option")

    if debug_option is None:
        raise Error("Invalid debug option")

    # AI model
    use_gpu = (device == devices[1])
    ai_model = AI_model(onnx_ai_model, use_gpu, cap_fps)

    is_using_cuda = ai_model.is_using_cuda()
    if is_using_cuda:
        Info("Running AI model on GPU (success)")
    else:
        Info("Running AI model on CPU")
        if device == devices[1]:
            Warning("Could not run AI model on GPU device. Check python console logs to debug.")

    if not hit_ante:
        ai_model.pred_dict[2]["hit"] = False

    # Create debug folders
    if debug_option == debug_options[2] or debug_option == debug_options[3]:
        Path(debug_folder).mkdir(exist_ok=True)
        for folder_idx in range(len(ai_model.pred_dict)):
            Path(os.path.join(debug_folder, str(folder_idx))).mkdir(exist_ok=True)

    # Variables
    t0 = time()
    nb_frames = 0
    nb_hits = 0

    time_per_iteration = 1 / 80.  # Cap to 80fps max
    t_start_iteration = time()

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
            PressKey(SPACE)
            ReleaseKey(SPACE)

            yield update(), image_pil, probs

            if debug_option == debug_options[2]:
                path = os.path.join(debug_folder, str(pred), "hit_{}.png".format(nb_hits))
                image_pil.save(path)
                nb_hits += 1

            sleep(0.5)
            t0 = time()
            nb_frames = 0
            continue

        # Compute fps
        t_diff = time() - t0
        if t_diff > 1.0:
            fps = round(nb_frames / t_diff, 1)

            if debug_option == debug_options[1]:
                yield fps, image_pil, update()
            else:
                yield fps, update(), update()

            t0 = time()
            nb_frames = 0

        # Cap AI model FPS
        if cap_fps:
            iteration_time = time() - t_start_iteration
            if iteration_time < time_per_iteration:
                sleep_time = time_per_iteration - iteration_time
                sleep(sleep_time)

            t_start_iteration = time()


if __name__ == "__main__":
    debug_folder = "saved_images"

    debug_options = ["None (default)",
                     "Display the monitored frame (a 224x224 center-cropped image, displayed at 1fps) instead of last hit skill check frame. Useful to check the monitored screen",
                     "Save hit skill check frames in {}".format(debug_folder),
                     "Save all skill check frames in {} (will impact fps)".format(debug_folder)]

    fps_info = "Number of frames per second the AI model analyses the monitored frame. Must be equal (or greater than) 60 to hit great skill checks properly. Check The GitHub FAQ for more details"
    devices = ["CPU (default)", "GPU"]

    demo = Interface(title="DBD Auto skill check",
                        description="Please refer to https://github.com/Manuteaa/dbd_autoSkillCheck",
                        fn=monitor,
                        submit_btn="RUN",
                        clear_btn=None,

                        inputs=[Dropdown(label="ONNX model filepath", choices=["model.onnx"], value="model.onnx", info="Filepath of the ONNX model (trained AI model)"),
                                Radio(choices=devices, value=devices[0], label="Device the AI onnx model will use"),
                                Dropdown(label="Debug options", choices=debug_options, value=debug_options[0], info="Optional options for debugging or analytics"),
                                Checkbox(label="Hit ante-frontier skill checks (uncheck if skill checks are hit too early)", value=True, info="Hit options"),
                                Checkbox(label="Reduce CPU usage (uncheck if AI model FPS < 60). No impact in GPU mode", value=True, info="AI options"),
                                ],

                        outputs=[Number(label="AI model FPS", info=fps_info),
                                 Image(label="Last hit skill check frame", height=224),
                                 Label(label="Skill check recognition")]
                        )
    demo.launch()
