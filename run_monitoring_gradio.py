import os
import pathlib
import time

import gradio as gr

from dbd.AI_model import AI_model
from dbd.utils.directkeys import PressKey, ReleaseKey, SPACE


def monitor(onnx_ai_model, debug_option):
    if onnx_ai_model is None or not os.path.exists(onnx_ai_model) or ".onnx" not in onnx_ai_model:
        raise gr.Error("Invalid onnx file", duration=0)

    if debug_option is None:
        raise gr.Error("Invalid debug option")

    # AI model
    ai_model = AI_model(onnx_ai_model)

    # Variables
    t0 = time.time()
    nb_frames = 0
    nb_hits = 0

    # Create debug folders
    if debug_option == debug_options[2] or debug_option == debug_options[3]:
        pathlib.Path(debug_folder).mkdir(exist_ok=True)
        for folder_idx in range(len(ai_model.pred_dict)):
            pathlib.Path(os.path.join(debug_folder, str(folder_idx))).mkdir(exist_ok=True)

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

            yield gr.update(), image_pil, probs

            if debug_option == debug_options[2]:
                path = os.path.join(debug_folder, str(pred), "hit_{}.png".format(nb_hits))
                image_pil.save(path)
                nb_hits += 1

            time.sleep(0.5)
            t0 = time.time()
            nb_frames = 0
            continue

        # Compute fps
        t_diff = time.time() - t0
        if t_diff > 1.0:
            fps = round(nb_frames / t_diff, 1)

            if debug_option == debug_options[1]:
                yield fps, image_pil, gr.update()
            else:
                yield fps, gr.update(), gr.update()

            t0 = time.time()
            nb_frames = 0


if __name__ == "__main__":
    debug_folder = "saved_images"

    debug_options = ["None (default)",
                     "Display the monitored frame (a 224x224 center-cropped image, displayed at 1fps) instead of last hit skill check frame. Useful to check the monitored screen",
                     "Save hit skill check frames in {}".format(debug_folder),
                     "Save all skill check frames in {} (will impact fps)".format(debug_folder)]

    fps_info = "Number of frames per second the AI model analyses the monitored frame. Must be equal (or greater than) 60 to hit great skill checks properly. Check The GitHub FAQ for more details"

    demo = gr.Interface(title="DBD Auto skill check",
                        description="Please refer to https://github.com/Manuteaa/dbd_autoSkillCheck",
                        fn=monitor,
                        submit_btn="RUN",
                        clear_btn=None,

                        inputs=[gr.Dropdown(label="ONNX model filepath", choices=["model.onnx"], value="model.onnx", info="Filepath of the ONNX model (trained AI model)"),
                                gr.Dropdown(label="Debug options", choices=debug_options, value=debug_options[0], info="Optional options for debugging or analytics")],

                        outputs=[gr.Number(label="AI model FPS", info=fps_info),
                                 gr.Image(label="Last hit skill check frame", height=224),
                                 gr.Label(label="Skill check recognition")]
                        )
    demo.launch()
