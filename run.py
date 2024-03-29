import numpy as np
import mss
import onnxruntime
import pyautogui
from PIL import Image
import os
from time import sleep

from dbd.utils.directkeys import PressKey, ReleaseKey, SPACE

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_monitor_attributes():
    width, height = pyautogui.size()
    object_size = 224

    monitor = {"top": height // 2 - object_size // 2,
               "left": width // 2 - object_size // 2,
               "width": object_size,
               "height": object_size}

    return monitor

def screenshot_to_numpy(screenshot):
    img = np.array(screenshot, dtype=np.float32)
    img = np.flip(img[:, :, :3], 2)
    img = img / 255.0
    img = (img - MEAN) / STD

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)

    img = np.float32(img)
    return img


if __name__ == '__main__':
    save_images = False  # if True, save detected great skill check images in saved_images/
    debug_monitor = False  # if True, check the image saved_images/monitored_image.png

    # Get monitor attributes to grab frames
    monitor = get_monitor_attributes()

    # Trained model
    filepath = "model.onnx"
    ort_session = onnxruntime.InferenceSession(filepath)
    input_name = ort_session.get_inputs()[0].name

    img_folder = "saved_images"
    if (save_images or debug_monitor) and not os.path.isdir(img_folder):
        os.mkdir(img_folder)

    with mss.mss() as sct:
        print("Monitoring the screen...")

        detected_images_idx = 0
        while True:
            screenshot = sct.grab(monitor)
            img = screenshot_to_numpy(screenshot)

            # To check the monitor settings
            if debug_monitor:
                image = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                image.save(os.path.join(img_folder, "monitored_image.png"))

            ort_inputs = {input_name: img}
            ort_outs = ort_session.run(None, ort_inputs)
            pred = np.argmax(np.squeeze(ort_outs, 0))

            # Hit !!
            if pred == 2:
                PressKey(SPACE)
                ReleaseKey(SPACE)
                sleep(1)

                if save_images:
                    img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                    img.save(os.path.join(img_folder, "{}.png".format(detected_images_idx)))
                    detected_images_idx += 1
