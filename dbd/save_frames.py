import mss
from mss.tools import to_png
import mss.tools
import time
import os
import pyautogui


def get_monitor_attributes():
    width, height = pyautogui.size()
    object_size_h_ratio = 320 / 1080
    object_size = int(object_size_h_ratio * height)

    return {
        "top": height // 2 - object_size // 2,
        "left": width // 2 - object_size // 2,
        "width": object_size,
        "height": object_size
    }


if __name__ == '__main__':
    # Make new dataset folder, where we save the frames
    dataset_folder = time.strftime("%Y%m%d-%H%M%S")
    os.mkdir(dataset_folder)

    # Get monitor attributes
    monitor = get_monitor_attributes()

    try:
        with mss.mss() as sct:
            i = 0

            # Infinite loop
            print("Starting to save frames in folder: {}".format(dataset_folder))
            while True:
                screenshot = sct.grab(monitor)
                output_file = os.path.join(dataset_folder, "{:05d}.png".format(i))
                to_png(screenshot.rgb, screenshot.size, output=output_file)

                i += 1

    except KeyboardInterrupt:
        print("\nCapture stopped.")

        if output_file and os.path.exists(output_file):
            os.remove(output_file)
            print("Last file removed: {} (may be corrupted)".format(output_file))
