from mss.tools import to_png
import mss.tools
import time
import os

from dbd.utils.monitoring_mss import Monitoring_mss


if __name__ == '__main__':
    # Make new dataset folder, where we save the frames
    dataset_folder = time.strftime("%Y%m%d-%H%M%S")
    os.mkdir(dataset_folder)

    # Get monitor attributes
    with Monitoring_mss(crop_size=320) as mon:

        try:
            i = 0

            # Infinite loop
            print("Starting to save frames in folder: {}".format(dataset_folder))
            while True:
                screenshot = mon._get_frame()
                output_file = os.path.join(dataset_folder, "{:05d}.png".format(i))
                to_png(screenshot.rgb, screenshot.size, output=output_file)

                i += 1

        except KeyboardInterrupt:
            print("\nCapture stopped.")

            if output_file and os.path.exists(output_file):
                os.remove(output_file)
                print("Last file removed: {} (may be corrupted)".format(output_file))
