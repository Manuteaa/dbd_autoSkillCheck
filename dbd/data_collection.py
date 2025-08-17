from mss.tools import to_png
import time
import os
import importlib.util


from dbd.utils.monitoring_mss import Monitoring_mss
from dbd.predict_folder import infer_from_folder_onnx
from dbd.utils.dataset_utils import delete_similar_images, delete_consecutive_images


if __name__ == '__main__':
    assert os.path.exists("models/model.onnx"), "Please run this script from the root of the project"

    # Make new dataset folder, where we save the frames
    dataset_folder = time.strftime("%Y%m%d-%H%M%S")
    os.mkdir(dataset_folder)

    # Get monitor attributes
    with Monitoring_mss(crop_size=320) as mon:
        try:
            # Infinite loop
            print("Starting to save frames in folder: {}".format(dataset_folder))
            print("Press Ctrl+C to stop capturing frames.")

            i = 0
            t0 = time.time()
            while True:

                screenshot = mon.get_raw_frame()
                output_file = os.path.join(dataset_folder, "{:05d}.png".format(i))
                to_png(screenshot.rgb, screenshot.size, output=output_file)

                i += 1
                elapsed_time = time.time() - t0
                if elapsed_time > 10:
                    print("Capturing frames at {} FPS".format(i // elapsed_time))
                    i = 0
                    t0 = time.time()

        except KeyboardInterrupt:
            print("\nCapture stopped.")

            if output_file and os.path.exists(output_file):
                os.remove(output_file)
                print("Last file removed: {} (may be corrupted)".format(output_file))

            # Create one folder per category
            for idx in range(0, 12):
                pred_folder_idx = os.path.join(dataset_folder, str(idx))
                os.makedirs(pred_folder_idx, exist_ok=True)

            # PREDICT
            print(f"Pre-annotation using AI. Please wait...", flush=True)
            t0 = time.time()
            torch_ok = importlib.util.find_spec("torch") is not None
            results1 = infer_from_folder_onnx(dataset_folder, "models/model.onnx", use_gpu=torch_ok, move=True)
            print(f"Pre-annotation using AI done in {time.time() - t0:.2f} seconds", flush=True)

            # Reduce frames of folder 0
            print("Reducing frames in folder 0...")
            folder_0 = os.path.join(dataset_folder, "0")
            delete_consecutive_images(folder_0, 50)
            delete_similar_images(folder_0)

            print("Data collection completed. Please review the folder: {}".format(dataset_folder), flush=True)
