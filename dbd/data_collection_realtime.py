import time
import os
import importlib.util
from PIL import Image

from dbd.AI_model import AI_model
from dbd.utils.monitoring_mss import Monitoring_mss


if __name__ == '__main__':
    NB_CPU_THREADS = 4  # Adjust based on your CPU capabilities
    model_path = "models/model.onnx"
    assert os.path.exists(model_path), "Please run this script from the root of the project"

    # Make new dataset folder, where we save the frames
    dataset_folder = time.strftime("%Y%m%d-%H%M%S")
    os.mkdir(dataset_folder)

    # Create one folder per category
    for idx in range(0, 11):
        pred_folder_idx = os.path.join(dataset_folder, str(idx))
        os.makedirs(pred_folder_idx, exist_ok=True)

    # Get monitor attributes
    with Monitoring_mss(crop_size=320) as mon:
        try:
            # Init AI model
            print("Loading AI model...")
            torch_ok = importlib.util.find_spec("torch") is not None
            ai_model = AI_model(model_path=model_path, use_gpu=torch_ok, nb_cpu_threads=NB_CPU_THREADS)
            ai_model.monitor.stop()
            print(f"AI model loaded using {ai_model.check_provider()} for inference")

            # Infinite loop
            print("Starting to collect data in folder: {}".format(dataset_folder))
            print("Press Ctrl+C to stop data collection.")

            i = 0
            image_idx = 0
            t0 = time.time()
            while True:
                frame_np = mon.get_frame_np()

                frame_np_224 = frame_np[48:272, 48:272]  # center crop to 224x224
                pred, _, _, _ = ai_model.predict(frame_np_224)

                if pred != 0:
                    output_file = os.path.join(dataset_folder, str(pred), "{:05d}.png".format(image_idx))
                    image_idx += 1

                    img_pil = Image.fromarray(frame_np)
                    img_pil.save(output_file)

                i += 1
                elapsed_time = time.time() - t0
                if elapsed_time > 10:
                    print("Capturing frames at {} FPS".format(i // elapsed_time))
                    i = 0
                    t0 = time.time()

        except KeyboardInterrupt:
            print("\nCapture stopped.")
            print("Data collection completed. Please review the folder: {}".format(dataset_folder), flush=True)
            input("Press Enter to Exit...")
