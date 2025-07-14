import os
import logging
import sys
from time import time, sleep
from typing import Optional, Tuple, Any, Generator

from gradio import (
    Dropdown, Radio, Number, Image, Label, Button, Slider,
    skip, Info, Warning, Error, Blocks, Row, Column, Markdown,
)

from dbd.AI_model import AI_model
from dbd.utils.directkeys import PressKey, ReleaseKey, SPACE
from dbd.utils.monitor import get_monitors, get_monitor_attributes, get_frame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('dbd_auto_skill_check.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


ai_model: Optional[AI_model] = None

def cleanup() -> float:
    """Clean up AI model resources safely."""
    global ai_model
    try:
        if ai_model is not None:
            logger.info("Cleaning up AI model resources...")
            ai_model.cleanup()
            del ai_model
            ai_model = None
            logger.info("AI model cleanup completed successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        ai_model = None
    return 0.


def monitor(ai_model_path: str, device: str, monitor_id: int, hit_ante: float, nb_cpu_threads: int) -> Generator[Tuple[Any, ...], None, None]:
    """
    Main monitoring function that runs the AI model for skill check detection.
    
    Args:
        ai_model_path: Path to the ONNX or TensorRT model file
        device: Device to use ('CPU' or 'GPU')
        monitor_id: ID of the monitor to capture
        hit_ante: Delay in milliseconds for ante-frontier hits
        nb_cpu_threads: Number of CPU threads to use
        
    Yields:
        Tuple containing FPS, image, and probabilities for UI updates
        
    Raises:
        Error: If model loading or initialization fails
    """
    # Input validation
    if not ai_model_path or not os.path.exists(ai_model_path):
        logger.error(f"Invalid AI model file: {ai_model_path}")
        raise Error("Invalid AI model file. Please select a valid model.", duration=0)

    if device not in ["CPU (default)", "GPU"]:
        logger.error(f"Invalid device option: {device}")
        raise Error("Invalid device option. Please select CPU or GPU.")

    if not isinstance(monitor_id, int) or monitor_id < 1:
        logger.error(f"Invalid monitor ID: {monitor_id}")
        raise Error("Invalid monitor option. Please select a valid monitor.")

    if not isinstance(nb_cpu_threads, int) or nb_cpu_threads < 1:
        logger.error(f"Invalid CPU threads count: {nb_cpu_threads}")
        nb_cpu_threads = 4  # Default fallback
        logger.warning(f"Using default CPU threads: {nb_cpu_threads}")

    use_gpu = (device == devices[1])
    logger.info(f"Starting monitoring with model: {ai_model_path}, device: {device}, monitor: {monitor_id}")

    try:
        global ai_model
        ai_model = AI_model(ai_model_path, use_gpu, nb_cpu_threads, monitor_id)
        execution_provider = ai_model.check_provider()
        logger.info(f"AI model loaded successfully with provider: {execution_provider}")
    except Exception as e:
        logger.error(f"Error loading AI model: {e}")
        raise Error(f"Error when loading AI model: {e}", duration=0)

    # Display provider information
    if execution_provider == "CUDAExecutionProvider":
        Info("Running AI model on GPU (success, CUDA)")
        logger.info("GPU acceleration active: CUDA")
    elif execution_provider == "DmlExecutionProvider":
        Info("Running AI model on GPU (success, DirectML)")
        logger.info("GPU acceleration active: DirectML")
    elif execution_provider == "TensorRT":
        Info("Running AI model on GPU (success, TensorRT)")
        logger.info("GPU acceleration active: TensorRT")
    else:
        Info(f"Running AI model on CPU (success, {nb_cpu_threads} threads)")
        logger.info(f"CPU execution active: {nb_cpu_threads} threads")
        if use_gpu:
            Warning("Could not run AI model on GPU device. Check python console logs to debug.")
            logger.warning("GPU requested but not available, falling back to CPU")

    # Initialize monitoring variables
    t0 = time()
    nb_frames = 0
    total_hits = 0
    logger.info("Starting skill check monitoring loop...")

    try:
        while True:
            try:
                # Capture and process frame
                screenshot = ai_model.grab_screenshot()
                image_pil = ai_model.screenshot_to_pil(screenshot)
                image_np = ai_model.pil_to_numpy(image_pil)
                nb_frames += 1

                # AI prediction
                pred, desc, probs, should_hit = ai_model.predict(image_np)

                if should_hit:
                    logger.debug(f"Skill check detected: {desc}")
                    
                    # Apply ante-frontier delay if needed
                    if pred == 2 and hit_ante > 0:
                        sleep(hit_ante * 0.001)
                        logger.debug(f"Applied ante-frontier delay: {hit_ante}ms")

                    # Execute key press
                    PressKey(SPACE)
                    sleep(0.005)
                    ReleaseKey(SPACE)
                    
                    total_hits += 1
                    logger.info(f"Skill check hit #{total_hits}: {desc}")

                    yield skip(), image_pil, probs

                    # Cooldown to avoid multiple hits
                    sleep(0.5)
                    t0 = time()
                    nb_frames = 0
                    continue

                # Compute and yield FPS
                t_diff = time() - t0
                if t_diff > 1.0:
                    fps = round(nb_frames / t_diff, 1)
                    logger.debug(f"AI model FPS: {fps}")
                    yield fps, skip(), skip()

                    t0 = time()
                    nb_frames = 0
                    
            except Exception as e:
                logger.error(f"Error in monitoring frame: {e}")
                # Continue monitoring despite frame errors
                continue

    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")
    except Exception as e:
        logger.error(f"Critical error in monitoring loop: {e}")
    finally:
        logger.info(f"Monitoring stopped. Total hits: {total_hits}")
        print("Monitoring stopped.")


if __name__ == "__main__":
    logger.info("Starting DBD Auto Skill Check application...")
    
    # Configuration constants
    MODELS_FOLDER = "models"
    DEFAULT_CROP_SIZE = 520  # For debug display
    
    fps_info = "Number of frames per second the AI model analyses the monitored frame."
    devices = ["CPU (default)", "GPU"]
    cpu_choices = [("Low", 2), ("Normal", 4), ("High", 6), ("Computer Killer Mode", 8)]

    try:
        # Find available AI models with validation
        if not os.path.exists(MODELS_FOLDER):
            logger.error(f"Models folder not found: {MODELS_FOLDER}")
            raise FileNotFoundError(f"Models folder '{MODELS_FOLDER}' does not exist. Please create it and add model files.")
        
        model_files = [
            (f, os.path.join(MODELS_FOLDER, f)) 
            for f in os.listdir(MODELS_FOLDER) 
            if f.endswith((".onnx", ".trt")) and os.path.isfile(os.path.join(MODELS_FOLDER, f))
        ]
        
        if len(model_files) == 0:
            logger.error(f"No valid AI models found in {MODELS_FOLDER}/")
            raise FileNotFoundError(f"No AI model files (.onnx or .trt) found in {MODELS_FOLDER}/. Please add model files.")
        
        logger.info(f"Found {len(model_files)} AI model(s): {[f[0] for f in model_files]}")

        # Monitor selection with error handling
        try:
            monitor_choices = get_monitors()
            if not monitor_choices:
                logger.error("No monitors detected")
                raise RuntimeError("No monitors detected. Please check your display setup.")
            logger.info(f"Detected {len(monitor_choices)} monitor(s)")
        except Exception as e:
            logger.error(f"Error detecting monitors: {e}")
            raise RuntimeError(f"Failed to detect monitors: {e}")

        def switch_monitor_cb(monitor_id: int):
            """Safely switch monitor and update preview."""
            try:
                monitor = get_monitor_attributes(monitor_id, crop_size=DEFAULT_CROP_SIZE)
                return get_frame(monitor)
            except Exception as e:
                logger.error(f"Error switching to monitor {monitor_id}: {e}")
                return None

        # Create the Gradio interface
        with Blocks(title="DBD Auto Skill Check", theme="soft") as webui:
            # Header
            Markdown(
                "<h1 style='text-align: center;'>🎯 DBD Auto Skill Check</h1>"
                "<p style='text-align: center; color: #666;'>AI-powered skill check detection for Dead by Daylight</p>", 
                elem_id="title"
            )
            Markdown(
                "<p style='text-align: center;'>"
                "<a href='https://github.com/Manuteaa/dbd_autoSkillCheck' target='_blank'>📂 GitHub Repository</a> | "
                "<a href='https://discord.gg/3mewehHHpZ' target='_blank'>💬 Discord Support</a>"
                "</p>"
            )

            with Row():
                with Column(variant="panel"):
                    with Column(variant="panel"):
                        Markdown("🤖 **AI Inference Settings**")
                        ai_model_path = Dropdown(
                            choices=model_files, 
                            value=model_files[0][1], 
                            label="AI Model to use (ONNX or TensorRT Engine)",
                            info="Select the trained AI model file"
                        )
                        device = Radio(
                            choices=devices, 
                            value=devices[0], 
                            label="Processing Device",
                            info="CPU is recommended for stability, GPU for performance"
                        )
                        monitor_id = Dropdown(
                            choices=monitor_choices, 
                            value=monitor_choices[0][1], 
                            label="Monitor to capture",
                            info="Select which monitor to analyze for skill checks"
                        )
                    
                    with Column(variant="panel"):
                        Markdown("⚙️ **Advanced Options**")
                        hit_ante = Slider(
                            minimum=0, maximum=50, step=5, value=20, 
                            label="Ante-frontier hit delay (ms)",
                            info="Delay for ante-frontier skill checks to compensate for latency"
                        )
                        cpu_stress = Radio(
                            label="CPU Usage Level",
                            choices=cpu_choices,
                            value=cpu_choices[1][1],
                            info="Higher values improve AI FPS but use more CPU"
                        )
                    
                    with Column():
                        with Row():
                            run_button = Button("🚀 START MONITORING", variant="primary", size="lg")
                            stop_button = Button("⏹️ STOP", variant="stop", size="lg")

                with Column(variant="panel"):
                    Markdown("📊 **Live Monitoring**")
                    fps = Number(
                        label="AI Model FPS", 
                        info=fps_info, 
                        interactive=False,
                        precision=1
                    )
                    image_pil = Image(
                        label="Last detected skill check", 
                        height=224, 
                        interactive=False,
                        show_label=True
                    )
                    probs = Label(
                        label="AI Prediction Confidence",
                        show_label=True
                    )

            # Event handlers
            monitoring = run_button.click(
                fn=monitor, 
                inputs=[ai_model_path, device, monitor_id, hit_ante, cpu_stress],
                outputs=[fps, image_pil, probs]
            )

            stop_button.click(fn=cleanup, inputs=None, outputs=fps)
            monitor_id.change(fn=switch_monitor_cb, inputs=monitor_id, outputs=image_pil)

        # Launch the web interface
        logger.info("Launching web interface...")
        webui.launch(
            server_name="127.0.0.1",
            server_port=7860,
            show_error=True,
            quiet=False
        )
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        print(f"Error: {e}")
        sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\nApplication interrupted by user.")
    
    finally:
        logger.info("Cleaning up and shutting down...")
        cleanup()
        logger.info("Application shutdown complete")
