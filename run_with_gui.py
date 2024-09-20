import os
import pathlib
import threading
import time
import tkinter as tk
from time import sleep
from tkinter import scrolledtext

from PIL import ImageTk

from dbd.AI_model import AI_model


class MonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DBD - Auto skill check")
        self.root.geometry("500x550")

        self.is_running = False
        self.save_hit = False

        # Setup GUI
        self.setup_gui()

    def setup_gui(self):
        # Run Button
        self.run_button = tk.Button(self.root, command=self.toggle_run, text="START")
        self.run_button.grid(row=0, column=0, padx=10, pady=20)

        # Label for FPS
        self.fps_label = tk.Label(self.root, text="Average AI fps: 0", font=("Helvetica", 12), fg="#F00")
        self.fps_label.grid(row=0, column=1, padx=10, pady=20, columnspan=2)

        # Create folders where we save images (if checkbox is checked)
        self.img_folder = "saved_images"
        pathlib.Path(self.img_folder).mkdir(exist_ok=True)
        for folder_idx in range(12): pathlib.Path(os.path.join(self.img_folder, str(folder_idx))).mkdir(exist_ok=True)

        # Images
        self.image_label_1 = tk.Label(self.root)
        self.image_label_1.grid(row=1, column=0, padx=0, pady=0)

        self.image_label_2 = tk.Label(self.root)
        self.image_label_2.grid(row=1, column=1, padx=0, pady=0)

        # Checkbox save hit skill checks
        self.feature_checkbox = tk.Checkbutton(self.root, text="Save hit frames in ./saved_images", command=self.toggle_save_hit)
        self.feature_checkbox.grid(row=2, column=1, padx=0, pady=0)

        # logs
        self.log_frame = tk.Frame(self.root)
        self.log_text = scrolledtext.ScrolledText(self.log_frame, height=10, width=60, state="disabled", bg="black", fg="white", wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=10)

        self.log_message("Press START to run the Auto skill check script")
        self.log_message("Ensure game fps\u226560 and AI fps\u226560")
        self.log_message("Left image: monitored frame (at 1fps)")
        self.log_message("Right image: last hit skill check")
        self.log_message("")
        self.log_message("Logs:")

    def toggle_run(self):
        if not self.is_running:
            self.log_message("START")
            self.is_running = True
            self.run_button.config(text="STOP")
            threading.Thread(target=self.monitor_loop).start()
        else:
            self.log_message("STOP")
            self.is_running = False
            self.run_button.config(text="START")
            self.update_fps_label(0.)

    def toggle_save_hit(self):
        self.save_hit = not self.save_hit

    def monitor_loop(self):
        # AI model
        ai_model = AI_model()
        self.log_message("Monitoring the screen...")
        self.log_message(ai_model.monitor)

        t0 = time.time()
        nb_frames = 0
        nb_hits = 0

        while self.is_running:
            screenshot = ai_model.grab_screenshot()
            image_pil = ai_model.screenshot_to_pil(screenshot)
            image_np = ai_model.pil_to_numpy(image_pil)
            nb_frames += 1

            pred, probs = ai_model.predict(image_np)
            hit, desc = ai_model.process(pred)

            # Debug only
            if pred != 0:
                path = os.path.join(self.img_folder, str(pred), "{}.png".format(nb_hits))
                self.log_message(f"{nb_hits}.png: {probs}")
                image_pil.save(path)
                nb_hits += 1

            if hit:
                # self.log_message(f"hit {desc} with confidence {100*prob:.1f}%")
                self.log_message(f"hit {desc}")
                self.display_image(image_pil, 1)

                if self.save_hit:
                    path = os.path.join(self.img_folder, str(pred), "hit_{}.png".format(nb_hits))
                    image_pil.save(path)
                    self.log_message("Saving frame " + path)
                    nb_hits += 1

                sleep(0.5)
                t0 = time.time()
                nb_frames = 0
                continue

            t_diff = time.time() - t0
            if t_diff > 1.0:
                fps = nb_frames / t_diff
                self.update_fps_label(fps)
                self.display_image(image_pil, 0)
                print(fps)

                t0 = time.time()
                nb_frames = 0

    def update_fps_label(self, fps):
        if fps > 58.:
            font = "#0F0"
            txt = "(Good)"
        else:
            font = "#F00"
            txt = "(Bad)"

        self.fps_label.config(text=f"Average AI fps: {fps:.1f} {txt}", fg=font)

    def display_image(self, image_pil, image_label_idx):
        if image_label_idx == 0:
            image_label = self.image_label_1
        else:
            image_label = self.image_label_2

        img_tk = ImageTk.PhotoImage(image_pil)
        image_label.config(image=img_tk)
        image_label.image = img_tk

    def log_message(self, message):
        """Logs a message to the log text box."""
        self.log_text.config(state="normal")  # Allow text to be inserted
        self.log_text.insert(tk.END, str(message) + "\n")  # Add the new message
        self.log_text.see(tk.END)  # Scroll to the end
        self.log_text.config(state="disabled")  # Make it read-only again


if __name__ == "__main__":
    root = tk.Tk()
    app = MonitorApp(root)
    root.mainloop()
