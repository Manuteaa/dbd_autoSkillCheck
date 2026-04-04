#!/usr/bin/env python3
"""
DBD Auto Skill Check - Terminal UI
A terminal-based alternative to the Gradio Web UI.
Works on both Windows and Linux (X11 & Wayland via OBS VirtualCam).
"""

import os
import sys
import json
import signal
import threading
import argparse
import logging
from datetime import datetime
from time import time, sleep
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.prompt import IntPrompt, Confirm
from rich import box

from dbd.AI_model import AI_model
from dbd.utils.directkeys import PressKey, ReleaseKey, SPACE, SHIFT, ACTIVE_INPUT_MODE
from dbd.utils.humanizer import Humanizer
from dbd.utils.monitoring_mss import Monitoring_mss

# Optional imports
try:
    from dbd.utils.monitoring_bettercam import Monitoring_bettercam
    bettercam_ok = True
except ImportError:
    bettercam_ok = False

try:
    from dbd.utils.monitoring_v4l2 import Monitoring_v4l2, V4L2_AVAILABLE
    v4l2_ok = V4L2_AVAILABLE
except ImportError:
    v4l2_ok = False
    V4L2_AVAILABLE = False


console = Console()

# Config file path (same directory as tui.py)
CONFIG_PATH = Path(__file__).parent / "config.json"
LOG_DIR = Path(__file__).parent / "logs"

def setup_logging():
    """Configure logging to file."""
    if not LOG_DIR.exists():
        LOG_DIR.mkdir()
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = LOG_DIR / f"session_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout) # optional: checks args
        ]
    )
    # Remove StreamHandler to avoid double printing in TUI
    logging.getLogger().handlers = [logging.FileHandler(log_file)]
    return log_file

# FPS Presets: (label, ante-frontier delay in ms)
FPS_PRESETS = [
    ("60 FPS (locked)", -250),
    ("90 FPS (locked)", -125),
    ("120 FPS (locked)", 0),
    ("Custom", None),
]


# ── Platform Detection ──────────────────────────────────────────────────────

def detect_platform():
    """Detect current platform and return info dict."""
    info = {
        "os": sys.platform,
        "display": "unknown",
        "is_wayland": False,
        "is_windows": False,
    }

    if sys.platform == "win32":
        info["display"] = "Windows"
        info["is_windows"] = True
    elif sys.platform == "darwin":
        info["display"] = "macOS"
    else:
        session_type = os.environ.get('XDG_SESSION_TYPE', '')
        wayland_display = os.environ.get('WAYLAND_DISPLAY', '')

        if session_type == 'wayland' or wayland_display:
            info["display"] = "Linux (Wayland)"
            info["is_wayland"] = True
        else:
            info["display"] = "Linux (X11)"

    return info


def get_platform_default_monitoring(platform_info):
    """Return the best default monitoring type for the detected platform."""
    if platform_info["is_windows"]:
        # Windows → bettercam if available, else mss
        if bettercam_ok:
            return "bettercam"
        return "mss"
    elif platform_info["is_wayland"]:
        # Wayland → v4l2 (OBS VirtualCam) if available, else mss
        if v4l2_ok:
            return "v4l2 (OBS VirtualCam)"
        return "mss"
    else:
        # X11 → mss (works natively)
        return "mss"


# ── Wayland Remote Access Consent ────────────────────────────────────────────

def trigger_wayland_consent():
    """
    Trigger the Wayland remote access consent dialog by sending a dummy key press.
    
    On Wayland, when an application (via pynput) first tries to simulate
    keyboard input, the compositor shows a "Remote Access" / "Input Capture"
    consent dialog. By doing this early at startup, the user can accept
    the permission while still configuring settings — rather than during
    active gameplay.
    """
    def _send_dummy_input():
        try:
            # A short press/release cycle is enough to trigger the consent dialog
            # Using SHIFT avoids typing unwanted characters in the TUI menu
            PressKey(SHIFT)
            sleep(0.05)
            ReleaseKey(SHIFT)
        except Exception:
            pass

    thread = threading.Thread(target=_send_dummy_input, daemon=True)
    thread.start()
    return thread


# ── Config File ──────────────────────────────────────────────────────────────

def load_config():
    """Load saved config from config.json. Returns dict or None."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None


def save_config(config):
    """Save config dict to config.json."""
    try:
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except IOError as e:
        console.print(f"[red]Could not save config: {e}[/red]")
        return False


# ── Main Application ─────────────────────────────────────────────────────────

class DBDAutoSkillCheck:
    """Terminal UI for DBD Auto Skill Check"""

    def __init__(self):
        self.ai_model = None
        self.running = False
        self.fps = 0.0
        self.last_hit_desc = ""
        self.last_probs = {}
        self.total_hits = 0
        self.session_start = None
        self.lock = threading.Lock()

        # Detect platform
        self.platform_info = detect_platform()

        # Set platform-aware defaults
        self.model_path = None
        self.use_gpu = False
        self.monitoring_type = get_platform_default_monitoring(self.platform_info)
        self.monitor_id = 0
        self.hit_ante = 0
        self.cpu_threads = 4
        self.humanizer = Humanizer()
        self.use_hesitation = True  # Default: active for realism

        # On Wayland, trigger input consent dialog early
        self._consent_thread = None
        if self.platform_info["is_wayland"]:
            console.print(
                "[cyan]🔗 Wayland detected — triggering remote access consent...[/cyan]"
            )
            self._consent_thread = trigger_wayland_consent()

    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def show_banner(self):
        banner_text = (
            "[bold cyan]"
            "  ____  ____  ____     _         _        ____  _    _ _ _\n"
            " |  _ \\| __ )| __ )   / \\  _   _| |_ ___ / ___|| | _(_) | |\n"
            " | | | |  _ \\| | | | / _ \\| | | | __/ _ \\\\___ \\| |/ / | | |\n"
            " | |_| | |_) | |_| |/ ___ \\ |_| | || (_) |___) |   <| | | |\n"
            " |____/|____/|____/_/   \\_\\__,_|\\__\\___/|____/|_|\\_\\_|_|_|\n"
            "                                            Check TUI\n"
            "[/bold cyan]"
        )
        console.print(Panel(banner_text, border_style="cyan", box=box.DOUBLE))
        console.print("  [dim]https://github.com/Manuteaa/dbd_autoSkillCheck[/dim]")
        console.print(f"  [dim]Platform: {self.platform_info['display']}[/dim]")

        # Show default capture method hint
        default_mon = get_platform_default_monitoring(self.platform_info)
        console.print(f"  [dim]Default Capture: {default_mon}[/dim]\n")

    def get_available_models(self):
        models_folder = "models"
        if not os.path.exists(models_folder):
            return []
        return [(f, os.path.join(models_folder, f))
                for f in os.listdir(models_folder)
                if f.endswith(".onnx") or f.endswith(".trt")]

    def get_monitoring_choices(self):
        choices = ["mss"]
        if bettercam_ok:
            choices.insert(0, "bettercam")
        if v4l2_ok:
            choices.append("v4l2 (OBS VirtualCam)")
        return choices

    def get_monitor_list(self, monitoring_type):
        try:
            if monitoring_type.startswith("v4l2") and v4l2_ok:
                return Monitoring_v4l2.get_monitors_info()
            elif monitoring_type == "bettercam" and bettercam_ok:
                return Monitoring_bettercam.get_monitors_info()
            else:
                return Monitoring_mss.get_monitors_info()
        except Exception as e:
            console.print(f"[red]Could not get monitor list: {e}[/red]")
            return []

    def apply_config(self, config):
        """Apply a loaded config dict to current settings."""
        if config is None:
            return

        if "use_gpu" in config:
            self.use_gpu = config["use_gpu"]
        if "monitoring_type" in config:
            # Validate the monitoring type is available
            choices = self.get_monitoring_choices()
            if config["monitoring_type"] in choices:
                self.monitoring_type = config["monitoring_type"]
        if "monitor_id" in config:
            self.monitor_id = config["monitor_id"]
        if "hit_ante" in config:
            self.hit_ante = config["hit_ante"]
        if "cpu_threads" in config:
            self.cpu_threads = config["cpu_threads"]
        if "use_hesitation" in config:
            self.use_hesitation = config["use_hesitation"]
        if "model_index" in config:
            models = self.get_available_models()
            idx = config["model_index"]
            if models and 0 <= idx < len(models):
                self.model_path = models[idx][1]

    def select_settings(self):
        """Interactive settings selection."""
        self.clear_screen()
        self.show_banner()

        console.print(Panel("[bold yellow]SETTINGS[/bold yellow]", box=box.ROUNDED))
        console.print()

        # --- Model ---
        models = self.get_available_models()
        if not models:
            console.print("[red]No ONNX/TRT model found in 'models/' folder![/red]")
            return False

        console.print("[bold cyan]AI Models:[/bold cyan]")
        for i, (name, _) in enumerate(models, 1):
            console.print(f"  [dim]{i}.[/dim] {name}")

        while True:
            choice = IntPrompt.ask("\n[yellow]Select model[/yellow]", default=1)
            if 1 <= choice <= len(models):
                self.model_path = models[choice - 1][1]
                break
            console.print("[red]Invalid selection![/red]")
        console.print(f"[green]> Model: {models[choice - 1][0]}[/green]\n")

        # --- Device ---
        console.print("[bold cyan]Device:[/bold cyan]")
        console.print("  [dim]1.[/dim] CPU (default)")
        console.print("  [dim]2.[/dim] GPU (CUDA/DirectML)")
        device_choice = IntPrompt.ask("[yellow]Select device[/yellow]", default=1)
        self.use_gpu = (device_choice == 2)
        console.print(f"[green]> Device: {'GPU' if self.use_gpu else 'CPU'}[/green]\n")

        # --- Monitoring ---
        monitoring_choices = self.get_monitoring_choices()
        console.print("[bold cyan]Screen Capture Method:[/bold cyan]")

        # Determine platform-aware default
        platform_default = get_platform_default_monitoring(self.platform_info)
        if platform_default in monitoring_choices:
            default_idx = monitoring_choices.index(platform_default) + 1
        else:
            default_idx = monitoring_choices.index("mss") + 1

        for i, method in enumerate(monitoring_choices, 1):
            extra = ""
            is_default = " [bold green](recommended)[/bold green]" if method == platform_default else ""
            if method.startswith("v4l2"):
                extra = " [dim](Linux - OBS VirtualCam, lowest latency)[/dim]"
            elif method == "bettercam":
                extra = " [dim](Windows - high performance)[/dim]"
            elif method == "mss":
                extra = " [dim](cross-platform, X11 only on Linux)[/dim]"
            console.print(f"  [dim]{i}.[/dim] {method}{extra}{is_default}")

        mon_choice = IntPrompt.ask("[yellow]Select method[/yellow]", default=default_idx)
        if 1 <= mon_choice <= len(monitoring_choices):
            self.monitoring_type = monitoring_choices[mon_choice - 1]
        console.print(f"[green]> Method: {self.monitoring_type}[/green]\n")

        # --- Monitor ---
        monitors = self.get_monitor_list(self.monitoring_type)
        if monitors:
            console.print("[bold cyan]Monitor / Source:[/bold cyan]")
            for i, (name, _) in enumerate(monitors, 1):
                console.print(f"  [dim]{i}.[/dim] {name}")
            monitor_choice = IntPrompt.ask("[yellow]Select monitor[/yellow]", default=1)
            if 1 <= monitor_choice <= len(monitors):
                self.monitor_id = monitors[monitor_choice - 1][1]
            console.print(f"[green]> Monitor: {monitors[monitor_choice - 1][0]}[/green]\n")

        # --- FPS Preset ---
        console.print("[bold cyan]FPS Preset (sets ante-frontier delay):[/bold cyan]")
        for i, (label, val) in enumerate(FPS_PRESETS, 1):
            detail = f" [{val}ms]" if val is not None else ""
            console.print(f"  [dim]{i}.[/dim] {label}{detail}")

        preset_choice = IntPrompt.ask("[yellow]Select preset[/yellow]", default=3)
        if 1 <= preset_choice <= len(FPS_PRESETS):
            label, val = FPS_PRESETS[preset_choice - 1]
            if val is not None:
                self.hit_ante = val
                console.print(f"[green]> Preset: {label} (delay: {val}ms)[/green]\n")
            else:
                # Custom
                self.hit_ante = IntPrompt.ask("[yellow]Custom delay (ms, -300 to 50)[/yellow]", default=0)
                console.print(f"[green]> Custom delay: {self.hit_ante}ms[/green]\n")

        # --- CPU Threads ---
        console.print("[bold cyan]CPU Thread Count:[/bold cyan]")
        console.print("  [dim]1.[/dim] Low (2 threads)")
        console.print("  [dim]2.[/dim] Normal (4 threads)")
        console.print("  [dim]3.[/dim] High (6 threads)")
        console.print("  [dim]4.[/dim] Maximum (8 threads)")
        thread_choice = IntPrompt.ask("[yellow]Select[/yellow]", default=2)
        self.cpu_threads = {1: 2, 2: 4, 3: 6, 4: 8}.get(thread_choice, 4)
        console.print(f"[green]> Threads: {self.cpu_threads}[/green]\n")

        # --- Humanizer Hesitation ---
        console.print("[bold cyan]Human-like Hesitation:[/bold cyan]")
        console.print("[dim]Adds random micro-delays (~7% chance) to mimic human reaction.[/dim]")
        console.print("[dim]Increases realism but introduces a slight risk of hitting 'Good' instead of 'Great'.[/dim]")
        self.use_hesitation = Confirm.ask("[yellow]Enable hesitation?[/yellow] (Recommended for anti-cheat)", default=True)
        status = "Active" if self.use_hesitation else "Disabled"
        console.print(f"[green]> Hesitation: {status}[/green]\n")

        return True

    def edit_defaults(self):
        """Interactive menu to edit and save default settings to config.json."""
        self.clear_screen()
        self.show_banner()

        console.print(
            Panel(
                "[bold yellow]DEFAULT SETTINGS EDITOR[/bold yellow]\n"
                "[dim]Configure defaults used by quick-start mode (-s)[/dim]",
                box=box.ROUNDED
            )
        )
        console.print()

        # Load existing config
        existing = load_config()
        if existing:
            console.print("[dim]Loaded existing config.json[/dim]\n")
        else:
            console.print("[dim]No existing config found, using platform defaults[/dim]\n")

        config = {}

        # --- Model ---
        models = self.get_available_models()
        if not models:
            console.print("[red]No ONNX/TRT model found in 'models/' folder![/red]")
            return

        console.print("[bold cyan]Default AI Model:[/bold cyan]")
        current_model_idx = 0
        if existing and "model_index" in existing:
            saved_idx = existing["model_index"]
            if 0 <= saved_idx < len(models):
                current_model_idx = saved_idx

        for i, (name, _) in enumerate(models, 1):
            marker = " [green](current)[/green]" if i - 1 == current_model_idx else ""
            console.print(f"  [dim]{i}.[/dim] {name}{marker}")

        choice = IntPrompt.ask(
            "\n[yellow]Select default model[/yellow]",
            default=current_model_idx + 1
        )
        if 1 <= choice <= len(models):
            config["model_index"] = choice - 1
        else:
            config["model_index"] = current_model_idx
        console.print(f"[green]> Model: {models[config['model_index']][0]}[/green]\n")

        # --- Device ---
        current_gpu = existing.get("use_gpu", False) if existing else False
        console.print("[bold cyan]Default Device:[/bold cyan]")
        console.print(f"  [dim]1.[/dim] CPU{' [green](current)[/green]' if not current_gpu else ''}")
        console.print(f"  [dim]2.[/dim] GPU (CUDA/DirectML){' [green](current)[/green]' if current_gpu else ''}")
        device_choice = IntPrompt.ask(
            "[yellow]Select default device[/yellow]",
            default=2 if current_gpu else 1
        )
        config["use_gpu"] = (device_choice == 2)
        console.print(f"[green]> Device: {'GPU' if config['use_gpu'] else 'CPU'}[/green]\n")

        # --- Monitoring ---
        monitoring_choices = self.get_monitoring_choices()
        platform_default = get_platform_default_monitoring(self.platform_info)
        current_mon = existing.get("monitoring_type", platform_default) if existing else platform_default

        console.print("[bold cyan]Default Screen Capture Method:[/bold cyan]")
        default_mon_idx = 1
        for i, method in enumerate(monitoring_choices, 1):
            extra = ""
            markers = []
            if method == current_mon:
                markers.append("[green](current)[/green]")
            if method == platform_default:
                markers.append("[cyan](platform default)[/cyan]")
            marker_str = " " + " ".join(markers) if markers else ""

            if method.startswith("v4l2"):
                extra = " [dim](Linux - OBS VirtualCam)[/dim]"
            elif method == "bettercam":
                extra = " [dim](Windows - high performance)[/dim]"
            elif method == "mss":
                extra = " [dim](cross-platform, X11 only)[/dim]"

            console.print(f"  [dim]{i}.[/dim] {method}{extra}{marker_str}")

            if method == current_mon:
                default_mon_idx = i

        mon_choice = IntPrompt.ask(
            "[yellow]Select default method[/yellow]",
            default=default_mon_idx
        )
        if 1 <= mon_choice <= len(monitoring_choices):
            config["monitoring_type"] = monitoring_choices[mon_choice - 1]
        else:
            config["monitoring_type"] = current_mon
        console.print(f"[green]> Method: {config['monitoring_type']}[/green]\n")

        # --- FPS Preset / Hit Ante ---
        current_ante = existing.get("hit_ante", 0) if existing else 0
        console.print("[bold cyan]Default FPS Preset (ante-frontier delay):[/bold cyan]")
        console.print(f"  [dim]Current delay: {current_ante}ms[/dim]")

        default_preset = 3  # 120 FPS
        for i, (label, val) in enumerate(FPS_PRESETS, 1):
            detail = f" [{val}ms]" if val is not None else ""
            marker = ""
            if val is not None and val == current_ante:
                marker = " [green](current)[/green]"
                default_preset = i
            console.print(f"  [dim]{i}.[/dim] {label}{detail}{marker}")

        preset_choice = IntPrompt.ask("[yellow]Select default preset[/yellow]", default=default_preset)
        if 1 <= preset_choice <= len(FPS_PRESETS):
            label, val = FPS_PRESETS[preset_choice - 1]
            if val is not None:
                config["hit_ante"] = val
                console.print(f"[green]> Preset: {label} (delay: {val}ms)[/green]\n")
            else:
                config["hit_ante"] = IntPrompt.ask(
                    "[yellow]Custom delay (ms, -300 to 50)[/yellow]",
                    default=current_ante
                )
                console.print(f"[green]> Custom delay: {config['hit_ante']}ms[/green]\n")
        else:
            config["hit_ante"] = current_ante

        # --- CPU Threads ---
        current_threads = existing.get("cpu_threads", 4) if existing else 4
        thread_map = {2: 1, 4: 2, 6: 3, 8: 4}
        default_thread_choice = thread_map.get(current_threads, 2)

        console.print("[bold cyan]Default CPU Thread Count:[/bold cyan]")
        console.print(f"  [dim]1.[/dim] Low (2 threads){' [green](current)[/green]' if current_threads == 2 else ''}")
        console.print(f"  [dim]2.[/dim] Normal (4 threads){' [green](current)[/green]' if current_threads == 4 else ''}")
        console.print(f"  [dim]3.[/dim] High (6 threads){' [green](current)[/green]' if current_threads == 6 else ''}")
        console.print(f"  [dim]4.[/dim] Maximum (8 threads){' [green](current)[/green]' if current_threads == 8 else ''}")
        thread_choice = IntPrompt.ask("[yellow]Select default[/yellow]", default=default_thread_choice)
        config["cpu_threads"] = {1: 2, 2: 4, 3: 6, 4: 8}.get(thread_choice, 4)
        console.print(f"[green]> Threads: {config['cpu_threads']}[/green]\n")

        # --- Humanizer Hesitation ---
        current_hes = existing.get("use_hesitation", True) if existing else True
        console.print("[bold cyan]Default Hesitation Setting:[/bold cyan]")
        console.print(f"  [dim]Current: {'Active' if current_hes else 'Disabled'}[/dim]")
        config["use_hesitation"] = Confirm.ask(
            "[yellow]Enable hesitation by default?[/yellow]",
            default=current_hes
        )
        console.print(f"[green]> Hesitation: {'Active' if config['use_hesitation'] else 'Disabled'}[/green]\n")

        # --- Summary & Save ---
        console.print(Panel("[bold]Default Settings Summary[/bold]", box=box.ROUNDED))

        summary = Table(box=box.SIMPLE, expand=True)
        summary.add_column("Setting", style="cyan")
        summary.add_column("Value", style="green")
        summary.add_row("Model", models[config["model_index"]][0])
        summary.add_row("Device", "GPU" if config["use_gpu"] else "CPU")
        summary.add_row("Capture Method", config["monitoring_type"])
        summary.add_row("Ante-frontier Delay", f"{config['hit_ante']}ms")
        summary.add_row("CPU Threads", str(config["cpu_threads"]))
        summary.add_row("Input Mode", ACTIVE_INPUT_MODE)
        summary.add_row("Hesitation", "Active" if config.get("use_hesitation", True) else "Disabled")
        console.print(summary)
        console.print()

        if Confirm.ask("[yellow]Save these defaults?[/yellow]", default=True):
            if save_config(config):
                console.print(f"[green]✓ Defaults saved to {CONFIG_PATH}[/green]")
                console.print("[dim]Use 'python tui.py -s' to quick-start with these defaults.[/dim]")
            else:
                console.print("[red]✗ Failed to save defaults.[/red]")
        else:
            console.print("[dim]Cancelled — defaults not saved.[/dim]")

    def create_status_display(self):
        """Build the live status display."""
        stats_table = Table(box=box.ROUNDED, expand=True)
        stats_table.add_column("Statistic", style="cyan")
        stats_table.add_column("Value", style="green")

        with self.lock:
            stats_table.add_row("FPS", f"{self.fps:.1f}")
            stats_table.add_row("Total Hits", str(self.total_hits))
            if self.session_start:
                elapsed = time() - self.session_start
                stats_table.add_row("Duration", f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}")
            if self.last_hit_desc:
                stats_table.add_row("Last Hit", self.last_hit_desc)

        probs_table = Table(box=box.ROUNDED, expand=True)
        probs_table.add_column("Class", style="cyan")
        probs_table.add_column("Probability", style="yellow")

        with self.lock:
            if self.last_probs:
                for label, prob in sorted(self.last_probs.items(), key=lambda x: x[1], reverse=True):
                    bar_len = int(prob * 20)
                    bar = "\u2588" * bar_len + "\u2591" * (20 - bar_len)
                    probs_table.add_row(label, f"{bar} {prob:.1%}")

        grid = Table.grid(expand=True)
        grid.add_column()
        grid.add_column()
        grid.add_row(
            Panel(stats_table, title="[bold]Stats[/bold]", border_style="blue"),
            Panel(probs_table, title="[bold]AI Predictions[/bold]", border_style="magenta")
        )

        status = "[bold green]\u25cf RUNNING[/bold green]" if self.running else "[bold red]\u25cf STOPPED[/bold red]"
        return Panel(
            grid,
            title=f"[bold cyan]DBD Auto Skill Check[/bold cyan] {status}",
            subtitle="[dim]Press Ctrl+C to stop[/dim]",
            border_style="cyan",
            box=box.DOUBLE
        )

    def create_monitoring(self):
        if self.monitoring_type.startswith("v4l2") and v4l2_ok:
            return Monitoring_v4l2(device_id=self.monitor_id, crop_size=224)
        elif self.monitoring_type == "bettercam" and bettercam_ok:
            return Monitoring_bettercam(monitor_id=self.monitor_id, crop_size=224, target_fps=240)
        else:
            return Monitoring_mss(monitor_id=self.monitor_id, crop_size=224)

    def ui_update_loop(self):
        """Background thread: update terminal UI."""
        try:
            with Live(self.create_status_display(), refresh_per_second=4, console=console) as live:
                while self.running:
                    live.update(self.create_status_display())
                    sleep(0.25)
        except Exception:
            pass

    def run(self, skip_config=False, enable_logging=False):
        """Main run function."""
        if enable_logging:
            log_file = setup_logging()
            console.print(f"[dim]Logging enabled: {log_file}[/dim]")
            logging.info(f"Session started. Platform: {self.platform_info['display']}")
            logging.info(f"Input Mode: {ACTIVE_INPUT_MODE}")

        if skip_config:
            self.clear_screen()
            self.show_banner()

            # Load saved config, or use platform defaults
            saved = load_config()
            if saved:
                self.apply_config(saved)
                console.print("[green]✓ Loaded saved defaults from config.json[/green]")
            else:
                console.print("[dim]No saved config — using platform defaults[/dim]")

            models = self.get_available_models()
            if not models:
                console.print("[red]No ONNX/TRT model found in 'models/' folder![/red]")
                return

            # If model_path not set by config, use first available
            if self.model_path is None:
                self.model_path = models[0][1]

            # Resolve monitor
            monitors = self.get_monitor_list(self.monitoring_type)
            if monitors and self.monitor_id == 0:
                self.monitor_id = monitors[0][1]

            # Determine model name for display
            model_name = os.path.basename(self.model_path)

            console.print(f"[dim]Quick start mode (-s)[/dim]")
            console.print(f"[dim]  Model     : {model_name}[/dim]")
            console.print(f"[dim]  Device    : {'GPU' if self.use_gpu else 'CPU'}[/dim]")
            console.print(f"[dim]  Capture   : {self.monitoring_type}[/dim]")
            console.print(f"[dim]  Delay     : {self.hit_ante}ms[/dim]")
            console.print(f"[dim]  Threads   : {self.cpu_threads}[/dim]")
            console.print(f"[dim]  Input Mode: {ACTIVE_INPUT_MODE}[/dim]")
            console.print(f"[dim]  Humanizer : {'Hesitation ON' if self.use_hesitation else 'Hesitation OFF'}[/dim]")
            console.print()
        else:
            if not self.select_settings():
                return
            console.print()
            if not Confirm.ask("[yellow]Start monitoring?[/yellow]", default=True):
                console.print("[dim]Cancelled.[/dim]")
                return
            self.clear_screen()
            self.show_banner()

        # Wait for consent thread if it's still running
        if self._consent_thread and self._consent_thread.is_alive():
            console.print("[dim]Waiting for consent check...[/dim]")
            self._consent_thread.join(timeout=3)

        # Load model
        console.print("[cyan]Loading AI model...[/cyan]")

        try:
            monitoring = self.create_monitoring()
            self.ai_model = AI_model(self.model_path, self.use_gpu, self.cpu_threads, monitoring)
            ep = self.ai_model.check_provider()

            if "CUDA" in ep:
                console.print("[green]Running on GPU (CUDA)[/green]")
            elif "Dml" in ep:
                console.print("[green]Running on GPU (DirectML)[/green]")
            elif "TensorRT" in ep:
                console.print("[green]Running on GPU (TensorRT)[/green]")
            else:
                console.print(f"[yellow]Running on CPU ({self.cpu_threads} threads)[/yellow]")
        except Exception as e:
            console.print(f"[red]Model loading error: {e}[/red]")
            return

        console.print("[green]Model loaded![/green]")
        console.print("[cyan]Starting monitoring...[/cyan]\n")

        self.running = True
        self.session_start = time()
        self.total_hits = 0

        # UI update in background thread
        ui_thread = threading.Thread(target=self.ui_update_loop, daemon=True)
        ui_thread.start()

        # Main monitoring loop (same thread as mss - mss is NOT thread-safe)
        t0 = time()
        nb_frames = 0

        try:
            while self.running:
                if self.ai_model is None:
                    break

                frame_np = self.ai_model.grab_screenshot()
                nb_frames += 1

                pred, desc, probs, should_hit = self.ai_model.predict(frame_np)
                with self.lock:
                    self.last_probs = probs

                if should_hit:
                    if pred == 2 and self.hit_ante > 0:
                        sleep(self.hit_ante * 0.001)

                    # Humanized key press
                    cooldown = self.humanizer.press(
                        SPACE, PressKey, ReleaseKey, use_hesitation=self.use_hesitation
                    )
                    
                    if enable_logging:
                        logging.info(f"HIT | Pred: {pred} | Desc: {desc} | Cooldown: {cooldown:.4f}s")

                    with self.lock:
                        self.total_hits += 1
                        self.last_hit_desc = desc

                    sleep(cooldown)
                    t0 = time()
                    nb_frames = 0
                    continue

                t_diff = time() - t0
                if t_diff > 1.0:
                    with self.lock:
                        self.fps = nb_frames / t_diff
                    t0 = time()
                    nb_frames = 0

        except KeyboardInterrupt:
            pass
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
        finally:
            self.running = False
            sleep(0.5)
            console.print("\n[yellow]Stopping...[/yellow]")
            if self.ai_model is not None:
                del self.ai_model
                self.ai_model = None
            # Close uinput device if active
            if sys.platform != "win32":
                try:
                    from dbd.utils.linux_uinput import close_controller
                    close_controller()
                except ImportError:
                    pass
            console.print("[green]Cleanup done.[/green]")
            if self.session_start:
                elapsed = time() - self.session_start
                console.print(f"\n[bold cyan]Session Summary:[/bold cyan]")
                console.print(f"  Duration: {int(elapsed // 60)}m {int(elapsed % 60)}s")
                console.print(f"  Total Hits: {self.total_hits}")


def main():
    parser = argparse.ArgumentParser(
        description="DBD Auto Skill Check - Terminal UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python tui.py          # Interactive settings\n"
               "  python tui.py -s       # Quick start with saved/platform defaults\n"
               "  python tui.py -d       # Edit & save default settings\n"
    )
    parser.add_argument("-s", "--skip", action="store_true",
                        help="Skip settings menu, start with saved defaults or platform defaults")
    parser.add_argument("-d", "--defaults", action="store_true",
                        help="Edit and save default settings to config.json")
    parser.add_argument("-l", "--log", action="store_true",
                        help="Enable logging to logs/ directory")
    args = parser.parse_args()

    app = DBDAutoSkillCheck()

    def signal_handler(sig, frame):
        app.running = False
    signal.signal(signal.SIGINT, signal_handler)

    try:
        if args.defaults:
            app.edit_defaults()
        else:
            app.run(skip_config=args.skip, enable_logging=args.log)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
