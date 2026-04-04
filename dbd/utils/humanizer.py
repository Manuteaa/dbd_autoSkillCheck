# humanizer.py
# Human-like key press simulation with per-installation fingerprinting.
#
# On first run, generates a unique timing fingerprint (humanizer_fingerprint.json)
# with randomized-but-reasonable parameters. Every installation gets slightly
# different timing characteristics, preventing anti-cheat from clustering users.
#
# Non-blocking design:
#   press() starts a background thread for the actual key press and returns
#   immediately with the recommended cooldown duration. The caller can continue
#   processing frames while the key press happens in the background.
#
# Fatigue & Hesitation:
#   - Fatigue: Gradually increases reaction time over long sessions (up to ~22%).
#     This simulates human tiredness but may reduce precision in competitive play.
#   - Hesitation: ~7% chance of adding 15-65ms micro-delay before pressing.
#     Increases realism but may cause "Good" instead of "Great" hits.
#   - For maximum precision, set use_hesitation=False when calling press().

import json
import random
import math
import time
import threading
import uuid
from pathlib import Path
from typing import Callable, Optional

# Fingerprint file lives next to the script (project root level)
_FINGERPRINT_DIR = Path(__file__).resolve().parent.parent.parent
_FINGERPRINT_PATH = _FINGERPRINT_DIR / "humanizer_fingerprint.json"


def _generate_fingerprint() -> dict:
    """Generate a unique timing fingerprint with randomized parameters.

    Each value is randomly chosen within a plausible human range.
    Two different installations will almost certainly get different values,
    making them look like two different humans to pattern analysis.
    """
    rng = random.SystemRandom()  # cryptographic RNG for uniqueness

    fp = {
        # Unique ID for this installation
        "id": uuid.uuid4().hex[:12],

        # Press duration (ex-Gaussian model)
        "press_mu": round(rng.uniform(0.165, 0.195), 4),       # 165-195ms center
        "press_sigma": round(rng.uniform(0.014, 0.024), 4),    # spread
        "press_exp_mean": round(rng.uniform(0.008, 0.018), 4), # exponential tail
        "press_min": round(rng.uniform(0.120, 0.140), 4),      # hard floor
        "press_max": round(rng.uniform(0.260, 0.300), 4),      # hard ceiling

        # Pre-press micro-delay
        "pre_delay_mu": round(rng.uniform(0.005, 0.010), 4),
        "pre_delay_sigma": round(rng.uniform(0.003, 0.006), 4),
        "pre_delay_max": round(rng.uniform(0.025, 0.040), 4),

        # Post-hit cooldown
        "cooldown_mu": round(rng.uniform(0.460, 0.510), 4),
        "cooldown_sigma": round(rng.uniform(0.030, 0.055), 4),
        "cooldown_exp_mean": round(rng.uniform(0.010, 0.022), 4),
        "cooldown_min": round(rng.uniform(0.360, 0.400), 4),
        "cooldown_max": round(rng.uniform(0.620, 0.680), 4),

        # Hesitation
        "hesitation_chance": round(rng.uniform(0.04, 0.10), 3),
        "hesitation_min": round(rng.uniform(0.012, 0.020), 4),
        "hesitation_max": round(rng.uniform(0.040, 0.065), 4),

        # Anti-repeat jitter threshold
        "anti_repeat_ms": round(rng.uniform(0.002, 0.005), 4),

        # Fatigue
        "fatigue_onset": rng.randint(15, 30),
        "fatigue_ramp": rng.randint(45, 75),
        "fatigue_max": round(rng.uniform(1.10, 1.22), 3),
        "fatigue_wave_amp": round(rng.uniform(0.015, 0.030), 4),
        "fatigue_wave_freq": round(rng.uniform(0.4, 0.8), 2),

        # Double-tap guard
        "min_inter_press": round(rng.uniform(0.320, 0.380), 4),
    }

    return fp


def load_fingerprint() -> dict:
    """Load fingerprint from disk, or generate and save a new one."""
    if _FINGERPRINT_PATH.exists():
        try:
            with open(_FINGERPRINT_PATH, "r") as f:
                fp = json.load(f)
            # Validate it has the expected keys
            if "id" in fp and "press_mu" in fp:
                return fp
        except (json.JSONDecodeError, IOError, KeyError):
            pass

    # First run — generate unique fingerprint
    fp = _generate_fingerprint()
    try:
        with open(_FINGERPRINT_PATH, "w") as f:
            json.dump(fp, f, indent=2)
    except IOError:
        pass  # still usable in memory even if save fails

    return fp


class Humanizer:
    """Human-like key press simulator with per-installation fingerprint.

    Non-blocking design:
        press() starts a background thread and returns immediately with the
        recommended cooldown. The caller can sleep(cooldown) while frames
        continue to be processed.

    Usage:
        from dbd.utils.humanizer import Humanizer
        from dbd.utils.directkeys import PressKey, ReleaseKey, SPACE

        humanizer = Humanizer()
        cooldown = humanizer.press(SPACE, PressKey, ReleaseKey)
        time.sleep(cooldown)  # Caller handles the cooldown sleep
    """

    def __init__(self, fingerprint: Optional[dict] = None):
        self._fp = fingerprint or load_fingerprint()
        self._hit_count: int = 0
        self._session_start: float = time.monotonic()
        self._last_press_time: float = 0.0
        self._lock = threading.Lock()
        self._recent_durations: list[float] = []
        self._max_recent: int = 6

    @property
    def fingerprint_id(self) -> str:
        return self._fp.get("id", "unknown")

    def reset(self):
        """Reset session state."""
        with self._lock:
            self._hit_count = 0
            self._session_start = time.monotonic()
            self._last_press_time = 0.0
            self._recent_durations.clear()

    @property
    def hit_count(self) -> int:
        return self._hit_count

    def _human_duration(self) -> float:
        """Generate a human-like press duration (seconds)."""
        fp = self._fp
        gaussian_part = random.gauss(fp["press_mu"], fp["press_sigma"])
        exp_tail = random.expovariate(1.0 / fp["press_exp_mean"])
        raw = gaussian_part + exp_tail
        duration = max(fp["press_min"], min(fp["press_max"], raw))
        return self._anti_repeat_jitter(duration)

    def _human_cooldown(self) -> float:
        """Post-hit cooldown duration (seconds)."""
        fp = self._fp
        base = random.gauss(fp["cooldown_mu"], fp["cooldown_sigma"])
        jitter = random.expovariate(1.0 / fp["cooldown_exp_mean"])
        return max(fp["cooldown_min"], min(fp["cooldown_max"], base + jitter))

    def _pre_press_delay(self) -> float:
        """Micro-delay before key press (seconds)."""
        fp = self._fp
        return max(0.0, min(fp["pre_delay_max"], random.gauss(fp["pre_delay_mu"], fp["pre_delay_sigma"])))

    def _anti_repeat_jitter(self, duration: float) -> float:
        """Prevent consecutive identical durations.
        
        Thread-safe: reads recent_durations under lock, then applies jitter.
        Result is clamped to [press_min, press_max].
        """
        fp = self._fp
        threshold = fp["anti_repeat_ms"]
        
        # Read recent durations under lock
        with self._lock:
            recent_durations = list(self._recent_durations)
        
        for recent in recent_durations:
            if abs(duration - recent) < threshold:
                shift = random.uniform(threshold, threshold * 2.5)
                if random.random() < 0.5:
                    shift = -shift
                duration += shift
                break
        
        # Clamp to valid bounds
        duration = max(fp["press_min"], min(fp["press_max"], duration))
        return duration

    def _fatigue_factor(self) -> float:
        """Gradual fatigue multiplier over long sessions.
        
        Note: Fatigue slows reaction times by up to 22% over long sessions.
        This simulates human tiredness but may reduce precision.
        For competitive play, consider disabling or reducing fatigue.
        """
        fp = self._fp
        hits = self._hit_count
        if hits < fp["fatigue_onset"]:
            return 1.0
        progress = min(1.0, (hits - fp["fatigue_onset"]) / fp["fatigue_ramp"])
        base_fatigue = 1.0 + (fp["fatigue_max"] - 1.0) * progress
        wave = fp["fatigue_wave_amp"] * math.sin(hits * fp["fatigue_wave_freq"])
        return max(1.0, base_fatigue + wave)

    def _maybe_hesitate(self) -> float:
        """Random micro-hesitation before pressing.
        
        Note: ~7% chance of adding 15-65ms delay. Increases realism but
        may cause "Good" instead of "Great" hits. Set use_hesitation=False
        in press() to disable.
        """
        fp = self._fp
        if random.random() < fp["hesitation_chance"]:
            return random.uniform(fp["hesitation_min"], fp["hesitation_max"])
        return 0.0

    def _do_press_blocking(self, key_code, press_fn: Callable, release_fn: Callable,
                           use_hesitation: bool = True) -> tuple[float, float]:
        """Perform the actual key press with delays (runs in background thread).
        
        Returns:
            (press_duration, cooldown) tuple for tracking.
        """
        fatigue = self._fatigue_factor()

        pre_delay = self._pre_press_delay() * fatigue
        press_dur = self._human_duration() * fatigue
        hesitation = self._maybe_hesitate() if use_hesitation else 0.0
        cooldown = self._human_cooldown() * fatigue

        # Wait for inter-press guard
        with self._lock:
            now = time.monotonic()
            since_last = now - self._last_press_time
            guard = self._fp["min_inter_press"]
            if since_last < guard and self._last_press_time > 0:
                wait_guard = guard - since_last
                # Release lock while sleeping
                self._lock.release()
                try:
                    time.sleep(wait_guard)
                finally:
                    self._lock.acquire()

        # Pre-delay + hesitation
        wait = pre_delay + hesitation
        if wait > 0.001:
            time.sleep(wait)

        # Press and hold
        press_fn(key_code)
        time.sleep(press_dur)
        release_fn(key_code)

        # Update state under lock
        with self._lock:
            self._hit_count += 1
            self._last_press_time = time.monotonic()
            self._recent_durations.append(press_dur)
            if len(self._recent_durations) > self._max_recent:
                self._recent_durations.pop(0)

        return press_dur, cooldown

    def press(self, key_code, press_fn: Callable, release_fn: Callable,
              use_hesitation: bool = True) -> float:
        """Perform a human-like key press (non-blocking).

        Starts a background thread for the actual key press and returns
        immediately with the recommended cooldown duration. The caller
        should sleep for the returned cooldown while continuing to process
        frames.

        Args:
            key_code: Key to press (e.g. SPACE)
            press_fn: PressKey function
            release_fn: ReleaseKey function
            use_hesitation: If True, allows random micro-hesitations (more realistic).
                            If False, disables hesitations (maximum precision).

        Returns:
            Recommended cooldown duration in seconds.
        """
        # Calculate cooldown immediately so caller can sleep
        fatigue = self._fatigue_factor()
        cooldown = self._human_cooldown() * fatigue

        # Start background thread for the actual press
        thread = threading.Thread(
            target=self._do_press_blocking,
            args=(key_code, press_fn, release_fn, use_hesitation),
            daemon=True,
        )
        thread.start()

        return cooldown

    def __repr__(self):
        return (
            f"Humanizer(id={self.fingerprint_id}, "
            f"hits={self._hit_count}, "
            f"fatigue={self._fatigue_factor():.2f}x)"
        )


_instance: Optional[Humanizer] = None
_instance_lock = threading.Lock()


def get_humanizer() -> Humanizer:
    """Global singleton Humanizer instance."""
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = Humanizer()
        return _instance


def humanized_press(key_code, press_fn: Callable, release_fn: Callable, use_hesitation: bool = True) -> float:
    """One-liner human-like key press. Returns cooldown in seconds (non-blocking)."""
    return get_humanizer().press(key_code, press_fn, release_fn, use_hesitation)


if __name__ == "__main__":
    fp = load_fingerprint()
    print("Humanizer — Fingerprint & Timing Test")
    print("=" * 60)
    print(f"\n  Fingerprint ID : {fp['id']}")
    print(f"  Config path    : {_FINGERPRINT_PATH}")
    print(f"\n  Key parameters:")
    print(f"    press_mu       = {fp['press_mu']*1000:.1f} ms")
    print(f"    press_sigma    = {fp['press_sigma']*1000:.1f} ms")
    print(f"    press_exp_mean = {fp['press_exp_mean']*1000:.1f} ms")
    print(f"    cooldown_mu    = {fp['cooldown_mu']*1000:.1f} ms")
    print(f"    fatigue_onset  = {fp['fatigue_onset']} hits")
    print(f"    fatigue_max    = {fp['fatigue_max']:.3f}x")
    print(f"    hesitation     = {fp['hesitation_chance']*100:.0f}%")

    h = Humanizer(fingerprint=fp)
    durations = []
    cooldowns = []
    for _ in range(200):
        dur = h._human_duration()
        cd = h._human_cooldown()
        h._hit_count += 1
        durations.append(dur * 1000)
        cooldowns.append(cd * 1000)

    def show_stats(values, label):
        avg = sum(values) / len(values)
        mn, mx = min(values), max(values)
        sv = sorted(values)
        p5 = sv[int(len(sv) * 0.05)]
        p95 = sv[int(len(sv) * 0.95)]
        p25 = sv[int(len(sv) * 0.25)]
        p75 = sv[int(len(sv) * 0.75)]
        print(f"\n  {label}:")
        print(f"    Mean     : {avg:.1f} ms")
        print(f"    Min/Max  : {mn:.1f} / {mx:.1f} ms")
        print(f"    P5-P95   : {p5:.1f} - {p95:.1f} ms")
        print(f"    P25-P75  : {p25:.1f} - {p75:.1f} ms")

    show_stats(durations, "Press Duration")
    show_stats(cooldowns, "Cooldown")

    # Simulate 3 different installations
    print(f"\n\n  Simulating 3 different installations:")
    print(f"  " + "-" * 55)
    for i in range(3):
        sim_fp = _generate_fingerprint()
        sim_h = Humanizer(fingerprint=sim_fp)
        durs = [sim_h._human_duration() * 1000 for _ in range(100)]
        avg = sum(durs) / len(durs)
        print(f"    Install {sim_fp['id']}: press_mu={sim_fp['press_mu']*1000:.1f}ms  actual_avg={avg:.1f}ms")
