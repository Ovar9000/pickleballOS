"""Paddle-pop detection from audio.

A paddle hit is a short, loud, broadband transient: the "pop". It is the most reliable contact cue
we have, because it survives exactly the cases that break vision (the ball hidden behind a player,
motion blur at contact). It is also ambiguous on its own: neighbouring courts pop too, and a bounce
on a hard court makes a smaller click. Fusion with the ball track (fusion.py) resolves that.

Detection is causal apart from a 40 ms peak-picking window: band-pass (where the pop's energy is
and crowd and wind noise isn't), short-time log energy, and a peak must rise a given number of dB
above the median level of the preceding second.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfilt

SR = 22050


def load_audio(path: str, start: float = 0.0, end: float | None = None, sr: int = SR) -> np.ndarray:
    """Mono float32 samples of [start, end) from a media file, via ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is needed to read audio")
    cmd = [ffmpeg, "-v", "error", "-ss", f"{start}"]
    if end is not None:
        cmd += ["-t", f"{end - start}"]
    cmd += ["-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


@dataclass
class Onset:
    t: float  # seconds, in the same clock as the video
    rise_db: float  # how far the peak rose above the recent background
    hf_ratio_db: float  # high band (2-6 kHz) vs low band (150-800 Hz) energy at the peak


class OnsetDetector:
    def __init__(self, sr: int = SR, band=(1500.0, 6000.0), win_s: float = 0.010, hop_s: float = 0.0025,
                 min_rise_db: float = 12.0, background_s: float = 1.0, peak_window_s: float = 0.04):
        self.sr = sr
        self.win = int(win_s * sr)
        self.hop = int(hop_s * sr)
        self.min_rise_db = min_rise_db
        self.bg_frames = max(1, int(background_s / hop_s))
        self.peak_frames = max(1, int(peak_window_s / hop_s))
        self._band = butter(4, band, btype="bandpass", fs=sr, output="sos")
        self._hi = butter(4, (2000.0, 6000.0), btype="bandpass", fs=sr, output="sos")
        self._lo = butter(4, (150.0, 800.0), btype="bandpass", fs=sr, output="sos")

    def _log_energy(self, x: np.ndarray) -> np.ndarray:
        n = 1 + max(0, (len(x) - self.win) // self.hop)
        idx = np.arange(self.win)[None, :] + self.hop * np.arange(n)[:, None]
        frames = x[idx]
        return 10 * np.log10(np.mean(frames * frames, axis=1) + 1e-12)

    def detect(self, audio: np.ndarray, t0: float = 0.0) -> list[Onset]:
        if len(audio) < self.win:
            return []
        e = self._log_energy(sosfilt(self._band, audio))
        e_hi = self._log_energy(sosfilt(self._hi, audio))
        e_lo = self._log_energy(sosfilt(self._lo, audio))
        out: list[Onset] = []
        last = -10**9
        for n in range(len(e)):
            lo, hi = max(0, n - self.peak_frames), min(len(e), n + self.peak_frames + 1)
            if e[n] < e[lo:hi].max():
                continue  # not the local peak
            past = e[max(0, n - self.bg_frames):max(1, n - self.peak_frames)]
            rise = e[n] - float(np.median(past))
            if rise < self.min_rise_db or n - last < self.peak_frames * 2:
                continue
            last = n
            t = t0 + (n * self.hop + self.win / 2) / self.sr
            out.append(Onset(t, rise, float(e_hi[n] - e_lo[n])))
        return out
