import numpy as np
import sounddevice as sd

# --- Audio settings ---
SAMPLE_RATE = 44100
BLOCK_SIZE  = 1024
CHANNELS    = 1

EPS           = 1e-12




def _rms(x): return float(np.sqrt(np.mean(x*x) + EPS))
def _db_amp(x): return 20.0 * np.log10(max(x, EPS))
def _snr01(snr_db, snr_floor_db, snr_ceil_db): return float(np.clip(
    (snr_db - snr_floor_db) / (snr_ceil_db - snr_floor_db), 0.0, 1.0))


class AudioReactive:
    def __init__(self):
        # runtime state
        self.bands = {
            "bass":   (20, 200),
            "mid":    (200, 3000),
            "treble": (3000, 8000),
        }
        # --- Per-band gain multipliers ---
        self.band_gain = {
            "bass": 1.5,
            "mid": 4.0,
            "treble": 3.0
        }
        # --- Robustness params ---
        self.open_db = 4.0
        self.close_db = 3.0
        self.noise_lerp = 0.995
        self.snr_floor_db = 1.0
        self.snr_ceil_db = 20.0

        self.alpha_attack, self.alpha_release = 0.6, 0.9

        # --- Precompute FFT bins ---
        self.freqs = np.fft.rfftfreq(BLOCK_SIZE, 1.0 / SAMPLE_RATE)

        self.state  = {k: 0.0 for k in self.bands}
        self.values = {k: 0.0 for k in self.bands}
        self.audio_array = np.zeros(len(self.bands))  # ordered array
        self.stream = None

        # noise/gate state
        self.noise_rms        = 1e-3
        self.noise_band_pw = {"bass": 1e-4, "mid": 1e-5, "treble": 1e-6}
        self.gate_open        = False
        self.update_with_bands()

    def update_with_bands(self):
        self.band_idx = {k: np.where((self.freqs >= lo) & (self.freqs < hi))[0]
                         for k, (lo, hi) in self.bands.items()}

    def _callback(self, indata, frames, time_info, status):
        if status:
            return

        # mono + DC removal
        x = indata[:, 0].astype(np.float32)
        x -= np.mean(x)

        # FFT power
        w  = np.hanning(len(x))
        X  = np.fft.rfft(x * w)
        ps = (np.abs(X)**2) / (np.sum(w**2) + EPS)

        # Per-band mean power (log-compressed)
        band_pw = {
            k: (0.0 if len(idx) == 0 else float(np.log1p(np.mean(ps[idx]))))
            for k, idx in self.band_idx.items()
        }

        # Gate logic
        level_db    = _db_amp(_rms(x))
        baseline_db = _db_amp(self.noise_rms)
        if self.gate_open:
            if level_db < baseline_db + self.close_db:
                self.gate_open = False
        else:
            if level_db > baseline_db + self.open_db:
                self.gate_open = True

        # Per-band outputs
        arr = []
        for k, p in band_pw.items():
            last = self.state.get(k, 0.0)
            if not self.gate_open:
                self.noise_rms = self.noise_lerp * self.noise_rms + (1.0 - self.noise_lerp) * _rms(x)
                self.noise_band_pw[k] = self.noise_lerp * self.noise_band_pw[k] + (1.0 - self.noise_lerp) * p
                target = 0.0
            else:
                snr   = (p + EPS) / (self.noise_band_pw[k] + EPS)
                norm  = _snr01(10.0 * np.log10(snr), self.snr_floor_db, self.snr_ceil_db)
                target = np.clip(norm * self.band_gain[k], 0, 1)

            # smoothing
            alpha = self.alpha_attack if target > last else self.alpha_release
            smoothed = alpha * last + (1.0 - alpha) * target

            self.state[k]  = smoothed
            self.values[k] = smoothed
            arr.append(smoothed)

        self.audio_array = np.array(arr)

    def start(self):
        self.stream = sd.InputStream(
            callback=self._callback,
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE
        )
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
