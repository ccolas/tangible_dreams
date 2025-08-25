import numpy as np
import sounddevice as sd

# --- Audio settings ---
SAMPLE_RATE = 44100
BLOCK_SIZE  = 1024
CHANNELS    = 1

BANDS = {
    "bass":   (20, 200),
    "mid":    (200, 3000),
    "treble": (3000, 8000),
}

# --- Per-band gain multipliers ---
factor = 0.5
BAND_GAIN = {
    "bass":   3.0 * factor,
    "mid":    8.0 * factor,
    "treble": 6.0 * factor,
}

# --- Robustness params ---
WARMUP_SEC    = 0.6
OPEN_DB       = 4.0
CLOSE_DB      = 3.0
NOISE_LERP    = 0.995
SNR_FLOOR_DB  = 1.0
SNR_CEIL_DB   = 20.0
EPS           = 1e-12

# --- Precompute FFT bins ---
FREQS = np.fft.rfftfreq(BLOCK_SIZE, 1.0 / SAMPLE_RATE)
BAND_IDXS = {k: np.where((FREQS >= lo) & (FREQS < hi))[0]
             for k,(lo,hi) in BANDS.items()}


def _rms(x): return float(np.sqrt(np.mean(x*x) + EPS))
def _db_amp(x): return 20.0 * np.log10(max(x, EPS))
def _snr01(snr_db): return float(np.clip(
    (snr_db - SNR_FLOOR_DB) / (SNR_CEIL_DB - SNR_FLOOR_DB), 0.0, 1.0))


class AudioReactive:
    def __init__(self):
        # runtime state
        self.state  = {k: 0.0 for k in BANDS}
        self.values = {k: 0.0 for k in BANDS}
        self.audio_array = np.zeros(len(BANDS))  # ordered array
        self.stream = None

        # noise/gate state
        self.init_blocks      = max(1, int(WARMUP_SEC * SAMPLE_RATE / BLOCK_SIZE))
        self.noise_rms        = 1e-6
        self.noise_band_pw    = {k: 1e-9 for k in BANDS}
        self.gate_open        = False
        self.warm_rms_samples = []
        self.warm_band_pw     = {k: [] for k in BANDS}

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
            for k, idx in BAND_IDXS.items()
        }

        # Warmup
        if self.init_blocks > 0:
            self.init_blocks -= 1
            self.warm_rms_samples.append(_rms(x))
            for k in band_pw:
                self.warm_band_pw[k].append(band_pw[k])
            for k in BANDS:
                self.values[k] = 0.0
            if self.init_blocks == 0:
                import numpy as _np
                self.noise_rms = max(1e-9, float(_np.median(self.warm_rms_samples)))
                for k in BANDS:
                    arr = self.warm_band_pw[k]
                    self.noise_band_pw[k] = max(1e-12, float(_np.percentile(arr, 30)))
            return

        # Gate logic
        level_db    = _db_amp(_rms(x))
        baseline_db = _db_amp(self.noise_rms)
        if self.gate_open:
            if level_db < baseline_db + CLOSE_DB:
                self.gate_open = False
        else:
            if level_db > baseline_db + OPEN_DB:
                self.gate_open = True

        # Per-band outputs
        arr = []
        for k, p in band_pw.items():
            last = self.state.get(k, 0.0)
            if not self.gate_open:
                self.noise_rms = NOISE_LERP * self.noise_rms + (1.0 - NOISE_LERP) * _rms(x)
                self.noise_band_pw[k] = NOISE_LERP * self.noise_band_pw[k] + (1.0 - NOISE_LERP) * p
                target = 0.0
            else:
                snr   = (p + EPS) / (self.noise_band_pw[k] + EPS)
                norm  = _snr01(10.0 * np.log10(snr))
                target = np.clip(norm * BAND_GAIN[k], 0, 1)

            # smoothing
            alpha_attack, alpha_release = 0.6, 0.9
            alpha = alpha_attack if target > last else alpha_release
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
