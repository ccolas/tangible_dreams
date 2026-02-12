import numpy as np
import sounddevice as sd

# --- Audio settings ---
SAMPLE_RATE = 44100
BLOCK_SIZE  = 1024   # one block ~23 ms; you can reduce to 512 later
EPS         = 1e-12

# --- Shared memory layout for visualizer process ---
_PS_LEN        = BLOCK_SIZE // 2 + 1
_SHM_AUDIO     = 0       # 6 doubles: bass_ema, bass_flux, mid_ema, mid_flux, treble_ema, treble_flux
_SHM_RMS_DB    = 6       # 1 double
_SHM_GAIN      = 7       # 3 doubles: bass, mid, treble
_SHM_EMA_BASE  = 10      # 3 doubles: bass, mid, treble EMA baselines (dB, for viz)
_SHM_BASS_HI   = 13      # 1 double: bass/mid crossover freq
_SHM_MID_HI    = 14      # 1 double: mid/treble crossover freq
_SHM_PS_VALID  = 15      # 1 double (0 or 1)
_SHM_PS        = 16      # _PS_LEN doubles
_SHM_SIZE      = 16 + _PS_LEN
_SHM_DTYPE     = np.float64
_SHM_NBYTES    = _SHM_SIZE * np.dtype(_SHM_DTYPE).itemsize


def _rms(x): return float(np.sqrt(np.mean(x*x) + EPS))
def _db_amp(x): return 20.0 * np.log10(max(x, EPS))


def _viz_process_main(shm_name, bands):
    """Visualizer in a separate process — owns its own GIL."""
    from multiprocessing.shared_memory import SharedMemory
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    shm_mem = SharedMemory(name=shm_name)
    shm = np.ndarray((_SHM_SIZE,), dtype=_SHM_DTYPE, buffer=shm_mem.buf)
    freqs = np.fft.rfftfreq(BLOCK_SIZE, 1.0 / SAMPLE_RATE)

    fig, (ax_circles, ax_bands, ax_ema_base, ax_gains, ax_spec) = plt.subplots(
        5, 1,
        figsize=(7, 12),
        gridspec_kw={"height_ratios": [0.6, 1.6, 1.6, 0.8, 1.2]},
    )

    # ---------- CIRCLES (6: EMA + flux per band) ----------
    ax_circles.set_xlim(-0.5, 5.5)
    ax_circles.set_ylim(0.0, 1.0)
    ax_circles.set_aspect("equal")
    ax_circles.axis("off")

    circle_xs     = [0, 1, 2, 3, 4, 5]
    circle_colors = ["blue", "blue", "green", "green", "red", "red"]
    circle_alphas = [0.6, 0.3, 0.6, 0.3, 0.6, 0.3]  # solid=EMA, faint=flux
    circles = [
        plt.Circle((circle_xs[i], 0.5), radius=0.05,
                    color=circle_colors[i], alpha=circle_alphas[i])
        for i in range(6)
    ]
    for c in circles:
        ax_circles.add_patch(c)

    # ---------- AUDIO SIGNALS (6 lines: EMA solid, flux dashed) ----------
    ax_bands.set_title("Audio Signals (0\u20131)")
    ax_bands.set_ylim(0, 1.05)
    MAX_HIST = 200
    ax_bands.set_xlim(0, MAX_HIST)

    hists = [[0.0] * MAX_HIST for _ in range(6)]
    labels = ["bass ema", "bass flux", "mid ema", "mid flux", "treble ema", "treble flux"]
    colors = ["blue", "blue", "green", "green", "red", "red"]
    styles = ["-", "--", "-", "--", "-", "--"]
    lines = []
    for lb, co, st in zip(labels, colors, styles):
        (ln,) = ax_bands.plot([], [], color=co, linestyle=st, label=lb)
        lines.append(ln)
    ax_bands.legend(loc="upper right", fontsize=7, ncol=2)

    # ---------- EMA BASELINES + RMS ----------
    ax_ema_base.set_title("EMA Baselines (dB) + RMS")
    ax_ema_base.set_ylabel("dB")
    ax_ema_base.set_ylim(-80, 0)
    ax_ema_base.set_xlim(0, MAX_HIST)

    rms_hist = [-60.0] * MAX_HIST
    ema_base_hists = [[-80.0] * MAX_HIST for _ in range(3)]

    (line_rms,) = ax_ema_base.plot([], [], color="black", label="RMS")
    (line_ema_bass,) = ax_ema_base.plot([], [], color="blue", linestyle="--", label="bass EMA")
    (line_ema_mid,) = ax_ema_base.plot([], [], color="green", linestyle="--", label="mid EMA")
    (line_ema_treb,) = ax_ema_base.plot([], [], color="red", linestyle="--", label="treble EMA")
    ax_ema_base.legend(loc="upper right", fontsize=7)

    # ---------- BAND GAINS ----------
    ax_gains.set_title("Band Gains")
    ax_gains.set_ylim(0, 8.0)
    gain_bars = ax_gains.bar(
        ["bass", "mid", "treble"],
        [0, 0, 0],
        color=["blue", "green", "red"],
    )

    # ---------- SPECTRUM PANEL ----------
    ax_spec.set_title("Frequency Spectrum (dB)")
    ax_spec.set_ylim(-100, 20)
    ax_spec.set_xlim(20, 8000)
    ax_spec.set_xscale("log")
    ax_spec.set_xlabel("Hz")

    (freq_line,) = ax_spec.plot([], [], color="black", linewidth=1.0)
    (bass_split_line,) = ax_spec.plot([], [], color="blue", linestyle="--")
    (mid_split_line,) = ax_spec.plot([], [], color="green", linestyle="--")

    hist_idx = [0]
    x_axis = np.arange(MAX_HIST)

    def update(frame):
        vals = shm[_SHM_AUDIO:_SHM_AUDIO + 6].copy()
        rms_db = float(shm[_SHM_RMS_DB])
        gains = shm[_SHM_GAIN:_SHM_GAIN + 3].copy()
        ema_bases = shm[_SHM_EMA_BASE:_SHM_EMA_BASE + 3].copy()

        # circles
        for c, v in zip(circles, vals):
            c.set_radius(0.05 + 0.4 * float(v))

        # histories
        idx = hist_idx[0]
        for i in range(6):
            hists[i][idx] = vals[i]
        rms_hist[idx] = rms_db
        for i in range(3):
            ema_base_hists[i][idx] = ema_bases[i]
        hist_idx[0] = (idx + 1) % MAX_HIST

        # audio signal lines
        for ln, h in zip(lines, hists):
            ln.set_data(x_axis, h)

        # EMA baseline + RMS lines
        line_rms.set_data(x_axis, rms_hist)
        line_ema_bass.set_data(x_axis, ema_base_hists[0])
        line_ema_mid.set_data(x_axis, ema_base_hists[1])
        line_ema_treb.set_data(x_axis, ema_base_hists[2])

        # gain bars
        for bar, g in zip(gain_bars, gains):
            bar.set_height(g)

        # spectrum
        if shm[_SHM_PS_VALID] > 0.5:
            ps = shm[_SHM_PS:_SHM_PS + _PS_LEN].copy()
            spec_db = 10 * np.log10(ps + EPS)
            freq_line.set_data(freqs, spec_db)

            bass_hi = float(shm[_SHM_BASS_HI])
            mid_hi  = float(shm[_SHM_MID_HI])
            bass_split_line.set_data([bass_hi, bass_hi], [-100, 20])
            mid_split_line.set_data([mid_hi, mid_hi], [-100, 20])

    ani = animation.FuncAnimation(fig, update, interval=100, blit=False)
    plt.tight_layout()
    plt.show()
    shm_mem.close()


class AudioReactive:
    def __init__(self, visualize=False, device_in=None, device_out=None, channels=2):
        """
        device_in / device_out:
          - None  -> default system devices
          - int   -> sounddevice device index
          - str   -> device name (e.g. 'visual_sink.monitor', 'Built-in Audio')
        channels:
          - 2 for stereo Spotify pass-through (recommended)
        """
        self.visualize = visualize
        self.device_in = device_in
        self.device_out = device_out
        self.channels = channels

        # runtime state
        self.bands = {
            "bass":   (20, 200),
            "mid":    (200, 3000),
            "treble": (3000, 8000),
        }
        # --- Per-band gain multipliers ---
        self.band_gain = {
            "bass": 1.0,
            "mid":  1.0,
            "treble": 1.0,
        }

        # --- Precompute FFT bins ---
        self.freqs = np.fft.rfftfreq(BLOCK_SIZE, 1.0 / SAMPLE_RATE)

        self.audio_array = np.zeros(6)  # [bass_ema, bass_flux, mid_ema, mid_flux, treble_ema, treble_flux]
        self.stream = None

        # --- EMA + spectral flux ---
        self.ema_sensitivity = 1.0      # slider 3: scales EMA deviation signal
        self.flux_sensitivity = 0.5     # slider 4: scales spectral flux signal
        self.ema_alpha = 0.995          # baseline tracking speed (~5s time constant)

        self._ema = {k: EPS for k in self.bands}
        self._prev_pw = {k: EPS for k in self.bands}
        self._ema_smooth = {k: 0.0 for k in self.bands}
        self._flux_smooth = {k: 0.0 for k in self.bands}

        # for viz
        self.last_ps = None
        self.last_rms_db = -60.0

        # --- delay control ---
        self.delay_seconds = 0.0  # user sets this (e.g. 0.05 for 50 ms)
        self.max_delay_seconds = 0.7  # max 700ms
        self.max_delay_frames = int(self.max_delay_seconds * SAMPLE_RATE)

        # stereo buffer: shape (max_frames, channels)
        self.delay_buffer = np.zeros((self.max_delay_frames, self.channels), dtype=np.float32)
        self.delay_write_pos = 0

        # --- Shared memory for visualizer process ---
        self._shm_mem = None
        self._shm_np = None
        self._viz_proc = None

        self.update_with_bands()
        if self.visualize:
            self._setup_shared_memory()
            self._start_visualization()

    def _setup_shared_memory(self):
        from multiprocessing.shared_memory import SharedMemory
        self._shm_mem = SharedMemory(create=True, size=_SHM_NBYTES)
        self._shm_np = np.ndarray((_SHM_SIZE,), dtype=_SHM_DTYPE,
                                  buffer=self._shm_mem.buf)
        self._shm_np[:] = 0.0
        self._shm_np[_SHM_RMS_DB] = -60.0
        self._shm_np[_SHM_GAIN:_SHM_GAIN + 3] = [
            self.band_gain[k] for k in self.bands
        ]
        self._shm_np[_SHM_EMA_BASE:_SHM_EMA_BASE + 3] = -80.0
        self._shm_np[_SHM_BASS_HI] = self.bands["bass"][1]
        self._shm_np[_SHM_MID_HI] = self.bands["mid"][1]

    def update_with_bands(self):
        self.band_idx = {
            k: np.where((self.freqs >= lo) & (self.freqs < hi))[0]
            for k, (lo, hi) in self.bands.items()
        }

    # FULL-DUPLEX CALLBACK: indata -> analysis, then copy to outdata
    def _callback(self, indata, outdata, frames, time_info, status):
        if status:
            pass

        # ===== DELAYED OUTPUT =====
        delay_frames = int(self.delay_seconds * SAMPLE_RATE)
        delay_frames = min(delay_frames, self.max_delay_frames - 1)

        buf = self.delay_buffer
        pos = self.delay_write_pos

        block = indata.copy()
        n = len(block)

        # write into buffer (wrap around)
        end = pos + n
        if end <= self.max_delay_frames:
            buf[pos:end] = block
        else:
            split = self.max_delay_frames - pos
            buf[pos:] = block[:split]
            buf[:n - split] = block[split:]

        # read delayed block
        read_pos = pos - delay_frames
        if read_pos < 0:
            read_pos += self.max_delay_frames

        end_r = read_pos + n
        if end_r <= self.max_delay_frames:
            out_block = buf[read_pos:end_r]
        else:
            split_r = self.max_delay_frames - read_pos
            out_block = np.vstack((buf[read_pos:], buf[:n - split_r]))

        outdata[:] = out_block
        self.delay_write_pos = (pos + n) % self.max_delay_frames

        # ---- analysis (mono mixdown) ----
        x = indata.astype(np.float32)
        if x.ndim == 2 and x.shape[1] > 1:
            x = x.mean(axis=1)
        else:
            x = x[:, 0] if x.ndim == 2 else x

        x -= np.mean(x)  # DC removal

        # FFT power
        w  = np.hanning(len(x))
        X  = np.fft.rfft(x * w)
        ps = (np.abs(X)**2) / (np.sum(w**2) + EPS)
        self.last_ps = ps

        # RMS for viz
        level_db = _db_amp(_rms(x))
        self.last_rms_db = level_db

        # ---- Per-band: raw power, EMA baseline, flux ----
        arr = []
        ema_bases_db = []
        for k, idx_arr in self.band_idx.items():
            pw = float(np.mean(ps[idx_arr])) + EPS if len(idx_arr) > 0 else EPS

            # Update EMA baseline (slow-moving average of raw power)
            ema = self._ema[k]
            ema = self.ema_alpha * ema + (1.0 - self.ema_alpha) * pw
            self._ema[k] = ema

            # EMA deviation: how far above baseline (ratio - 1)
            ema_dev = max(0.0, pw / ema - 1.0)

            # Spectral flux: positive change, normalized by baseline
            flux = max(0.0, pw - self._prev_pw[k]) / (ema + EPS)
            self._prev_pw[k] = pw

            # Apply sensitivity + per-band gain, clip to [0, 1]
            ema_target = min(1.0, ema_dev * self.ema_sensitivity * self.band_gain[k])
            flux_target = min(1.0, flux * self.flux_sensitivity * self.band_gain[k])

            # Smoothing — EMA: medium attack, slow release (pumping)
            ema_last = self._ema_smooth[k]
            a = 0.3 if ema_target > ema_last else 0.93
            ema_out = a * ema_last + (1.0 - a) * ema_target

            # Smoothing — flux: fast attack, fast release (transient)
            flux_last = self._flux_smooth[k]
            a = 0.15 if flux_target > flux_last else 0.6
            flux_out = a * flux_last + (1.0 - a) * flux_target

            self._ema_smooth[k] = ema_out
            self._flux_smooth[k] = flux_out

            arr.extend([ema_out, flux_out])
            ema_bases_db.append(10.0 * np.log10(ema + EPS))

        self.audio_array = np.array(arr, dtype=np.float32)
        # [bass_ema, bass_flux, mid_ema, mid_flux, treble_ema, treble_flux]

        # ---- Push to shared memory for visualizer ----
        shm = self._shm_np
        if shm is not None:
            shm[_SHM_AUDIO:_SHM_AUDIO + 6] = arr
            shm[_SHM_RMS_DB] = level_db
            shm[_SHM_GAIN:_SHM_GAIN + 3] = [
                self.band_gain[k] for k in self.bands
            ]
            shm[_SHM_EMA_BASE:_SHM_EMA_BASE + 3] = ema_bases_db
            shm[_SHM_BASS_HI] = self.bands["bass"][1]
            shm[_SHM_MID_HI] = self.bands["mid"][1]
            shm[_SHM_PS:_SHM_PS + len(ps)] = ps
            shm[_SHM_PS_VALID] = 1.0

    def _start_visualization(self):
        from multiprocessing import Process
        self._viz_proc = Process(
            target=_viz_process_main,
            args=(self._shm_mem.name, dict(self.bands)),
            daemon=True,
        )
        self._viz_proc.start()

    def start(self):
        self.stream = sd.Stream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=self.channels,
            dtype='float32',
            callback=self._callback,
            device=(self.device_in, self.device_out),
        )
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if self._viz_proc is not None:
            self._viz_proc.terminate()
            self._viz_proc.join(timeout=2)
            self._viz_proc = None
        if self._shm_mem is not None:
            self._shm_mem.close()
            self._shm_mem.unlink()
            self._shm_mem = None
            self._shm_np = None


if __name__ == '__main__':
    audio = AudioReactive(
        visualize=True,
        device_in="pulse",
        device_out="pulse",
        channels=2
    )
    audio.start()

    import time

    while True:
        time.sleep(1)
