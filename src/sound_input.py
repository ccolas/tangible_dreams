import atexit
import numpy as np
import sounddevice as sd

# Force-reinitialize PortAudio to clear stale state from a previous crash
sd._terminate()
sd._initialize()

# --- Audio settings ---
SAMPLE_RATE = 44100
BLOCK_SIZE  = 1024   # one block ~23 ms (need >=1024 for bass FFT resolution)
EPS         = 1e-12


def _rms(x): return float(np.sqrt(np.mean(x*x) + EPS))
def _db_amp(x): return 20.0 * np.log10(max(x, EPS))
def _snr01(snr_db, snr_floor_db, snr_ceil_db): return float(np.clip(
    (snr_db - snr_floor_db) / (snr_ceil_db - snr_floor_db), 0.0, 1.0))


class AudioGate:
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
            "bass":   (30, 150),
            "mid":    (150, 3000),
            "treble": (3000, 8000),
        }
        # --- Per-band gain multipliers ---
        self.band_gain = {
            "bass": 4.0,
            "mid":  3.0,
            "treble": 4.0,
        }
        # Per-band gate: knobs 21-23 control what fraction of the last 10s the gate is open
        self.gate_target_fraction = {"bass": 0.10, "mid": 0.10, "treble": 0.10}
        self.threshold_db = {"bass": -40.0, "mid": -40.0, "treble": -40.0}  # computed

        # --- Gate params ---
        self.noise_lerp      = 0.995
        self.snr_floor_db    = 1.0
        self.snr_ceil_db     = 20.0

        self.alpha_attack, self.alpha_release = 0.5, 0.9

        # --- Precompute FFT bins ---
        self.freqs = np.fft.rfftfreq(BLOCK_SIZE, 1.0 / SAMPLE_RATE)

        self.state  = {k: 0.0 for k in self.bands}
        self.values = {k: 0.0 for k in self.bands}
        self.audio_array = np.zeros(len(self.bands))  # [bass, mid, treble]
        self.stream = None

        # noise/gate state
        self.noise_rms     = 1e-3
        self.noise_band_pw = {"bass": 1e-4, "mid": 1e-5, "treble": 1e-6}
        self.gate_open     = {k: False for k in self.bands}
        self._gate_hysteresis_db = 3.0

        # Per-band dB history for adaptive thresholds (last 10s)
        self._gate_hist_seconds = 10.0
        self._gate_hist_size = int(self._gate_hist_seconds * SAMPLE_RATE / BLOCK_SIZE)
        self._band_db_history = {k: np.full(self._gate_hist_size, -80.0) for k in self.bands}
        self._band_hist_idx = 0

        # for viz (read by viz thread)
        self.last_ps = None
        self.last_rms_db = -60.0
        self.last_band_db = {k: -80.0 for k in self.bands}

        # --- delay control ---
        self.delay_seconds = 0.0
        self.max_delay_seconds = 0.6
        self.max_delay_frames = int(self.max_delay_seconds * SAMPLE_RATE)
        self.delay_buffer = np.zeros((self.max_delay_frames, self.channels), dtype=np.float32)
        self.delay_write_pos = 0

        self.update_with_bands()
        atexit.register(self.stop)

        self._viz_thread = None
        if self.visualize:
            self._start_visualization()

    def update_with_bands(self):
        self.band_idx = {
            k: np.where((self.freqs >= lo) & (self.freqs < hi))[0]
            for k, (lo, hi) in self.bands.items()
        }

    # ------------------------------------------------------------------ callback
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

        end = pos + n
        if end <= self.max_delay_frames:
            buf[pos:end] = block
        else:
            split = self.max_delay_frames - pos
            buf[pos:] = block[:split]
            buf[:n - split] = block[split:]

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

        # ===== ANALYSIS (mono mixdown) =====
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

        # Per-band mean power (log-compressed)
        band_pw = {
            k: (0.0 if len(idx) == 0 else float(np.log1p(np.mean(ps[idx]))))
            for k, idx in self.band_idx.items()
        }

        # Gate logic
        level_db = _db_amp(_rms(x))
        self.last_rms_db = level_db
        self.last_band_db = {
            k: 10.0 * np.log10(band_pw[k] + EPS)
            for k in band_pw
        }

        # Update per-band dB history
        for k in self.bands:
            self._band_db_history[k][self._band_hist_idx] = self.last_band_db[k]
        self._band_hist_idx = (self._band_hist_idx + 1) % self._gate_hist_size

        # Per-band adaptive gate
        for k in self.bands:
            pct = 100.0 * (1.0 - self.gate_target_fraction[k])
            thresh_open = float(np.percentile(self._band_db_history[k], pct))
            thresh_close = thresh_open - self._gate_hysteresis_db
            self.threshold_db[k] = thresh_open  # expose for viz
            if self.gate_open[k]:
                if self.last_band_db[k] < thresh_close:
                    self.gate_open[k] = False
            else:
                if self.last_band_db[k] > thresh_open:
                    self.gate_open[k] = True

        # Per-band outputs
        arr = []
        for k, p in band_pw.items():
            last = self.state.get(k, 0.0)
            if not self.gate_open[k]:
                self.noise_band_pw[k] = (
                    self.noise_lerp * self.noise_band_pw[k]
                    + (1.0 - self.noise_lerp) * p
                )
                target = 0.0
            else:
                snr  = (p + EPS) / (self.noise_band_pw[k] + EPS)
                norm = _snr01(10.0 * np.log10(snr), self.snr_floor_db, self.snr_ceil_db)
                target = np.clip(norm * self.band_gain[k], 0.0, 1.0)

            alpha = self.alpha_attack if target > last else self.alpha_release
            smoothed = alpha * last + (1.0 - alpha) * target

            self.state[k]  = smoothed
            self.values[k] = smoothed
            arr.append(smoothed)

        self.audio_array = np.array(arr, dtype=np.float32)

    # ------------------------------------------------------------------ viz
    def _start_visualization(self):
        import threading
        self._viz_thread = threading.Thread(target=self._viz_loop, daemon=True)
        self._viz_thread.start()

    def _viz_loop(self):
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        MAX_HIST = 100
        x_axis = np.arange(MAX_HIST)
        band_names = ["bass", "mid", "treble"]
        band_colors = ["#2196F3", "#4CAF50", "#F44336"]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.patch.set_facecolor('#1a1a1a')
        for ax in axes.flat:
            ax.set_facecolor('#2a2a2a')
            ax.tick_params(colors='gray')

        # ---- Top-left: Band Reactivity (0-1) ----
        ax_bands = axes[0, 0]
        ax_bands.set_title("Band Reactivity (0\u20131)", color='white', fontsize=10)
        ax_bands.set_ylim(0, 1.05)
        ax_bands.set_xlim(0, MAX_HIST)

        bass_hist = [0.0] * MAX_HIST
        mid_hist = [0.0] * MAX_HIST
        treble_hist = [0.0] * MAX_HIST
        (bass_line,) = ax_bands.plot([], [], color=band_colors[0], linewidth=2, label="bass")
        (mid_line,) = ax_bands.plot([], [], color=band_colors[1], linewidth=2, label="mid")
        (treble_line,) = ax_bands.plot([], [], color=band_colors[2], linewidth=2, label="treble")
        ax_bands.legend(loc="upper right", fontsize=7, facecolor='#333', labelcolor='white')

        # ---- Top-right: Per-band dB + Thresholds ----
        ax_rms = axes[0, 1]
        ax_rms.set_title("Band dB + Thresholds", color='white', fontsize=10)
        ax_rms.set_ylabel("dB", color='gray')
        ax_rms.set_ylim(-50, 10)
        ax_rms.set_xlim(0, MAX_HIST)

        bass_db_hist = [-60.0] * MAX_HIST
        mid_db_hist = [-60.0] * MAX_HIST
        treble_db_hist = [-60.0] * MAX_HIST
        (line_bass_db,) = ax_rms.plot([], [], color=band_colors[0], linewidth=1.5, label="bass")
        (line_mid_db,) = ax_rms.plot([], [], color=band_colors[1], linewidth=1.5, label="mid")
        (line_treble_db,) = ax_rms.plot([], [], color=band_colors[2], linewidth=1.5, label="treble")
        (thresh_bass_line,) = ax_rms.plot([], [], color=band_colors[0], linestyle="--", alpha=0.5)
        (thresh_mid_line,) = ax_rms.plot([], [], color=band_colors[1], linestyle="--", alpha=0.5)
        (thresh_treble_line,) = ax_rms.plot([], [], color=band_colors[2], linestyle="--", alpha=0.5)
        ax_rms.legend(loc="upper right", fontsize=7, facecolor='#333', labelcolor='white')

        # ---- Bottom-left: Spectrum ----
        ax_spec = axes[1, 0]
        ax_spec.set_title("Spectrum", color='white', fontsize=10)
        ax_spec.set_ylim(-100, 20)
        ax_spec.set_xlim(20, 8000)
        ax_spec.set_xscale("log")
        ax_spec.set_xlabel("Hz", color='gray')
        (freq_line,) = ax_spec.plot([], [], color="white", linewidth=0.8)
        (bass_split,) = ax_spec.plot([], [], color=band_colors[0], linestyle="--", alpha=0.6)
        (mid_split,) = ax_spec.plot([], [], color=band_colors[1], linestyle="--", alpha=0.6)

        # ---- Bottom-right: Gains + Delay ----
        ax_ctrl = axes[1, 1]
        ax_ctrl.set_title("Gains + Delay", color='white', fontsize=10)
        ax_ctrl.set_xlim(-0.5, 3.5)
        ax_ctrl.set_ylim(0, 9)
        gain_bars = ax_ctrl.bar(range(3), [0]*3, color=band_colors, width=0.6)
        delay_bar = ax_ctrl.bar([3], [0], color='#607D8B', width=0.6)
        ax_ctrl.set_xticks(range(4))
        ax_ctrl.set_xticklabels(band_names + ["delay"], fontsize=9, color='white')
        gain_texts = [ax_ctrl.text(i, 0.1, "", ha='center', va='bottom',
                                   fontsize=10, color='white', fontweight='bold')
                      for i in range(3)]
        delay_text = ax_ctrl.text(3, 0.1, "", ha='center', va='bottom',
                                  fontsize=10, color='white', fontweight='bold')

        hist_idx = [0]

        def _update(frame):
            vals = self.audio_array.copy()
            band_db = self.last_band_db
            gains = [self.band_gain[k] for k in band_names]
            thresholds = self.threshold_db

            idx = hist_idx[0]
            bass_db_hist[idx] = band_db['bass']
            mid_db_hist[idx] = band_db['mid']
            treble_db_hist[idx] = band_db['treble']
            bass_hist[idx] = float(vals[0]) if len(vals) > 0 else 0.0
            mid_hist[idx] = float(vals[1]) if len(vals) > 1 else 0.0
            treble_hist[idx] = float(vals[2]) if len(vals) > 2 else 0.0
            hist_idx[0] = (idx + 1) % MAX_HIST

            # Band reactivity lines
            bass_line.set_data(x_axis, bass_hist)
            mid_line.set_data(x_axis, mid_hist)
            treble_line.set_data(x_axis, treble_hist)

            # Per-band dB + thresholds
            line_bass_db.set_data(x_axis, bass_db_hist)
            line_mid_db.set_data(x_axis, mid_db_hist)
            line_treble_db.set_data(x_axis, treble_db_hist)
            thresh_bass_line.set_data(x_axis, [thresholds['bass']] * MAX_HIST)
            thresh_mid_line.set_data(x_axis, [thresholds['mid']] * MAX_HIST)
            thresh_treble_line.set_data(x_axis, [thresholds['treble']] * MAX_HIST)

            all_db = [v for h in (bass_db_hist, mid_db_hist, treble_db_hist) for v in h if v > -200]
            all_thresh = list(thresholds.values())
            if all_db:
                lo = min(min(all_db), min(all_thresh)) - 5
                hi = max(max(all_db), max(all_thresh)) + 5
                ax_rms.set_ylim(lo, hi)

            # Gain bars + delay bar
            for bar, g, txt in zip(gain_bars, gains, gain_texts):
                bar.set_height(g)
                txt.set_text(f"{g:.1f}")
                txt.set_y(g + 0.15)
            d_ms = self.delay_seconds * 1000
            d_h = min(d_ms / 150.0 * 9.0, 8.5)
            delay_bar[0].set_height(d_h)
            delay_text.set_text(f"{d_ms:.0f}ms")
            delay_text.set_y(d_h + 0.15)

            # Spectrum
            ps = self.last_ps
            if ps is not None:
                freq_line.set_data(self.freqs, 10 * np.log10(ps + EPS))
                bass_split.set_data([self.bands['bass'][1]]*2, [-100, 20])
                mid_split.set_data([self.bands['mid'][1]]*2, [-100, 20])

        fig.tight_layout()
        ani = animation.FuncAnimation(fig, _update, interval=30,
                                      blit=False, cache_frame_data=False)
        plt.show()

    # ------------------------------------------------------------------ start/stop
    def start(self):
        print(f"[AudioGate] in={self.device_in}, out={self.device_out}, ch={self.channels}")
        self.stream = sd.Stream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=self.channels,
            dtype='float32',
            callback=self._callback,
            device=(self.device_in, self.device_out),
        )
        self.stream.start()
        print(f"[AudioGate] Stream started: {self.stream.samplerate}Hz")

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None


if __name__ == '__main__':
    audio = AudioGate(
        visualize=True,
        device_in="pulse",
        device_out="pulse",
        channels=2
    )
    audio.start()

    import time
    while True:
        time.sleep(1)