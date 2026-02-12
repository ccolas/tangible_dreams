import atexit
import ctypes
import multiprocessing as mp
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

# Shared buffer layout (all float32):
#   [0:3]   ema values (bass, mid, treble)
#   [3:6]   flux values (bass, mid, treble)
#   [6:9]   band_db
#   [9:12]  threshold_db
#   [12:15] band_gain
#   [15]    delay_seconds
#   [16]    flux_decay
#   [17]    alpha_attack
#   [18]    alpha_release
#   [19]    bass lo freq
#   [20]    bass/mid crossover freq
#   [21]    mid/treble crossover freq
#   [22:22+PS] last_ps
_PS_SIZE = BLOCK_SIZE // 2 + 1   # rfft output length (513 for BLOCK_SIZE=1024)
_SHARED_META = 22
_SHARED_SIZE = _SHARED_META + _PS_SIZE


def _viz_process_main(shared_buf, freqs, band_names, band_colors):
    """Runs in a separate process — matplotlib has its own GIL here."""
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    MAX_HIST = 100
    x_axis = np.arange(MAX_HIST)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.patch.set_facecolor('#1a1a1a')
    for ax in axes.flat:
        ax.set_facecolor('#2a2a2a')
        ax.tick_params(colors='gray')

    # ---- Top-left: EMA Reactivity ----
    ax_ema = axes[0, 0]
    ax_ema.set_title("EMA (sustained)", color='white', fontsize=10)
    ax_ema.set_ylim(0, 1.05)
    ax_ema.set_xlim(0, MAX_HIST)
    ema_hists = {k: [0.0] * MAX_HIST for k in band_names}
    ema_lines = {}
    for i, k in enumerate(band_names):
        (ema_lines[k],) = ax_ema.plot([], [], color=band_colors[i], linewidth=2, label=k)
    ax_ema.legend(loc="upper right", fontsize=7, facecolor='#333', labelcolor='white')

    # ---- Top-mid: Flux Reactivity ----
    ax_flux = axes[0, 1]
    ax_flux.set_title("Flux (transients)", color='white', fontsize=10)
    ax_flux.set_ylim(0, 1.05)
    ax_flux.set_xlim(0, MAX_HIST)
    flux_hists = {k: [0.0] * MAX_HIST for k in band_names}
    flux_lines = {}
    for i, k in enumerate(band_names):
        (flux_lines[k],) = ax_flux.plot([], [], color=band_colors[i], linewidth=2, label=k)
    ax_flux.legend(loc="upper right", fontsize=7, facecolor='#333', labelcolor='white')

    # ---- Top-right: Per-band dB + Thresholds ----
    ax_rms = axes[0, 2]
    ax_rms.set_title("Band dB + Thresholds", color='white', fontsize=10)
    ax_rms.set_ylabel("dB", color='gray')
    ax_rms.set_ylim(-50, 10)
    ax_rms.set_xlim(0, MAX_HIST)
    db_hists = {k: [-60.0] * MAX_HIST for k in band_names}
    db_lines = {}
    thresh_lines = {}
    for i, k in enumerate(band_names):
        (db_lines[k],) = ax_rms.plot([], [], color=band_colors[i], linewidth=1.5, label=k)
        (thresh_lines[k],) = ax_rms.plot([], [], color=band_colors[i], linestyle="--", alpha=0.5)
    ax_rms.legend(loc="upper right", fontsize=7, facecolor='#333', labelcolor='white')

    # ---- Bottom-left: Spectrum + crossover lines ----
    ax_spec = axes[1, 0]
    ax_spec.set_title("Spectrum", color='white', fontsize=10)
    ax_spec.set_ylim(-100, 20)
    ax_spec.set_xlim(20, 10000)
    ax_spec.set_xscale("log")
    ax_spec.set_xlabel("Hz", color='gray')
    (freq_line,) = ax_spec.plot([], [], color="white", linewidth=0.8)
    # 3 crossover vertical lines: bass_lo, bass/mid, mid/treble
    (xover_lo,) = ax_spec.plot([], [], color=band_colors[0], linestyle=":", alpha=0.7, linewidth=1.5)
    (xover_mid,) = ax_spec.plot([], [], color=band_colors[1], linestyle=":", alpha=0.7, linewidth=1.5)
    (xover_hi,) = ax_spec.plot([], [], color=band_colors[2], linestyle=":", alpha=0.7, linewidth=1.5)
    xover_lo_txt = ax_spec.text(30, 15, "", color=band_colors[0], fontsize=7, ha='left')
    xover_mid_txt = ax_spec.text(150, 15, "", color=band_colors[1], fontsize=7, ha='left')
    xover_hi_txt = ax_spec.text(3000, 15, "", color=band_colors[2], fontsize=7, ha='left')

    # ---- Bottom-mid: Controls ----
    ax_ctrl = axes[1, 1]
    ax_ctrl.set_title("Controls", color='white', fontsize=10)
    ax_ctrl.set_xlim(-0.5, 4.5)
    ax_ctrl.set_ylim(0, 10)
    # bars: bass_gain, mid_gain, treble_gain, delay, flux_decay
    bar_colors = band_colors + ['#607D8B', '#FF9800']
    bar_labels = band_names + ["delay", "flux\ndecay"]
    ctrl_bars = ax_ctrl.bar(range(5), [0]*5, color=bar_colors, width=0.6)
    ax_ctrl.set_xticks(range(5))
    ax_ctrl.set_xticklabels(bar_labels, fontsize=8, color='white')
    ctrl_texts = [ax_ctrl.text(i, 0.1, "", ha='center', va='bottom',
                               fontsize=9, color='white', fontweight='bold')
                  for i in range(5)]
    # Text annotations for attack/release
    param_text = ax_ctrl.text(0.98, 0.98, "", transform=ax_ctrl.transAxes,
                              ha='right', va='top', fontsize=8, color='#aaa',
                              family='monospace')

    # Hide empty bottom-right panel
    axes[1, 2].set_visible(False)

    hist_idx = [0]

    def _update(frame):
        buf = np.frombuffer(shared_buf.get_obj(), dtype=np.float32)
        ema = buf[0:3].copy()
        flux = buf[3:6].copy()
        band_db = {"bass": buf[6], "mid": buf[7], "treble": buf[8]}
        thresholds = {"bass": buf[9], "mid": buf[10], "treble": buf[11]}
        gains = [buf[12], buf[13], buf[14]]
        delay_s = buf[15]
        flux_decay = buf[16]
        alpha_atk = buf[17]
        alpha_rel = buf[18]
        bass_lo = buf[19]
        bass_mid = buf[20]
        mid_treble = buf[21]
        ps = buf[_SHARED_META:_SHARED_META + _PS_SIZE].copy()

        idx = hist_idx[0]
        for i, k in enumerate(band_names):
            ema_hists[k][idx] = float(ema[i])
            flux_hists[k][idx] = float(flux[i])
            db_hists[k][idx] = band_db[k]
        hist_idx[0] = (idx + 1) % MAX_HIST

        # EMA reactivity
        for k in band_names:
            ema_lines[k].set_data(x_axis, ema_hists[k])

        # Flux reactivity
        for k in band_names:
            flux_lines[k].set_data(x_axis, flux_hists[k])

        # Per-band dB + thresholds
        for k in band_names:
            db_lines[k].set_data(x_axis, db_hists[k])
            thresh_lines[k].set_data(x_axis, [thresholds[k]] * MAX_HIST)

        all_db = [v for k in band_names for v in db_hists[k] if v > -200]
        all_thresh = list(thresholds.values())
        if all_db:
            lo = min(min(all_db), min(all_thresh)) - 5
            hi = max(max(all_db), max(all_thresh)) + 5
            ax_rms.set_ylim(lo, hi)

        # Controls bars
        d_ms = delay_s * 1000
        bar_vals = gains + [min(d_ms / 150.0 * 9.0, 8.5), flux_decay * 9.0]
        bar_txts = [f"{g:.1f}" for g in gains] + [f"{d_ms:.0f}ms", f"{flux_decay:.2f}"]
        for bar, val, txt_obj, txt_str in zip(ctrl_bars, bar_vals, ctrl_texts, bar_txts):
            bar.set_height(val)
            txt_obj.set_text(txt_str)
            txt_obj.set_y(val + 0.15)

        param_text.set_text(f"attack={alpha_atk:.2f}\nrelease={alpha_rel:.2f}")

        # Spectrum + crossover lines
        if np.any(ps > 0):
            freq_line.set_data(freqs, 10 * np.log10(ps + EPS))

        if bass_lo > 0:
            xover_lo.set_data([bass_lo]*2, [-100, 20])
            xover_lo_txt.set_text(f"{bass_lo:.0f}")
            xover_lo_txt.set_x(bass_lo)
        if bass_mid > 0:
            xover_mid.set_data([bass_mid]*2, [-100, 20])
            xover_mid_txt.set_text(f"{bass_mid:.0f}")
            xover_mid_txt.set_x(bass_mid)
        if mid_treble > 0:
            xover_hi.set_data([mid_treble]*2, [-100, 20])
            xover_hi_txt.set_text(f"{mid_treble:.0f}")
            xover_hi_txt.set_x(mid_treble)

    fig.tight_layout()
    ani = animation.FuncAnimation(fig, _update, interval=30,
                                  blit=False, cache_frame_data=False)
    plt.show()


####################################################################
# Simple mode: 3 signals (bass/mid/treble EMA), 2x2 viz
####################################################################

_SIMPLE_SHARED_META = 13
_SIMPLE_SHARED_SIZE = _SIMPLE_SHARED_META + _PS_SIZE


def _viz_process_simple(shared_buf, freqs, bands, band_names, band_colors):
    """Viz for AudioGateSimple — 2x2 layout, EMA only."""
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    MAX_HIST = 100
    x_axis = np.arange(MAX_HIST)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor('#1a1a1a')
    for ax in axes.flat:
        ax.set_facecolor('#2a2a2a')
        ax.tick_params(colors='gray')

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

    ax_spec = axes[1, 0]
    ax_spec.set_title("Spectrum", color='white', fontsize=10)
    ax_spec.set_ylim(-100, 20)
    ax_spec.set_xlim(20, 8000)
    ax_spec.set_xscale("log")
    ax_spec.set_xlabel("Hz", color='gray')
    (freq_line,) = ax_spec.plot([], [], color="white", linewidth=0.8)
    (bass_split,) = ax_spec.plot([], [], color=band_colors[0], linestyle="--", alpha=0.6)
    (mid_split,) = ax_spec.plot([], [], color=band_colors[1], linestyle="--", alpha=0.6)

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
        buf = np.frombuffer(shared_buf.get_obj(), dtype=np.float32)
        vals = buf[0:3].copy()
        band_db = {"bass": buf[3], "mid": buf[4], "treble": buf[5]}
        thresholds = {"bass": buf[6], "mid": buf[7], "treble": buf[8]}
        gains = [buf[9], buf[10], buf[11]]
        delay_s = buf[12]
        ps = buf[_SIMPLE_SHARED_META:_SIMPLE_SHARED_META + _PS_SIZE].copy()

        idx = hist_idx[0]
        bass_db_hist[idx] = band_db['bass']
        mid_db_hist[idx] = band_db['mid']
        treble_db_hist[idx] = band_db['treble']
        bass_hist[idx] = float(vals[0])
        mid_hist[idx] = float(vals[1])
        treble_hist[idx] = float(vals[2])
        hist_idx[0] = (idx + 1) % MAX_HIST

        bass_line.set_data(x_axis, bass_hist)
        mid_line.set_data(x_axis, mid_hist)
        treble_line.set_data(x_axis, treble_hist)

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

        for bar, g, txt in zip(gain_bars, gains, gain_texts):
            bar.set_height(g)
            txt.set_text(f"{g:.1f}")
            txt.set_y(g + 0.15)
        d_ms = delay_s * 1000
        d_h = min(d_ms / 150.0 * 9.0, 8.5)
        delay_bar[0].set_height(d_h)
        delay_text.set_text(f"{d_ms:.0f}ms")
        delay_text.set_y(d_h + 0.15)

        if np.any(ps > 0):
            freq_line.set_data(freqs, 10 * np.log10(ps + EPS))
            bass_split.set_data([bands['bass'][1]]*2, [-100, 20])
            mid_split.set_data([bands['mid'][1]]*2, [-100, 20])

    fig.tight_layout()
    ani = animation.FuncAnimation(fig, _update, interval=30,
                                  blit=False, cache_frame_data=False)
    plt.show()


class AudioGateSimple:
    """Original 3-signal mode: bass/mid/treble EMA only. No flux."""

    def __init__(self, visualize=False, device_in=None, device_out=None, channels=2):
        self.visualize = visualize
        self.device_in = device_in
        self.device_out = device_out
        self.channels = channels
        self.mode = "simple"

        self.bands = {"bass": (30, 150), "mid": (150, 3000), "treble": (3000, 8000)}
        self.band_gain = {"bass": 4.0, "mid": 3.0, "treble": 4.0}
        self.gate_target_fraction = {"bass": 0.10, "mid": 0.10, "treble": 0.10}
        self.threshold_db = {"bass": -40.0, "mid": -40.0, "treble": -40.0}

        self.noise_lerp = 0.995
        self.snr_floor_db = 1.0
        self.snr_ceil_db = 20.0
        self.alpha_attack, self.alpha_release = 0.5, 0.9

        self.freqs = np.fft.rfftfreq(BLOCK_SIZE, 1.0 / SAMPLE_RATE)
        self.state = {k: 0.0 for k in self.bands}
        self.values = {k: 0.0 for k in self.bands}
        self.audio_array = np.zeros(3)
        self.stream = None

        self.noise_rms = 1e-3
        self.noise_band_pw = {"bass": 1e-4, "mid": 1e-5, "treble": 1e-6}
        self.gate_open = {k: False for k in self.bands}
        self._gate_hysteresis_db = 3.0

        self._gate_hist_seconds = 10.0
        self._gate_hist_size = int(self._gate_hist_seconds * SAMPLE_RATE / BLOCK_SIZE)
        self._band_db_history = {k: np.full(self._gate_hist_size, -80.0) for k in self.bands}
        self._band_hist_idx = 0

        self.last_ps = None
        self.last_rms_db = -60.0
        self.last_band_db = {k: -80.0 for k in self.bands}

        self.delay_seconds = 0.0
        self.max_delay_seconds = 0.6
        self.max_delay_frames = int(self.max_delay_seconds * SAMPLE_RATE)
        self.delay_buffer = np.zeros((self.max_delay_frames, self.channels), dtype=np.float32)
        self.delay_write_pos = 0

        self.update_with_bands()
        atexit.register(self.stop)

        self._shared_buf = mp.Array(ctypes.c_float, _SIMPLE_SHARED_SIZE)
        self._viz_proc = None
        if self.visualize:
            self._start_visualization()

    def update_with_bands(self):
        self.band_idx = {
            k: np.where((self.freqs >= lo) & (self.freqs < hi))[0]
            for k, (lo, hi) in self.bands.items()
        }

    def _callback(self, indata, outdata, frames, time_info, status):
        if status:
            pass

        # Delayed output
        delay_frames = min(int(self.delay_seconds * SAMPLE_RATE), self.max_delay_frames - 1)
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
        read_pos = (pos - delay_frames) % self.max_delay_frames
        end_r = read_pos + n
        if end_r <= self.max_delay_frames:
            outdata[:] = buf[read_pos:end_r]
        else:
            split_r = self.max_delay_frames - read_pos
            outdata[:] = np.vstack((buf[read_pos:], buf[:n - split_r]))
        self.delay_write_pos = (pos + n) % self.max_delay_frames

        # Analysis
        x = indata.astype(np.float32)
        if x.ndim == 2 and x.shape[1] > 1:
            x = x.mean(axis=1)
        else:
            x = x[:, 0] if x.ndim == 2 else x
        x -= np.mean(x)

        w = np.hanning(len(x))
        X = np.fft.rfft(x * w)
        ps = (np.abs(X)**2) / (np.sum(w**2) + EPS)
        self.last_ps = ps

        band_pw = {
            k: (0.0 if len(idx) == 0 else float(np.log1p(np.mean(ps[idx]))))
            for k, idx in self.band_idx.items()
        }

        level_db = _db_amp(_rms(x))
        self.last_rms_db = level_db
        self.last_band_db = {k: 10.0 * np.log10(band_pw[k] + EPS) for k in band_pw}

        for k in self.bands:
            self._band_db_history[k][self._band_hist_idx] = self.last_band_db[k]
        self._band_hist_idx = (self._band_hist_idx + 1) % self._gate_hist_size

        for k in self.bands:
            pct = 100.0 * (1.0 - self.gate_target_fraction[k])
            thresh_open = float(np.percentile(self._band_db_history[k], pct))
            thresh_close = thresh_open - self._gate_hysteresis_db
            self.threshold_db[k] = thresh_open
            if self.gate_open[k]:
                if self.last_band_db[k] < thresh_close:
                    self.gate_open[k] = False
            else:
                if self.last_band_db[k] > thresh_open:
                    self.gate_open[k] = True

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
                snr = (p + EPS) / (self.noise_band_pw[k] + EPS)
                norm = _snr01(10.0 * np.log10(snr), self.snr_floor_db, self.snr_ceil_db)
                target = np.clip(norm * self.band_gain[k], 0.0, 1.0)
            alpha = self.alpha_attack if target > last else self.alpha_release
            smoothed = alpha * last + (1.0 - alpha) * target
            self.state[k] = smoothed
            self.values[k] = smoothed
            arr.append(smoothed)

        self.audio_array = np.array(arr, dtype=np.float32)

        if self._viz_proc is not None:
            sbuf = np.frombuffer(self._shared_buf.get_obj(), dtype=np.float32)
            sbuf[0:3] = self.audio_array[:3]
            sbuf[3] = self.last_band_db['bass']
            sbuf[4] = self.last_band_db['mid']
            sbuf[5] = self.last_band_db['treble']
            sbuf[6] = self.threshold_db['bass']
            sbuf[7] = self.threshold_db['mid']
            sbuf[8] = self.threshold_db['treble']
            sbuf[9] = self.band_gain['bass']
            sbuf[10] = self.band_gain['mid']
            sbuf[11] = self.band_gain['treble']
            sbuf[12] = self.delay_seconds
            if self.last_ps is not None:
                sbuf[_SIMPLE_SHARED_META:_SIMPLE_SHARED_META + len(self.last_ps)] = self.last_ps.astype(np.float32)

    def _start_visualization(self):
        band_names = ["bass", "mid", "treble"]
        band_colors = ["#2196F3", "#4CAF50", "#F44336"]
        self._viz_proc = mp.Process(
            target=_viz_process_simple,
            args=(self._shared_buf, self.freqs.copy(), dict(self.bands),
                  band_names, band_colors),
            daemon=True,
        )
        self._viz_proc.start()

    def start(self):
        print(f"[AudioGateSimple] in={self.device_in}, out={self.device_out}, ch={self.channels}")
        self.stream = sd.Stream(
            samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
            channels=self.channels, dtype='float32',
            callback=self._callback, device=(self.device_in, self.device_out),
        )
        self.stream.start()
        print(f"[AudioGateSimple] Stream started: {self.stream.samplerate}Hz")

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if self._viz_proc and self._viz_proc.is_alive():
            self._viz_proc.terminate()
            self._viz_proc = None


####################################################################
# Flux mode: 6 signals (3 EMA + 3 flux), 2x3 viz
####################################################################

class AudioGate:
    """6-signal mode: bass/mid/treble EMA + bass/mid/treble flux."""

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
        self.mode = "flux"

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

        # --- Flux params ---
        self.flux_decay = 0.85  # 0=instant drop, 0.95=slow tail

        # --- Precompute FFT bins ---
        self.freqs = np.fft.rfftfreq(BLOCK_SIZE, 1.0 / SAMPLE_RATE)

        self.state  = {k: 0.0 for k in self.bands}
        self.flux_state = {k: 0.0 for k in self.bands}
        self._prev_band_pw = {k: 0.0 for k in self.bands}
        self.values = {k: 0.0 for k in self.bands}
        self.audio_array = np.zeros(6)  # [bass_ema, mid_ema, treble_ema, bass_flux, mid_flux, treble_flux]
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

        # for viz (read by viz process)
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

        # Shared memory for visualization process
        self._shared_buf = mp.Array(ctypes.c_float, _SHARED_SIZE)
        self._viz_proc = None
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

        # Per-band EMA + flux outputs
        ema_arr = []
        flux_arr = []
        for k, p in band_pw.items():
            last_ema = self.state.get(k, 0.0)

            # --- Flux: half-wave rectified spectral change ---
            flux_raw = max(0.0, p - self._prev_band_pw[k])
            self._prev_band_pw[k] = p

            if not self.gate_open[k]:
                self.noise_band_pw[k] = (
                    self.noise_lerp * self.noise_band_pw[k]
                    + (1.0 - self.noise_lerp) * p
                )
                ema_target = 0.0
                flux_target = 0.0
            else:
                snr  = (p + EPS) / (self.noise_band_pw[k] + EPS)
                norm = _snr01(10.0 * np.log10(snr), self.snr_floor_db, self.snr_ceil_db)
                ema_target = np.clip(norm * self.band_gain[k], 0.0, 1.0)

                snr_flux = (flux_raw + EPS) / (self.noise_band_pw[k] + EPS)
                norm_flux = _snr01(10.0 * np.log10(snr_flux), self.snr_floor_db, self.snr_ceil_db)
                flux_target = np.clip(norm_flux * self.band_gain[k], 0.0, 1.0)

            # EMA smoothing (attack/release)
            alpha = self.alpha_attack if ema_target > last_ema else self.alpha_release
            smoothed = alpha * last_ema + (1.0 - alpha) * ema_target
            self.state[k] = smoothed
            self.values[k] = smoothed
            ema_arr.append(smoothed)

            # Flux envelope: instant attack, configurable decay
            self.flux_state[k] = max(flux_target, self.flux_state[k] * self.flux_decay)
            flux_arr.append(self.flux_state[k])

        self.audio_array = np.array(ema_arr + flux_arr, dtype=np.float32)

        # Write to shared memory for viz process (no lock needed — single writer)
        if self._viz_proc is not None:
            sbuf = np.frombuffer(self._shared_buf.get_obj(), dtype=np.float32)
            sbuf[0:3] = self.audio_array[:3]    # ema
            sbuf[3:6] = self.audio_array[3:6]   # flux
            sbuf[6] = self.last_band_db['bass']
            sbuf[7] = self.last_band_db['mid']
            sbuf[8] = self.last_band_db['treble']
            sbuf[9] = self.threshold_db['bass']
            sbuf[10] = self.threshold_db['mid']
            sbuf[11] = self.threshold_db['treble']
            sbuf[12] = self.band_gain['bass']
            sbuf[13] = self.band_gain['mid']
            sbuf[14] = self.band_gain['treble']
            sbuf[15] = self.delay_seconds
            sbuf[16] = self.flux_decay
            sbuf[17] = self.alpha_attack
            sbuf[18] = self.alpha_release
            sbuf[19] = self.bands['bass'][0]
            sbuf[20] = self.bands['bass'][1]      # = mid lo
            sbuf[21] = self.bands['mid'][1]        # = treble lo
            if self.last_ps is not None:
                sbuf[_SHARED_META:_SHARED_META + len(self.last_ps)] = self.last_ps.astype(np.float32)

    # ------------------------------------------------------------------ viz
    def _start_visualization(self):
        band_names = ["bass", "mid", "treble"]
        band_colors = ["#2196F3", "#4CAF50", "#F44336"]
        self._viz_proc = mp.Process(
            target=_viz_process_main,
            args=(self._shared_buf, self.freqs.copy(), band_names, band_colors),
            daemon=True,
        )
        self._viz_proc.start()

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
        if self._viz_proc and self._viz_proc.is_alive():
            self._viz_proc.terminate()
            self._viz_proc = None


class AudioPassthrough:
    """Capture audio from an input device, delay it, and output to speakers.
    No frequency analysis. Used in CV mode for synth audio through Scarlett."""

    def __init__(self, device_in=None, device_out=None, channels=2):
        self.device_in = device_in
        self.device_out = device_out
        self.channels = channels
        self.stream = None

        # delay control (same interface as AudioGate so app.py can read delay_seconds)
        self.delay_seconds = 0.0
        self.max_delay_seconds = 0.6
        self.max_delay_frames = int(self.max_delay_seconds * SAMPLE_RATE)
        self.delay_buffer = np.zeros((self.max_delay_frames, self.channels), dtype=np.float32)
        self.delay_write_pos = 0

        atexit.register(self.stop)

    def _callback(self, indata, outdata, frames, time_info, status):
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

    def start(self):
        print(f"[AudioPassthrough] in={self.device_in}, out={self.device_out}, ch={self.channels}")
        self.stream = sd.Stream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=self.channels,
            dtype='float32',
            callback=self._callback,
            device=(self.device_in, self.device_out),
        )
        self.stream.start()
        print(f"[AudioPassthrough] Stream started: {self.stream.samplerate}Hz")

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
