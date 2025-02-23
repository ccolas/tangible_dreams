import sounddevice as sd
import numpy as np
from collections import deque
import threading
import time


class AudioProcessor:
    def __init__(self, buffer_size=1024, sample_rate=44100, n_bands=8):
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.n_bands = n_bands
        self.audio_buffer = deque(maxlen=buffer_size)
        self.spectrum_values = np.zeros(n_bands)
        self.smoothed_values = np.zeros(n_bands)
        self.running = False

        # Noise threshold and smoothing parameters
        self.noise_threshold = 0.1  # Ignore values below this
        self.smoothing_factor = 0.3  # How fast values change (0-1)

        # Initialize buffer with zeros
        for _ in range(buffer_size):
            self.audio_buffer.append(0)

        # Frequency bands (in Hz)
        self.bands = np.logspace(np.log10(20), np.log10(20000), n_bands + 1)

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.audio_buffer.extend(indata[:, 0])

    def process_audio(self):
        """Convert audio buffer to frequency bands with noise reduction and smoothing"""
        data = np.array(list(self.audio_buffer))

        # Compute RMS to check if there's significant audio
        rms = np.sqrt(np.mean(data ** 2))
        if rms < 0.01:  # Very low audio level
            self.spectrum_values = np.zeros(self.n_bands)
            return

        # Apply Hanning window
        window = np.hanning(len(data))
        data = data * window

        # Compute FFT
        spectrum = np.abs(np.fft.rfft(data))
        freq = np.fft.rfftfreq(len(data), 1 / self.sample_rate)

        # Compute band energies
        new_values = np.zeros(self.n_bands)
        for i in range(self.n_bands):
            mask = (freq >= self.bands[i]) & (freq < self.bands[i + 1])
            new_values[i] = np.mean(spectrum[mask])

        # Normalize
        max_val = np.max(new_values)
        if max_val > 0:
            new_values = new_values / max_val

        # Apply noise threshold
        new_values[new_values < self.noise_threshold] = 0

        # Smooth values over time
        self.smoothed_values = (self.smoothing_factor * new_values +
                                (1 - self.smoothing_factor) * self.smoothed_values)

        self.spectrum_values = self.smoothed_values

    def get_bands(self):
        """Get current frequency band values"""
        return self.smoothed_values.copy()

    def start(self):
        """Start audio capture"""
        self.running = True
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            callback=self.audio_callback
        )
        self.stream.start()

        self.process_thread = threading.Thread(target=self._process_loop)
        self.process_thread.start()

    def _process_loop(self):
        while self.running:
            self.process_audio()
            time.sleep(1 / 30)  # Update at 30Hz

    def stop(self):
        self.running = False
        if hasattr(self, 'process_thread'):
            self.process_thread.join()
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()

    def set_threshold(self, value):
        """Adjust noise threshold during runtime"""
        self.noise_threshold = value

    def set_smoothing(self, value):
        """Adjust smoothing factor during runtime"""
        self.smoothing_factor = value