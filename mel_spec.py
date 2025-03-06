import numpy as np
import scipy.io.wavfile as wav
import struct
import math

class MelFilterBank:
    def __init__(self, samplerate, nfft, low_freq, high_freq, n_mel):
        self.samplerate = samplerate
        self.nfft = nfft
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.n_mel = n_mel

        self.low_mel = self.f_to_mel(low_freq)
        self.high_mel = self.f_to_mel(high_freq)
        self.d_mel = (self.high_mel - self.low_mel) / (n_mel + 1)

        self.mel_points = [self.low_mel + i * self.d_mel for i in range(n_mel + 2)]
        self.freq_bins = [self.mel_to_f(m) for m in self.mel_points]
        self.bin_indices = [int(math.floor((nfft + 1) * f / samplerate)) for f in self.freq_bins]
        self.filters = self.create_filter_banks()

    def f_to_mel(self, f):
        return 2595 * math.log10(1 + f / 700)

    def mel_to_f(self, m):
        return 700 * (10**(m / 2595) - 1)

    def create_filter_banks(self):
        filters = []
        for m in range(1, self.n_mel + 1):
            filter_bank = np.zeros(self.nfft // 2 + 1)
            for k in range(self.nfft // 2 + 1):
                if self.bin_indices[m - 1] <= k < self.bin_indices[m]:
                    filter_bank[k] = (k - self.bin_indices[m - 1]) / (self.bin_indices[m] - self.bin_indices[m - 1])
                elif self.bin_indices[m] <= k <= self.bin_indices[m + 1]:
                    filter_bank[k] = (self.bin_indices[m + 1] - k) / (self.bin_indices[m + 1] - self.bin_indices[m])
            filters.append(filter_bank)
        return np.array(filters)

    def apply_filter_bank(self, spectrum):
        return np.dot(self.filters, spectrum)

class MelSpectrogram:
    def __init__(self, samplerate, frame_size, n_mel, min_freq, max_freq):
        self.nfft = frame_size
        self.filter_bank = MelFilterBank(samplerate, frame_size, min_freq, max_freq, n_mel)

    def compute(self, signal):
        windowed = self.apply_hamming_window(signal)
        spectrum = self.compute_fft(windowed)
        return self.filter_bank.apply_filter_bank(spectrum)

    def apply_hamming_window(self, buffer):
        return buffer * (0.54 - 0.46 * np.cos(2 * np.pi * np.arange(len(buffer)) / (len(buffer) - 1)))

    def compute_fft(self, buffer):
        complex_spectrum = np.fft.rfft(buffer)
        return np.abs(complex_spectrum)

def process_wav_file(file_path):
    samplerate, data = wav.read(file_path)
    spectrogram_data = process_chunk(data, samplerate)
    save_spectrogram(spectrogram_data, 'debug_spectrogram_py.png')

def process_chunk(data, samplerate, frame_size=2048, hop_size=512, n_mel=40, min_freq=300, max_freq=8000):
    
    if data.ndim > 1:
        data = data[:, 0]
    
    mel_spec = MelSpectrogram(samplerate, frame_size, n_mel, min_freq, max_freq)
    spectrogram_data = []

    for i in range(0, len(data) - frame_size, hop_size):
        chunk = data[i:i + frame_size].astype(np.float32)
        mel_data = mel_spec.compute(chunk)
        spectrogram_data.append(mel_data)
    
    return np.array(spectrogram_data).T

def save_spectrogram(spectrogram_data, filename):
    from PIL import Image
    min_val, max_val = np.min(spectrogram_data), np.max(spectrogram_data)
    norm_spectrogram = ((spectrogram_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    image = Image.fromarray(norm_spectrogram[::-1], mode='L')
    image.save(filename)

# Example usage
process_wav_file('your_audio_file.wav')
