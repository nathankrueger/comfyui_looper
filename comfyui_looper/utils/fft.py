import tempfile
import os
import wave
from dataclasses import dataclass
from typing import Self

import scipy.integrate as integrate
from scipy import signal
import numpy as np
from numpy.typing import NDArray
from pydub import AudioSegment

@dataclass
class WaveFile:
    samples: NDArray
    sample_rate: int

    def get_samples(self, start_percentage: float, end_percentage: float) -> NDArray:
        assert end_percentage > start_percentage
        assert start_percentage >= 0.0
        assert start_percentage < 100.0
        assert end_percentage <= 100.0

        total_samples = self.samples.shape[0]
        start_sample = round((start_percentage / 100.0) * total_samples)
        end_sample = round((end_percentage / 100.0) * total_samples)
        return self.samples[start_sample:end_sample]

    @staticmethod
    def get_wavefile(mp3_path: str) -> Self:
        """
        This function will take an .mp3 file, and return wave samples,
        currently for the left channel only.
        """

        wf: WaveFile = None
        temp_file = None

        try:
            temp_file = tempfile.NamedTemporaryFile(mode='wb', delete=False)
            sound_content = AudioSegment.from_file(mp3_path)
            sound_content.export(temp_file.name, format="wav")

            with wave.open(temp_file.name, 'rb') as wav_file:
                # Get parameters
                num_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_rate = wav_file.getframerate()
                num_frames = wav_file.getnframes()

                # Read the frames as bytes
                frames = wav_file.readframes(num_frames)

                # Convert bytes to integers (adjust dtype based on sample width)
                if sample_width == 1:
                    dtype = np.int8
                elif sample_width == 2:
                    dtype = np.int16
                else:
                    raise ValueError("Unsupported sample width")

                samples = np.frombuffer(frames, dtype=dtype)

                # If stereo, split into left and right channels
                if num_channels == 2:
                    left_channel = samples[::2]
                    right_channel = samples[1::2]
                    averaged_channels = (left_channel + right_channel) / 2
                else:
                    left_channel = samples
                    averaged_channels = left_channel

                wf = WaveFile(samples=averaged_channels, sample_rate=frame_rate)

            temp_file.close()

        finally:
            os.unlink(temp_file.name)
        
        return wf

def get_power_in_freq_ranges(samples: NDArray, freq_ranges: list[tuple[int, int]]) -> tuple[float, ...]:
    _, powvals = signal.welch(samples, 44100, nperseg = 22050 // 10, nfft = 30000)
    result: list[float] = []
    for freq_range in freq_ranges:
        low_f = freq_range[0]
        high_f = freq_range[1]
        assert high_f >= low_f
        assert low_f >= 0

        result.append(float(integrate.trapezoid(powvals[low_f:high_f])))

    # normalize to max value of 1.0
    min_val = np.min(result)
    max_val = np.max(result)
    result = (result - min_val) / (max_val - min_val)

    return tuple(result)

if __name__ == '__main__':
    wf: WaveFile = WaveFile.get_wavefile("emotinium_ii.mp3")
    one_percent = wf.get_samples(0.0, 1.0)
    fpow = get_power_in_freq_ranges(one_percent, [(0,20), (21,300), (301, 4000), (4001, 20000)])
    print(fpow)