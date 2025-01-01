import os
import wave
import tempfile
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

    def __post_init__(self):
        self.nperseg = None #(self.sample_rate // 2) // 10,
        _, powvals = signal.welch(
            x=self.samples,
            fs=self.sample_rate,
            nperseg=self.nperseg,
            return_onesided=True
        )
        self.max_pow_val = np.max(powvals)

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
    def get_wavefile(mp3_path: str, length_seconds: float | None = None) -> Self:
        """ 
        This factory method takes an .mp3 file, and return a WaveFile object
        """

        wf: WaveFile = None
        temp_file = None

        try:
            temp_file = tempfile.NamedTemporaryFile(mode='wb', delete=False)
            sound_content = AudioSegment.from_file(mp3_path)
            sound_content.export(temp_file.name, format='wav')

            with wave.open(temp_file.name, 'rb') as wav_file:
                # get wave parameters
                num_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_rate = wav_file.getframerate()
                num_frames = wav_file.getnframes()

                # read the frames as bytes
                frames = wav_file.readframes(num_frames)

                # convert bytes to integers (adjust dtype based on sample width)
                if sample_width == 1:
                    dtype = np.int8
                elif sample_width == 2:
                    dtype = np.int16
                else:
                    raise ValueError("Unsupported sample width")

                samples = np.frombuffer(frames, dtype=dtype)

                # if stereo, split into left and right channels, then average them
                if num_channels == 2:
                    left_channel = samples[::2]
                    right_channel = samples[1::2]
                    averaged_channels = (left_channel + right_channel) / 2
                else:
                    left_channel = samples
                    averaged_channels = left_channel

                # trim as needed
                if length_seconds is not None:
                    num_samples_to_keep = round(frame_rate * length_seconds)
                    averaged_channels = averaged_channels[:num_samples_to_keep]
                
                wf = WaveFile(samples=averaged_channels, sample_rate=frame_rate)

        finally:
            temp_file.close()
            os.unlink(temp_file.name)
        
        return wf

    def get_power_in_freq_ranges(self, start_percentage: float, end_percentage: float, freq_ranges: list[tuple[int, int]]) -> tuple[float, ...]:
        samples_trimmed = self.get_samples(start_percentage, end_percentage)
        _, powvals = signal.welch(
            x=samples_trimmed,
            fs=self.sample_rate,
            nperseg=self.nperseg,
            return_onesided=True
        )

        result: list[float] = []
        for freq_range in freq_ranges:
            low_f = freq_range[0]
            high_f = freq_range[1]
            assert high_f >= low_f
            assert low_f >= 0

            result.append(float(integrate.trapezoid(powvals[low_f:high_f])))

        # normalize to max value of ~100.0
        # TODO: don't keep recalculating the overall max, figure out a better solution as
        # the 'welch' algorithm on all samples has different max value vs. the trimmed samples, so this doesn't
        # accurately normalize to 100.0
        result = (result / self.max_pow_val) * 100.0

        return tuple(result)

if __name__ == '__main__':
    wf: WaveFile = WaveFile.get_wavefile("emotinium_ii.mp3")
    fpow = wf.get_power_in_freq_ranges(start_percentage=0.0, end_percentage=1.0, freq_ranges=[(0,20), (21,300), (301, 4000), (4001, 20000)])
    print(fpow)