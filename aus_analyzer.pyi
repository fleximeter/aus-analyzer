import numpy as np
from typing import Tuple

"""
Analyzes a real FFT frame using the aus-rust crate.

:param magnitude_spectrum: The real FFT magnitude spectrum to analyze
:param fft_size: The FFT size
:param sample_rate: The audio sample rate
:return: A dictionary with the spectral analysis and the STFT data
"""
def analyze_rfft(magnitude_spectrum: np.ndarray, fft_size: int, sample_rate: int) -> dict: ...

"""
Analyzes an audio file using the aus-rust crate. Loads the audio file and performs the analysis.
The analysis that is returned also contains the STFT magnitude and phase spectrograms.

:param file: The file name to load and analyze
:param fft_size: The FFT size for the STFT
:param max_num_threads: The maximum number of threads to use (if set to 0, this will be set to the maximum allowed)
:return: A dictionary with the spectral analysis and the STFT data
"""
def analyze_stft(file: str, fft_size: int, max_num_threads: int) -> dict: ...

"""
Performs the real FFT on an audio array using the aus-rust crate.

:param audio: A 1D array of audio samples
:param fft_size: The FFT size
:return: A tuple with the magnitude spectrum and phase spectrum
"""
def rfft(audio: np.ndarray, fft_size: int) -> Tuple[np.ndarray, np.ndarray]: ...

"""
Performs the inverse real FFT on an audio array using the aus-rust crate.

:param magnitude_spectrum: A 1D array containing the magnitude spectrum
:param phase_spectrum: A 1D array containing the phase spectrum
:param fft_size: The FFT size
:return: The resynthesized audio
"""
def irfft(magnitude_spectrum: np.ndarray, phase_spectrum: np.ndarray, fft_size: int) -> np.ndarray: ...

"""
Performs the real STFT on an audio array using the aus-rust crate.

:param audio: A 1D array of audio samples
:param fft_size: The FFT size
:return: A tuple with the magnitude spectrogram and the phase spectrogram
"""
def rstft(audio: np.ndarray, fft_size: int) -> Tuple[np.ndarray, np.ndarray]: ...

"""
Performs the inverse real STFT on an audio array using the aus-rust crate.

:param magnitude_spectrogram: A 2D array containing the magnitude spectrogram
:param phase_spectrogram: A 2D array containing the phase spectrogram
:param fft_size: The FFT size
:return: The resynthesized audio
"""
def irstft(magnitude_spectrogram: np.ndarray, phase_spectrogram: np.ndarray, fft_size: int) -> np.ndarray: ...
