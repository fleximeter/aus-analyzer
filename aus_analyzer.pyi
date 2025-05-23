import numpy as np
from typing import Tuple

def analyze_frame(audio: np.ndarray, sample_rate: int, num_mels: int, num_mfccs: int, analyze_f0: bool) -> dict:
    """
    Analyzes an audio frame using the `aus-rust` crate. Returns an analysis
    dictionary with analysis features. 

    Parameters
    ----------
    :param audio: An array of audio samples to featurize
    :param sample_rate: The audio sample rate
    :param num_mels: The number of Mel bands
    :param num_mfccs: The number of MFCCs to return
    :param analyze_f0: Whether or not to compute the fundamental frequency analysis (using pYin)

    Returns
    -------
    :return: A dictionary with the analysis
    """
    ...

def analyze_rfft(magnitude_spectrum: np.ndarray, fft_size: int, sample_rate: int, num_mels: int, num_mfccs: int) -> dict:
    """
    Analyzes a real FFT frame using the `aus-rust` crate. Returns an analysis
    dictionary with spectral analysis features. 

    Parameters
    ----------
    :param magnitude_spectrum: The real FFT magnitude spectrum to analyze
    :param fft_size: The FFT size
    :param sample_rate: The audio sample rate
    :param num_mels: The number of Mel bands
    :param num_mfccs: The number of MFCCs to return

    Returns
    -------
    :return: A dictionary with the spectral analysis and the STFT data
    """
    ...

def analyze_rstft(file: str, fft_size: int, num_mels: int = 128, num_mfccs: int = 20, max_num_threads: int = 4) -> dict:
    """
    Analyzes an audio file using the `aus-rust` crate.
    Loads the audio file, runs the STFT, and extracts spectral features.
    The analysis dictionary that is returned also contains the STFT magnitude and phase spectrograms.
    Each spectral feature is provided in an array where element 0 corresponds to STFT frame 0,
    element 1 corresponds to STFT frame 1, etc.

    Parameters
    ----------
    :param file: The file name to load and analyze
    :param fft_size: The FFT size for the STFT
    :param num_mels: The number of Mel bands
    :param num_mfccs: The number of MFCCs
    :param max_num_threads: The maximum number of threads to use (if set to 0, this will be set to the maximum allowed)

    Returns
    -------
    :return: A dictionary with the spectral analysis and the STFT data
    """
    ...

def rfft(audio: np.ndarray, fft_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs the real FFT on an audio array using the `aus-rust` crate. This is really
    a wrapper function for `rustfft`. For efficiency, this function also decomposes
    the complex spectrum into magnitude and phase spectra.

    Parameters
    ----------
    :param audio: A 1D array of audio samples
    :param fft_size: The FFT size
    
    Returns
    -------
    :return: A tuple with the magnitude spectrum and phase spectrum
    """
    ...

def irfft(magnitude_spectrum: np.ndarray, phase_spectrum: np.ndarray, fft_size: int) -> np.ndarray:
    """
    Performs the inverse real FFT on an audio array using the aus-rust crate. This is really
    a wrapper function for `rustfft`.

    Parameters
    ----------
    :param magnitude_spectrum: A 1D array containing the magnitude spectrum
    :param phase_spectrum: A 1D array containing the phase spectrum
    :param fft_size: The FFT size

    Returns
    -------
    :return: The resynthesized audio
    """
    ...

def rstft(audio: np.ndarray, fft_size: int, hop_size: int, window: str = "hanning") -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs the real STFT on an audio array using the aus-rust crate. The STFT is performed
    using `rustfft`. For efficiency, this function also decomposes
    the complex spectrogram into magnitude and phase spectrograms.

    Parameters
    ----------
    :param audio: A 1D array of audio samples
    :param fft_size: The FFT size
    :param hop_size: The hop size
    :param window: The window type (`"hanning"`, `"hanning"`, `"bartlett"`, `"blackman"`). If something else is provided, the window type will default to `"hanning"`.

    Returns
    -------
    :return: A tuple with the magnitude spectrogram and the phase spectrogram
    """
    ...

def irstft(magnitude_spectrogram: np.ndarray, phase_spectrogram: np.ndarray, fft_size: int, hop_size: int, window: str = "hanning") -> np.ndarray:
    """
    Performs the inverse real STFT on an audio array using the aus-rust crate. The ISTFT is performed
    using `rustfft`.

    Parameters
    ----------
    :param magnitude_spectrogram: A 2D array containing the magnitude spectrogram
    :param phase_spectrogram: A 2D array containing the phase spectrogram
    :param fft_size: The FFT size
    :param hop_size: The hop size
    :param window: The window type (`"hanning"`, `"hanning"`, `"bartlett"`, `"blackman"`). If something else is provided, the window type will default to `"hanning"`.

    Returns
    -------
    :return: The resynthesized audio
    """
    ...
