//! # frame
//! The `frame` module contains analysis functionality for a single audio frame.

use crate::AnalysisError;
use aus::{analysis, spectrum};

const MAX_FRAME_SIZE: usize = 32768;

/// Represents an analysis of a chunk of audio, with no reference to its context.
/// This means that features like spectral difference, which require an additional
/// FFT frame for comparison, are not included.
pub struct FrameAnalysis {
    alpha_ratio: f64,
    autocorrelation: Vec<f64>,
    f0_estimation: Option<f64>,
    hammarberg_index: f64,
    harmonicity: f64,
    mfccs: Vec<f64>,
    power_spectrum: Vec<f64>,
    spectral_centroid: f64,
    spectral_entropy: f64,
    spectral_flatness: f64,
    spectral_kurtosis: f64,
    spectral_rolloff_50: f64,
    spectral_rolloff_75: f64,
    spectral_rolloff_90: f64,
    spectral_rolloff_95: f64,
    spectral_skewness: f64,
    spectral_slope: f64,
    spectral_slope_01khz: f64,
    spectral_slope_15khz: f64,
    spectral_slope_05khz: f64,
    spectral_variance: f64,
    zero_crossing_rate: f64,
}

/// Analyzes a chunk of audio using a suite of analysis tools from `aus`.
/// A FFT will be applied to the entire audio chunk.
/// Because this is not a STFT, this function is arbitrarily limited to 
/// audio chunks of 32,768 frames or fewer (it raises an error for larger chunks).
/// The sample rate is required because it affects pitch analysis. 
/// The fundamental frequency analysis in this package uses the pYin algorithm,
/// which adds a significant time penalty to the analysis process.
/// Because of this, you can specify not to analyze the fundamental frequency (`analyze_f0`).
fn analyze(audio: &Vec<f64>, sample_rate: usize, analyze_f0: bool) -> Result<FrameAnalysis, AnalysisError> {
    if audio.len() > MAX_FRAME_SIZE {
        return Err(AnalysisError { msg: String::from("Cannot analyze audio chunks larger than 32,768 frames.") });
    } else if sample_rate == 0 {
        return Err(AnalysisError { msg: String::from("The sample rate must be greater than 0.") });
    }
    let imaginary_spectrum = spectrum::rfft(audio, audio.len());
    let (magnitude_spectrum, phase_spectrum) = spectrum::complex_to_polar_rfft(&imaginary_spectrum);
    let power_spectrum = analysis::make_power_spectrum(&magnitude_spectrum);
    
    unimplemented!();    
}
