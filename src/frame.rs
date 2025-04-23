//! # frame
//! The `frame` module contains analysis functionality for a single audio frame.

use crate::AnalysisError;
use aus::{analysis, spectrum};

const MAX_FRAME_SIZE: usize = 32768;

/// Represents an analysis of a chunk of audio, with no reference to its context.
/// This means that features like spectral difference, which require an additional
/// FFT frame for comparison, are not included.
pub struct FrameAnalysis {
    pub alpha_ratio: f64,
    pub autocorrelation: Vec<f64>,
    pub f0_estimation: Option<f64>,
    pub hammarberg_index: f64,
    pub harmonicity: f64,
    pub mfccs: Vec<f64>,
    pub power_spectrum: Vec<f64>,
    pub spectral_centroid: f64,
    pub spectral_entropy: f64,
    pub spectral_flatness: f64,
    pub spectral_kurtosis: f64,
    pub spectral_rolloff_50: f64,
    pub spectral_rolloff_75: f64,
    pub spectral_rolloff_90: f64,
    pub spectral_rolloff_95: f64,
    pub spectral_skewness: f64,
    pub spectral_slope: f64,
    pub spectral_slope_01khz: f64,
    pub spectral_slope_15khz: f64,
    pub spectral_slope_05khz: f64,
    pub spectral_variance: f64,
    pub zero_crossing_rate: f64,
}

/// Analyzes a chunk of audio using a suite of analysis tools from `aus`.
/// A FFT will be applied to the entire audio chunk.
/// Because this is not a STFT, this function is arbitrarily limited to 
/// audio chunks of 32,768 frames or fewer (it raises an error for larger chunks).
/// The sample rate is required because it affects pitch analysis. 
/// The fundamental frequency analysis in this package uses the pYin algorithm,
/// which adds a significant time penalty to the analysis process.
/// Because of this, you can specify not to analyze the fundamental frequency (`analyze_f0`).
pub fn analyze(audio: &Vec<f64>, sample_rate: u32, analyze_f0: bool) -> Result<FrameAnalysis, AnalysisError> {
    // handle error cases
    if audio.len() > MAX_FRAME_SIZE {
        return Err(AnalysisError { msg: String::from("Cannot analyze audio chunks larger than 32,768 frames.") });
    } else if sample_rate == 0 {
        return Err(AnalysisError { msg: String::from("The sample rate must be greater than 0.") });
    }

    // produce the spectrum
    let fft_size = audio.len();
    let window = aus::generate_window_hamming(fft_size);

    // NOTE: DOUBLE CHECK THAT THIS WINDOWING WORKS!
    let windowed_audio = audio.iter().zip(window.iter()).map(|(x, y)| x * y).collect::<Vec<_>>();
    let imaginary_spectrum = spectrum::rfft(&windowed_audio, fft_size);
    let (magnitude_spectrum, _) = spectrum::complex_to_polar_rfft(&imaginary_spectrum);
    let magnitude_spectrum_sum = magnitude_spectrum.iter().sum();
    let power_spectrum = analysis::make_power_spectrum(&magnitude_spectrum);
    let power_spectrum_sum = power_spectrum.iter().sum();
    let spectrum_pmf = analysis::make_spectrum_pmf(&power_spectrum, power_spectrum_sum);
    let rfft_freqs = spectrum::rfftfreq(fft_size, sample_rate);

    // produce the spectral analysis results
    let analysis_spectral_centroid = analysis::computation::compute_spectral_centroid(&magnitude_spectrum, &rfft_freqs, magnitude_spectrum_sum);
    let analysis_spectral_variance = analysis::computation::compute_spectral_variance(&spectrum_pmf, &rfft_freqs, analysis_spectral_centroid);
    let analysis_spectral_skewness = analysis::computation::compute_spectral_skewness(&spectrum_pmf, &rfft_freqs, analysis_spectral_centroid, analysis_spectral_variance);
    let analysis_spectral_kurtosis = analysis::computation::compute_spectral_kurtosis(&spectrum_pmf, &rfft_freqs, analysis_spectral_centroid, analysis_spectral_variance);
    let analysis_spectral_entropy = analysis::computation::compute_spectral_entropy(&spectrum_pmf);
    let analysis_spectral_flatness = analysis::computation::compute_spectral_flatness(&magnitude_spectrum, magnitude_spectrum_sum);
    let analysis_spectral_roll_off_50 = analysis::computation::compute_spectral_roll_off_point(&power_spectrum, &rfft_freqs, power_spectrum_sum, 0.5);
    let analysis_spectral_roll_off_75 = analysis::computation::compute_spectral_roll_off_point(&power_spectrum, &rfft_freqs, power_spectrum_sum, 0.75);
    let analysis_spectral_roll_off_90 = analysis::computation::compute_spectral_roll_off_point(&power_spectrum, &rfft_freqs, power_spectrum_sum, 0.9);
    let analysis_spectral_roll_off_95 = analysis::computation::compute_spectral_roll_off_point(&power_spectrum, &rfft_freqs, power_spectrum_sum, 0.95);
    let analysis_spectral_slope = analysis::computation::compute_spectral_slope(&power_spectrum, power_spectrum_sum);
    let analysis_spectral_slope_0_1_khz = analysis::computation::compute_spectral_slope_region(&power_spectrum, &rfft_freqs, 0.0, 1000.0, sample_rate);
    let analysis_spectral_slope_1_5_khz = analysis::computation::compute_spectral_slope_region(&power_spectrum, &rfft_freqs, 1000.0, 5000.0, sample_rate);
    let analysis_spectral_slope_0_5_khz = analysis::computation::compute_spectral_slope_region(&power_spectrum, &rfft_freqs, 0.0, 5000.0, sample_rate);
    let hammarberg_index = analysis::hammarberg_index(&magnitude_spectrum, &rfft_freqs);
    let alpha_ratio = analysis::alpha_ratio(&magnitude_spectrum, &rfft_freqs);
    let harmonicity = analysis::harmonicity(&magnitude_spectrum, true);
    let autocorrelation = match analysis::autocorrelation(&audio, fft_size) {
        Ok(autocor) => autocor,
        Err(err) => return Err(AnalysisError { msg: err.error_msg })
    };
    let mel_spectrum = analysis::mel::make_mel_spectrum(&power_spectrum, analysis::mel::freq_to_mel(20.0), analysis::mel::freq_to_mel(8000.0), 26, &rfft_freqs);
    let log_spectrum: Vec<f64> = analysis::make_log_spectrum(&mel_spectrum.0, 10e-8);
    let mfccs = analysis::mel::mfcc(&log_spectrum, 2.0); // then use indices 11-15
    
    // optionally produce fundamental frequency
    let f0 = match analyze_f0 {
        true => Some(analysis::pyin_pitch_estimator_single(&audio, sample_rate, 50.0, 5000.0)),
        false => None
    };

    // get raw audio features
    let zcr = analysis::zero_crossing_rate(&audio, sample_rate);

    Ok(FrameAnalysis {
        alpha_ratio: alpha_ratio,
        autocorrelation: autocorrelation,
        f0_estimation: f0,
        hammarberg_index: hammarberg_index,
        harmonicity: harmonicity,
        mfccs: mfccs,
        power_spectrum: power_spectrum,
        spectral_centroid: analysis_spectral_centroid,
        spectral_entropy: analysis_spectral_entropy,
        spectral_flatness: analysis_spectral_flatness,
        spectral_kurtosis: analysis_spectral_kurtosis,
        spectral_rolloff_50: analysis_spectral_roll_off_50,
        spectral_rolloff_75: analysis_spectral_roll_off_75,
        spectral_rolloff_90: analysis_spectral_roll_off_90,
        spectral_rolloff_95: analysis_spectral_roll_off_95,
        spectral_skewness: analysis_spectral_skewness,
        spectral_slope: analysis_spectral_slope,
        spectral_slope_01khz: analysis_spectral_slope_0_1_khz,
        spectral_slope_15khz: analysis_spectral_slope_1_5_khz,
        spectral_slope_05khz: analysis_spectral_slope_0_5_khz,
        spectral_variance: analysis_spectral_variance,
        zero_crossing_rate: zcr
    })
}
