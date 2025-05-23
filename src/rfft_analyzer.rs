//! # rfft_analyzer.rs
//! The `rfft_analyzer` module contains analysis functionality for a single rFFT frame.

use aus::{analysis, spectrum};

/// Represents an analysis of a rFFT frame, with no reference to its context.
/// This means that features like spectral difference, which require an additional
/// FFT frame for comparison, are not included.
pub struct RFFTAnalysis {
    pub alpha_ratio: f64,
    pub hammarberg_index: f64,
    pub harmonicity: f64,
    pub mel_spectrum: Vec<f64>,
    pub mfccs: Vec<f64>,
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
}

/// Analyzes a real FFT frame
pub fn analyzer(magnitude_spectrum: &[f64], fft_size: usize, sample_rate: u32, num_mels: usize, num_mfccs: usize) -> RFFTAnalysis {
    let power_spectrum = analysis::make_power_spectrum(&magnitude_spectrum);
    let rfft_freqs = spectrum::rfftfreq(fft_size, sample_rate);
    let magnitude_spectrum_sum = magnitude_spectrum.iter().sum();
    let power_spectrum_sum = power_spectrum.iter().sum();
    let spectrum_pmf = analysis::make_spectrum_pmf(&power_spectrum, power_spectrum_sum);
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

    // Eyben notes an author that recommends computing the slope of these spectral bands separately.
    let analysis_spectral_slope_0_1_khz = analysis::computation::compute_spectral_slope_region(&power_spectrum, &rfft_freqs, 0.0, 1000.0, sample_rate);
    let analysis_spectral_slope_1_5_khz = analysis::computation::compute_spectral_slope_region(&power_spectrum, &rfft_freqs, 1000.0, 5000.0, sample_rate);
    let analysis_spectral_slope_0_5_khz = analysis::computation::compute_spectral_slope_region(&power_spectrum, &rfft_freqs, 0.0, 5000.0, sample_rate);
    
    let mel_filter = analysis::mel::MelFilterbank::new(0.0, sample_rate as f64 / 2.0, num_mels, &rfft_freqs, true);
    let mel_spectrum = mel_filter.filter(&power_spectrum);
    let mfccs = analysis::mel::mfcc_spectrum(&mel_spectrum, num_mfccs, None);
    
    RFFTAnalysis {
        alpha_ratio: analysis::alpha_ratio(magnitude_spectrum, &rfft_freqs),
        hammarberg_index: analysis::hammarberg_index(magnitude_spectrum, &rfft_freqs),
        harmonicity: analysis::harmonicity(&magnitude_spectrum, true),
        mel_spectrum: mel_spectrum,
        mfccs: mfccs,
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
        spectral_slope_05khz: analysis_spectral_slope_0_5_khz,
        spectral_slope_15khz: analysis_spectral_slope_1_5_khz,
        spectral_variance: analysis_spectral_variance,
    }
}