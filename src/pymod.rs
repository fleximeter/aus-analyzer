use pyo3::prelude::*;
use pyo3::exceptions::{PyIOError, PyValueError};
use aus;
use pyo3::types::PyDict;
use pyo3::Bound;
use numpy::pyo3::Python;
use numpy::{PyArray2, IntoPyArray};
use crate::rstft_analyzer;
use crate::AnalysisError;
use crate::frame;
use crate::rfft_analyzer;

/// A module for working with spectral analysis and synthesis in Rust.
#[pymodule]
pub fn aus_analyzer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(analyze_frame, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_rfft, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_rstft, m)?)?;
    m.add_function(wrap_pyfunction!(rfft, m)?)?;
    m.add_function(wrap_pyfunction!(irfft, m)?)?;
    m.add_function(wrap_pyfunction!(rstft, m)?)?;
    m.add_function(wrap_pyfunction!(irstft, m)?)?;
    Ok(())
}

/// Gets the window type from a string
fn str_to_window_type(window: &str) -> aus::WindowType {
    match window.to_lowercase().as_str() {
        "hamming" => aus::WindowType::Hamming,
        "hanning" => aus::WindowType::Hanning,
        "bartlett" => aus::WindowType::Bartlett,
        "blackman" => aus::WindowType::Blackman,
        _ => aus::WindowType::Hanning
    }
}

/// Analyzes an audio chunk
#[pyfunction]
pub fn analyze_frame(py: Python, audio: Vec<f64>, sample_rate: u32, num_mels: usize, num_mfccs: usize, analyze_f0: bool) -> PyResult<Bound<'_, PyDict>> {
    let analysis = match frame::analyze(&audio, sample_rate, num_mels, num_mfccs, analyze_f0) {
        Ok(a) => a,
        Err(err) => return Err(PyValueError::new_err(err.msg))
    };
    let analysis_dict = PyDict::new(py);
    match analysis_dict.set_item(String::from("alpha_ratio"), analysis.alpha_ratio) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The alpha ratio could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("autocorrelation"), analysis.autocorrelation) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The autocorrelation could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("f0_estimation"), analysis.f0_estimation) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The f0 estimation could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("hammarberg_index"), analysis.hammarberg_index) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The hammarberg index could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("harmonicity"), analysis.harmonicity) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The harmonicity could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("mel_spectrum"), analysis.mel_spectrum) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The Mel spectrum could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("mfccs"), analysis.mfccs) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The MFCCs could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("power_spectrum"), analysis.power_spectrum) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The power spectrum could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("spectral_centroid"), analysis.spectral_centroid) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral centroid could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("spectral_entropy"), analysis.spectral_entropy) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral entropy could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("spectral_flatness"), analysis.spectral_flatness) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral flatness could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("spectral_kurtosis"), analysis.spectral_kurtosis) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral kurtosis could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("spectral_roll_off_50"), analysis.spectral_rolloff_50) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral roll off 50 could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("spectral_roll_off_75"), analysis.spectral_rolloff_75) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral roll off 75 could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("spectral_roll_off_90"), analysis.spectral_rolloff_90) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral roll off 90 could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("spectral_roll_off_95"), analysis.spectral_rolloff_95) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral roll off 95 could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("spectral_skewness"), analysis.spectral_skewness) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral skewness could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("spectral_slope"), analysis.spectral_slope) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral slope could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("spectral_slope_0_1_khz"), analysis.spectral_slope_01khz) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral slope 0-1kHz could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("spectral_slope_1_5_khz"), analysis.spectral_slope_15khz) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral slope 1-5kHz could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("spectral_slope_0_5_khz"), analysis.spectral_slope_05khz) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral slope 0-5kHz could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("spectral_variance"), analysis.spectral_variance) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral variance could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("zero_crossing_rate"), analysis.zero_crossing_rate) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The zero crossing rate could not be added to the analysis dictionary: {}", err)))
    };
    Ok(analysis_dict)
}

/// Analyzes a magnitude spectrum
#[pyfunction]
pub fn analyze_rfft(py: Python, magnitude_spectrum: Vec<f64>, fft_size: usize, sample_rate: u32, num_mels: usize, num_mfccs: usize) -> PyResult<Bound<'_, PyDict>> {
    let analysis = rfft_analyzer::analyzer(&magnitude_spectrum, fft_size, sample_rate, num_mels, num_mfccs);
    let analysis_dict = PyDict::new(py);
    match analysis_dict.set_item(String::from("alpha_ratio"), analysis.alpha_ratio) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The alpha ratio could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("hammarberg_index"), analysis.hammarberg_index) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The Hammarberg index could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("harmonicity"), analysis.harmonicity) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The harmonicity could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("mel_spectrum"), analysis.mel_spectrum) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The Mel spectrum could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("mfccs"), analysis.mfccs) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The MFCCs could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("spectral_centroid"), analysis.spectral_centroid) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral centroid could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("spectral_variance"), analysis.spectral_variance) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral variance could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("spectral_skewness"), analysis.spectral_skewness) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral skewness could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("spectral_kurtosis"), analysis.spectral_kurtosis) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral kurtosis could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("spectral_entropy"), analysis.spectral_entropy) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral entropy could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("spectral_flatness"), analysis.spectral_flatness) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral flatness could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("spectral_roll_off_50"), analysis.spectral_rolloff_50) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral roll off 50 could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("spectral_roll_off_75"), analysis.spectral_rolloff_75) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral roll off 75 could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("spectral_roll_off_90"), analysis.spectral_rolloff_90) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral roll off 90 could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("spectral_roll_off_95"), analysis.spectral_rolloff_95) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral roll off 95 could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("spectral_slope"), analysis.spectral_slope) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral slope could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("spectral_slope_0_1_khz"), analysis.spectral_slope_01khz) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral slope 0-1kHz could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("spectral_slope_1_5_khz"), analysis.spectral_slope_15khz) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral slope 1-5kHz could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("spectral_slope_0_5_khz"), analysis.spectral_slope_05khz) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral slope 0-5kHz could not be added to the analysis dictionary: {}", err)))
    };
    Ok(analysis_dict)
}

/// Loads an audio file and analyzes it
#[pyfunction]
#[pyo3(signature = (file, fft_size, num_mels=128, num_mfccs=20, max_num_threads=4))]
pub fn analyze_rstft(py: Python, file: String, fft_size: usize, num_mels: usize, num_mfccs: usize, max_num_threads: usize) -> PyResult<Bound<'_, PyDict>> {
    let mut audio_file = match aus::read(&file) {
        Ok(file) => file,
        Err(err) => {
            return match err {
                aus::AudioError::FileCorrupt => {
                    Err(PyIOError::new_err("The audio file ({}) was corrupt and could not be read."))
                },
                aus::AudioError::FileInaccessible(msg) => {
                    Err(PyIOError::new_err(format!("The file ({}) was inaccessible: {}", file, msg)))
                },
                aus::AudioError::NumChannels(msg) => {
                    Err(PyIOError::new_err(format!("The file ({}) had a channel error: {}", file, msg)))
                },
                aus::AudioError::SampleValueOutOfRange(msg) => {
                    Err(PyIOError::new_err(format!("The file ({}) had a sample value out of range: {}", file, msg)))
                },
                aus::AudioError::NumFrames(msg) => {
                    Err(PyIOError::new_err(format!("The file ({}) had a frame number problem: {}", file, msg)))
                },
                aus::AudioError::WrongFormat(msg) => {
                    Err(PyIOError::new_err(format!("The file ({}) had a format issue: {}", file, msg)))
                }
            };
        }
    };

    let analysis = rstft_analyzer::analyze(&mut audio_file.samples[0], fft_size, fft_size / 2, audio_file.sample_rate, num_mels, num_mfccs, Some(max_num_threads));
    let analysis_map = match make_analysis_map(py, analysis) {
        Ok(analysis) => analysis,
        Err(err) => return Err(PyValueError::new_err(format!("The analysis dictionary could not be created: {}", err.msg)))
    };
    Ok(analysis_map)
}

/// Converts a Vector of Analysis structs to a PyDict.
/// Each Analysis entry gets added to a vector containing all entries like it.
/// For example, the PyDict will contain a key called "spectral_centroid" corresponding
/// to an array of the spectral centroids.
fn make_analysis_map(py: Python, analysis: rstft_analyzer::StftAnalysis) -> Result<Bound<'_, PyDict>, AnalysisError> {
    let num_analysis_frames: usize = analysis.analyses.len();
    let analysis_dict = PyDict::new(py);
    let mut power_spectrogram: Vec<Vec<f32>> = Vec::new();
    let mut mel_spectrogram: Vec<Vec<f32>> = Vec::new();
    let mut mfccs: Vec<Vec<f32>> = Vec::new();
    let mut alpha_ratio: Vec<f32> = vec![0.0; num_analysis_frames];
    let mut hammarberg_index: Vec<f32> = vec![0.0; num_analysis_frames];
    let mut harmonicity: Vec<f32> = vec![0.0; num_analysis_frames];
    let mut difference: Vec<f32> = vec![0.0; num_analysis_frames];
    let mut flux: Vec<f32> = vec![0.0; num_analysis_frames];
    let mut centroid: Vec<f32> = vec![0.0; num_analysis_frames];
    let mut variance: Vec<f32> = vec![0.0; num_analysis_frames];
    let mut skewness: Vec<f32> = vec![0.0; num_analysis_frames];
    let mut kurtosis: Vec<f32> = vec![0.0; num_analysis_frames];
    let mut entropy: Vec<f32> = vec![0.0; num_analysis_frames];
    let mut flatness: Vec<f32> = vec![0.0; num_analysis_frames];
    let mut roll_50: Vec<f32> = vec![0.0; num_analysis_frames];
    let mut roll_75: Vec<f32> = vec![0.0; num_analysis_frames];
    let mut roll_90: Vec<f32> = vec![0.0; num_analysis_frames];
    let mut roll_95: Vec<f32> = vec![0.0; num_analysis_frames];
    let mut slope: Vec<f32> = vec![0.0; num_analysis_frames];
    let mut slope01: Vec<f32> = vec![0.0; num_analysis_frames];
    let mut slope15: Vec<f32> = vec![0.0; num_analysis_frames];
    let mut slope05: Vec<f32> = vec![0.0; num_analysis_frames];
    for i in 0..analysis.power_spectrogram.len() {
        let mut magnitude_spectrum: Vec<f32> = Vec::with_capacity(analysis.power_spectrogram[i].len());
        for j in 0..analysis.power_spectrogram[i].len() {
            magnitude_spectrum.push(analysis.power_spectrogram[i][j] as f32);
        }
        power_spectrogram.push(magnitude_spectrum);
    }
    for i in 0..analysis.mel_spectrogram.len() {
        let mut mel_spectrum: Vec<f32> = Vec::with_capacity(analysis.mel_spectrogram[i].len());        
        for j in 0..analysis.mel_spectrogram[i].len() {
            mel_spectrum.push(analysis.mel_spectrogram[i][j] as f32);
        }
        mel_spectrogram.push(mel_spectrum);
    }
    for i in 0..analysis.mfccs.len() {
        let mut mfcc_frame: Vec<f32> = Vec::with_capacity(analysis.mfccs[i].len());        
        for j in 0..analysis.mfccs[i].len() {
            mfcc_frame.push(analysis.mfccs[i][j] as f32);
        }
        mfccs.push(mfcc_frame);
    }
    for i in 0..num_analysis_frames {
        alpha_ratio[i] = analysis.analyses[i].alpha_ratio as f32;
        hammarberg_index[i] = analysis.analyses[i].hammarberg_index as f32;
        harmonicity[i] = analysis.analyses[i].harmonicity as f32;
        difference[i] = analysis.analyses[i].spectral_difference as f32;
        flux[i] = analysis.analyses[i].spectral_flux as f32;
        centroid[i] = analysis.analyses[i].spectral_centroid as f32;
        variance[i] = analysis.analyses[i].spectral_variance as f32;
        skewness[i] = analysis.analyses[i].spectral_skewness as f32;
        kurtosis[i] = analysis.analyses[i].spectral_kurtosis as f32;
        entropy[i] = analysis.analyses[i].spectral_entropy as f32;
        flatness[i] = analysis.analyses[i].spectral_flatness as f32;
        roll_50[i] = analysis.analyses[i].spectral_rolloff_50 as f32;
        roll_75[i] = analysis.analyses[i].spectral_rolloff_75 as f32;
        roll_90[i] = analysis.analyses[i].spectral_rolloff_90 as f32;
        roll_95[i] = analysis.analyses[i].spectral_rolloff_95 as f32;
        slope[i] = analysis.analyses[i].spectral_slope as f32;
        slope01[i] = analysis.analyses[i].spectral_slope_01khz as f32;
        slope15[i] = analysis.analyses[i].spectral_slope_15khz as f32;
        slope05[i] = analysis.analyses[i].spectral_slope_05khz as f32;
    }
    match analysis_dict.set_item(String::from("power_spectrogram"), match PyArray2::from_vec2(py, &power_spectrogram) {
        Ok(x) => x,
        Err(err) => return Err(AnalysisError{msg: err.to_string()})
    }) {
        Ok(_) => (),
        Err(err) => return Err(AnalysisError{msg: err.to_string()})
    }
    match analysis_dict.set_item(String::from("mel_spectrogram"), match PyArray2::from_vec2(py, &mel_spectrogram) {
        Ok(x) => x,
        Err(err) => return Err(AnalysisError{msg: err.to_string()})
    }) {
        Ok(_) => (),
        Err(err) => return Err(AnalysisError{msg: err.to_string()})
    }
    match analysis_dict.set_item(String::from("mfccs"), match PyArray2::from_vec2(py, &mfccs) {
        Ok(x) => x,
        Err(err) => return Err(AnalysisError{msg: err.to_string()})
    }) {
        Ok(_) => (),
        Err(err) => return Err(AnalysisError{msg: err.to_string()})
    }
    match analysis_dict.set_item(String::from("alpha_ratio"), alpha_ratio.into_pyarray(py).to_owned()) {
        Ok(_) => (),
        Err(err) => return Err(AnalysisError{msg: err.to_string()})
    };
    match analysis_dict.set_item(String::from("hammarberg_index"), hammarberg_index.into_pyarray(py).to_owned()) {
        Ok(_) => (),
        Err(err) => return Err(AnalysisError{msg: err.to_string()})
    };
    match analysis_dict.set_item(String::from("harmonicity"), harmonicity.into_pyarray(py).to_owned()) {
        Ok(_) => (),
        Err(err) => return Err(AnalysisError{msg: err.to_string()})
    };
    match analysis_dict.set_item(String::from("spectral_difference"), difference.into_pyarray(py).to_owned()) {
        Ok(_) => (),
        Err(err) => return Err(AnalysisError{msg: err.to_string()})
    };
    match analysis_dict.set_item(String::from("spectral_flux"), flux.into_pyarray(py).to_owned()) {
        Ok(_) => (),
        Err(err) => return Err(AnalysisError{msg: err.to_string()})
    };
    match analysis_dict.set_item(String::from("spectral_centroid"), centroid.into_pyarray(py).to_owned()) {
        Ok(_) => (),
        Err(err) => return Err(AnalysisError{msg: err.to_string()})
    };
    match analysis_dict.set_item(String::from("spectral_variance"), variance.into_pyarray(py).to_owned()) {
        Ok(_) => (),
        Err(err) => return Err(AnalysisError{msg: err.to_string()})
    };
    match analysis_dict.set_item(String::from("spectral_skewness"), skewness.into_pyarray(py).to_owned()) {
        Ok(_) => (),
        Err(err) => return Err(AnalysisError{msg: err.to_string()})
    };
    match analysis_dict.set_item(String::from("spectral_kurtosis"), kurtosis.into_pyarray(py).to_owned()) {
        Ok(_) => (),
        Err(err) => return Err(AnalysisError{msg: err.to_string()})
    };
    match analysis_dict.set_item(String::from("spectral_entropy"), entropy.into_pyarray(py).to_owned()) {
        Ok(_) => (),
        Err(err) => return Err(AnalysisError{msg: err.to_string()})
    };
    match analysis_dict.set_item(String::from("spectral_flatness"), flatness.into_pyarray(py).to_owned()) {
        Ok(_) => (),
        Err(err) => return Err(AnalysisError{msg: err.to_string()})
    };
    match analysis_dict.set_item(String::from("spectral_roll_off_50"), roll_50.into_pyarray(py).to_owned()) {
        Ok(_) => (),
        Err(err) => return Err(AnalysisError{msg: err.to_string()})
    };
    match analysis_dict.set_item(String::from("spectral_roll_off_75"), roll_75.into_pyarray(py).to_owned()) {
        Ok(_) => (),
        Err(err) => return Err(AnalysisError{msg: err.to_string()})
    };
    match analysis_dict.set_item(String::from("spectral_roll_off_90"), roll_90.into_pyarray(py).to_owned()) {
        Ok(_) => (),
        Err(err) => return Err(AnalysisError{msg: err.to_string()})
    };
    match analysis_dict.set_item(String::from("spectral_roll_off_95"), roll_95.into_pyarray(py).to_owned()) {
        Ok(_) => (),
        Err(err) => return Err(AnalysisError{msg: err.to_string()})
    };
    match analysis_dict.set_item(String::from("spectral_slope"), slope.into_pyarray(py).to_owned()) {
        Ok(_) => (),
        Err(err) => return Err(AnalysisError{msg: err.to_string()})
    };
    match analysis_dict.set_item(String::from("spectral_slope_0_1_khz"), slope01.into_pyarray(py).to_owned()) {
        Ok(_) => (),
        Err(err) => return Err(AnalysisError{msg: err.to_string()})
    };
    match analysis_dict.set_item(String::from("spectral_slope_1_5_khz"), slope15.into_pyarray(py).to_owned()) {
        Ok(_) => (),
        Err(err) => return Err(AnalysisError{msg: err.to_string()})
    };
    match analysis_dict.set_item(String::from("spectral_slope_0_5_khz"), slope05.into_pyarray(py).to_owned()) {
        Ok(_) => (),
        Err(err) => return Err(AnalysisError{msg: err.to_string()})
    };
    Ok(analysis_dict)
}

/// Computes the real FFT using the aus library
#[pyfunction]
pub fn rfft<'py>(py: Python<'py>, audio: Vec<f64>, fft_size: usize) -> PyResult<(Bound<'py, numpy::PyArray<f32, numpy::ndarray::Dim<[usize; 1]>>>, Bound<'py, numpy::PyArray<f32, numpy::ndarray::Dim<[usize; 1]>>>)> {
    let (magnitude_spectrum, phase_spectrum) = aus::spectrum::complex_to_polar_rfft(&aus::spectrum::rfft(&audio, fft_size));
    let mut magspectrum1: Vec<f32> = vec![0.0; magnitude_spectrum.len()];
    let mut phasespectrum1: Vec<f32> = vec![0.0; phase_spectrum.len()];
    for i in 0..magnitude_spectrum.len() {
        magspectrum1[i] = magnitude_spectrum[i] as f32;
        phasespectrum1[i] = phase_spectrum[i] as f32;
    }
    let x: (Bound<'py, numpy::PyArray<f32, numpy::ndarray::Dim<[usize; 1]>>>, Bound<'py, numpy::PyArray<f32, numpy::ndarray::Dim<[usize; 1]>>>)  = (magspectrum1.into_pyarray(py), phasespectrum1.into_pyarray(py));
    Ok(x)
}

/// Computes the inverse real FFT using the aus library
#[pyfunction]
pub fn irfft<'py>(py: Python<'py>, magnitude_spectrum: Vec<f64>, phase_spectrum: Vec<f64>, fft_size: usize) -> PyResult<Bound<'py, numpy::PyArray<f32, numpy::ndarray::Dim<[usize; 1]>>>> {
    let imaginary_spectrum = match aus::spectrum::polar_to_complex_rfft(&magnitude_spectrum, &phase_spectrum) {
        Ok(x) => x,
        Err(err) => return Err(PyValueError::new_err(format!("Could not perform the inverse rFFT: {}", err.error_msg)))
    };
    let audio = match aus::spectrum::irfft(&imaginary_spectrum, fft_size) {
        Ok(x) => x,
        Err(err) => return Err(PyValueError::new_err(format!("Could not perform the inverse rFFT: {}", err.error_msg)))
    };
    let mut audio1: Vec<f32> = vec![0.0; audio.len()];
    for i in 0..audio.len() {
        audio1[i] = audio[i] as f32;
    }
    Ok(audio1.into_pyarray(py))
}

/// Computes the real STFT using the aus library
#[pyfunction]
#[pyo3(signature = (audio, fft_size, hop_size, window="hanning"))]
pub fn rstft<'py>(py: Python<'py>, audio: Vec<f64>, fft_size: usize, hop_size: usize, window: &str) -> PyResult<(Bound<'py, numpy::PyArray<f32, numpy::ndarray::Dim<[usize; 2]>>>, Bound<'py, numpy::PyArray<f32, numpy::ndarray::Dim<[usize; 2]>>>)> {
    let (magnitude_spectrogram, phase_spectrogram) = aus::spectrum::complex_to_polar_rstft(&aus::spectrum::rstft(&audio, fft_size, hop_size, str_to_window_type(window)));
    let mut magspectrogram1: Vec<Vec<f32>> = Vec::new();
    let mut phasespectrogram1: Vec<Vec<f32>> = Vec::new();
    for i in 0..magnitude_spectrogram.len() {
        let mut magspectrum: Vec<f32> = vec![0.0; magnitude_spectrogram[i].len()];
        let mut phasespectrum: Vec<f32> = vec![0.0; fft_size / 2 + 1];
        for j in 0..magnitude_spectrogram[i].len() {
            magspectrum[j] = magnitude_spectrogram[i][j] as f32;
            phasespectrum[j] = phase_spectrogram[i][j] as f32;
        }
        magspectrogram1.push(magspectrum);
        phasespectrogram1.push(phasespectrum);
    }
    let x: (Bound<'py, numpy::PyArray<f32, numpy::ndarray::Dim<[usize; 2]>>>, Bound<'py, numpy::PyArray<f32, numpy::ndarray::Dim<[usize; 2]>>>) = (
        match PyArray2::from_vec2(py, &magspectrogram1) {
        Ok(y) => y,
        Err(err) => return Err(PyValueError::new_err(format!("Could not create the Python array: {}", err.to_string())))
    }, match PyArray2::from_vec2(py, &phasespectrogram1) {
        Ok(y) => y,
        Err(err) => return Err(PyValueError::new_err(format!("Could not create the Python array: {}", err.to_string())))
    });
    Ok(x)
}

/// Computes the inverse real STFT using the aus library
#[pyfunction]
#[pyo3(signature = (magnitude_spectrogram, phase_spectrogram, fft_size, hop_size, window="hanning"))]
pub fn irstft<'py>(py: Python<'py>, magnitude_spectrogram: Vec<Vec<f64>>, phase_spectrogram: Vec<Vec<f64>>, fft_size: usize, hop_size: usize, window: &str) -> PyResult<Bound<'py, numpy::PyArray<f32, numpy::ndarray::Dim<[usize; 1]>>>> {
    let imaginary_spectrogram = match aus::spectrum::polar_to_complex_rstft(&magnitude_spectrogram, &phase_spectrogram) {
        Ok(x) => x,
        Err(err) => return Err(PyValueError::new_err(format!("Could not perform the inverse rSTFT: {}", err.error_msg)))
    };
    let audio = match aus::spectrum::irstft(&imaginary_spectrogram, fft_size, hop_size, str_to_window_type(window)) {
        Ok(x) => x,
        Err(err) => return Err(PyValueError::new_err(format!("Could not perform the inverse rSTFT: {}", err.error_msg)))
    };
    let mut audio1: Vec<f32> = vec![0.0; audio.len()];
    for i in 0..audio.len() {
        audio1[i] = audio[i] as f32;
    }
    Ok(audio1.into_pyarray(py))
}


