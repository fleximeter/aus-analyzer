use pyo3::prelude::*;
use pyo3::exceptions::{PyIOError, PyValueError};
use aus;
use pyo3::types::PyDict;
use pyo3::Bound;
use numpy::pyo3::Python;
use numpy::{PyArray2, IntoPyArray};
mod analyzer;

#[derive(Debug, Clone)]
struct AnalysisError {
    pub msg: String
}

/// Gets the window type from a string
fn str_to_window(window: &String) -> aus::WindowType {
    match window.to_lowercase().as_str() {
        "hamming" => aus::WindowType::Hamming,
        "hanning" => aus::WindowType::Hanning,
        "bartlett" => aus::WindowType::Bartlett,
        "blackman" => aus::WindowType::Blackman,
        _ => aus::WindowType::Hanning
    }
}

/// Analyzes a magnitude spectrum
#[pyfunction]
fn analyze_rfft(py: Python, magnitude_spectrum: Vec<f64>, fft_size: usize, sample_rate: u32) -> PyResult<Bound<'_, PyDict>> {
    let analysis = aus::analysis::analyzer(&magnitude_spectrum, fft_size, sample_rate);
    let analysis_dict = PyDict::new(py);
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
    match analysis_dict.set_item(String::from("spectral_roll_off_50"), analysis.spectral_roll_off_50) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral roll off 50 could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("spectral_roll_off_75"), analysis.spectral_roll_off_75) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral roll off 75 could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("spectral_roll_off_90"), analysis.spectral_roll_off_90) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral roll off 90 could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("spectral_roll_off_95"), analysis.spectral_roll_off_95) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral roll off 95 could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("spectral_slope"), analysis.spectral_slope) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral slope could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("spectral_slope_0_1_khz"), analysis.spectral_slope_0_1_khz) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral slope 0-1kHz could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("spectral_slope_1_5_khz"), analysis.spectral_slope_1_5_khz) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral slope 1-5kHz could not be added to the analysis dictionary: {}", err)))
    };
    match analysis_dict.set_item(String::from("spectral_slope_0_5_khz"), analysis.spectral_slope_0_5_khz) {
        Ok(_) => (),
        Err(err) => return Err(PyValueError::new_err(format!("The spectral slope 0-5kHz could not be added to the analysis dictionary: {}", err)))
    };
    Ok(analysis_dict)
}

/// Loads an audio file and analyzes it
#[pyfunction]
fn analyze_rstft(py: Python, file: String, fft_size: usize, max_num_threads: usize) -> PyResult<Bound<'_, PyDict>> {
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

    let analysis = analyzer::analyze_audio_file(&mut audio_file.samples[0], fft_size, audio_file.sample_rate, Some(max_num_threads));
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
fn make_analysis_map(py: Python, analysis: analyzer::StftAnalysis) -> Result<Bound<'_, PyDict>, AnalysisError> {
    let arr_len: usize = analysis.analysis.len();
    let analysis_dict = PyDict::new(py);
    let fft_size = analysis.magnitude_spectrogram[0].len();
    let mut magnitude_spectrogram: Vec<Vec<f32>> = Vec::new();
    let mut phase_spectrogram: Vec<Vec<f32>> = Vec::new();
    let mut centroid: Vec<f32> = vec![0.0; arr_len];
    let mut variance: Vec<f32> = vec![0.0; arr_len];
    let mut skewness: Vec<f32> = vec![0.0; arr_len];
    let mut kurtosis: Vec<f32> = vec![0.0; arr_len];
    let mut entropy: Vec<f32> = vec![0.0; arr_len];
    let mut flatness: Vec<f32> = vec![0.0; arr_len];
    let mut roll_50: Vec<f32> = vec![0.0; arr_len];
    let mut roll_75: Vec<f32> = vec![0.0; arr_len];
    let mut roll_90: Vec<f32> = vec![0.0; arr_len];
    let mut roll_95: Vec<f32> = vec![0.0; arr_len];
    let mut slope: Vec<f32> = vec![0.0; arr_len];
    let mut slope01: Vec<f32> = vec![0.0; arr_len];
    let mut slope15: Vec<f32> = vec![0.0; arr_len];
    let mut slope05: Vec<f32> = vec![0.0; arr_len];
    for i in 0..arr_len {
        let mut magnitude_spectrum = vec![0.0; fft_size];
        let mut phase_spectrum = vec![0.0; fft_size];
        for j in 0..fft_size {
            magnitude_spectrum[j] = analysis.magnitude_spectrogram[i][j] as f32;
            phase_spectrum[j] = analysis.phase_spectrogram[i][j] as f32;
        }
        centroid[i] = analysis.analysis[i].spectral_centroid as f32;
        variance[i] = analysis.analysis[i].spectral_variance as f32;
        skewness[i] = analysis.analysis[i].spectral_skewness as f32;
        kurtosis[i] = analysis.analysis[i].spectral_kurtosis as f32;
        entropy[i] = analysis.analysis[i].spectral_entropy as f32;
        flatness[i] = analysis.analysis[i].spectral_flatness as f32;
        roll_50[i] = analysis.analysis[i].spectral_roll_off_50 as f32;
        roll_75[i] = analysis.analysis[i].spectral_roll_off_75 as f32;
        roll_90[i] = analysis.analysis[i].spectral_roll_off_90 as f32;
        roll_95[i] = analysis.analysis[i].spectral_roll_off_95 as f32;
        slope[i] = analysis.analysis[i].spectral_slope as f32;
        slope01[i] = analysis.analysis[i].spectral_slope_0_1_khz as f32;
        slope15[i] = analysis.analysis[i].spectral_slope_1_5_khz as f32;
        slope05[i] = analysis.analysis[i].spectral_slope_0_5_khz as f32;
        magnitude_spectrogram.push(magnitude_spectrum);
        phase_spectrogram.push(phase_spectrum);
    }
    match analysis_dict.set_item(String::from("magnitude_spectrogram"), match PyArray2::from_vec2(py, &magnitude_spectrogram) {
        Ok(x) => x,
        Err(err) => return Err(AnalysisError{msg: err.to_string()})
    }) {
        Ok(_) => (),
        Err(err) => return Err(AnalysisError{msg: err.to_string()})
    }
    match analysis_dict.set_item(String::from("phase_spectrogram"), match PyArray2::from_vec2(py, &phase_spectrogram) {
        Ok(x) => x,
        Err(err) => return Err(AnalysisError{msg: err.to_string()})
    }) {
        Ok(_) => (),
        Err(err) => return Err(AnalysisError{msg: err.to_string()})
    }
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
fn rfft<'py>(py: Python<'py>, audio: Vec<f64>, fft_size: usize) -> PyResult<(Bound<'py, numpy::PyArray<f32, numpy::ndarray::Dim<[usize; 1]>>>, Bound<'py, numpy::PyArray<f32, numpy::ndarray::Dim<[usize; 1]>>>)> {
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
fn irfft<'py>(py: Python<'py>, magnitude_spectrum: Vec<f64>, phase_spectrum: Vec<f64>, fft_size: usize) -> PyResult<Bound<'py, numpy::PyArray<f32, numpy::ndarray::Dim<[usize; 1]>>>> {
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
fn rstft<'py>(py: Python<'py>, audio: Vec<f64>, fft_size: usize, hop_size: usize, window: String) -> PyResult<(Bound<'py, numpy::PyArray<f32, numpy::ndarray::Dim<[usize; 2]>>>, Bound<'py, numpy::PyArray<f32, numpy::ndarray::Dim<[usize; 2]>>>)> {
    let (magnitude_spectrogram, phase_spectrogram) = aus::spectrum::complex_to_polar_rstft(&aus::spectrum::rstft(&audio, fft_size, hop_size, str_to_window(&window)));
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
fn irstft<'py>(py: Python<'py>, magnitude_spectrogram: Vec<Vec<f64>>, phase_spectrogram: Vec<Vec<f64>>, fft_size: usize, hop_size: usize, window: String) -> PyResult<Bound<'py, numpy::PyArray<f32, numpy::ndarray::Dim<[usize; 1]>>>> {
    let imaginary_spectrogram = match aus::spectrum::polar_to_complex_rstft(&magnitude_spectrogram, &phase_spectrogram) {
        Ok(x) => x,
        Err(err) => return Err(PyValueError::new_err(format!("Could not perform the inverse rSTFT: {}", err.error_msg)))
    };
    let audio = match aus::spectrum::irstft(&imaginary_spectrogram, fft_size, hop_size, str_to_window(&window)) {
        Ok(x) => x,
        Err(err) => return Err(PyValueError::new_err(format!("Could not perform the inverse rSTFT: {}", err.error_msg)))
    };
    let mut audio1: Vec<f32> = vec![0.0; audio.len()];
    for i in 0..audio.len() {
        audio1[i] = audio[i] as f32;
    }
    Ok(audio1.into_pyarray(py))
}

/// A module for working with spectral analysis and synthesis in Rust.
#[pymodule]
fn aus_analyzer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(analyze_rfft, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_rstft, m)?)?;
    m.add_function(wrap_pyfunction!(rfft, m)?)?;
    m.add_function(wrap_pyfunction!(irfft, m)?)?;
    m.add_function(wrap_pyfunction!(rstft, m)?)?;
    m.add_function(wrap_pyfunction!(irstft, m)?)?;
    Ok(())
}
