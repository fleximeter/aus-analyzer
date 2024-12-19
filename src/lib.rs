use aus::analysis::Analysis;
use pyo3::prelude::*;
use pyo3::exceptions::PyIOError;
use aus;
use pyo3::types::PyDict;
use numpy::pyo3::Python;
use numpy::IntoPyArray;

/// Loads an audio file and analyzes it
#[pyfunction]
fn analyze(py: Python, file: String, fft_size: usize, max_num_threads: usize) -> PyResult<Bound<'_, PyDict>> {
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

    let analysis = aus::mp::stft_analysis(&mut audio_file.samples[0], fft_size, audio_file.sample_rate, Some(max_num_threads));
    let analysis_map = make_analysis_map(py, analysis);
    Ok(analysis_map)
}

// Converts a Vector of Analysis structs to a HashMap
fn make_analysis_map(py: Python, analysis: Vec<Analysis>) -> Bound<'_, PyDict> {
    let arr_len: usize = analysis.len();
    let analysis_dict = PyDict::new(py);
    let mut centroid = vec![0.0; arr_len];
    let mut variance = vec![0.0; arr_len];
    let mut skewness = vec![0.0; arr_len];
    let mut kurtosis = vec![0.0; arr_len];
    let mut entropy = vec![0.0; arr_len];
    let mut flatness = vec![0.0; arr_len];
    let mut roll_50 = vec![0.0; arr_len];
    let mut roll_75 = vec![0.0; arr_len];
    let mut roll_90 = vec![0.0; arr_len];
    let mut roll_95 = vec![0.0; arr_len];
    let mut slope = vec![0.0; arr_len];
    let mut slope01 = vec![0.0; arr_len];
    let mut slope15 = vec![0.0; arr_len];
    let mut slope05 = vec![0.0; arr_len];
    for i in 0..arr_len {
        centroid[i] = analysis[i].spectral_centroid;
        variance[i] = analysis[i].spectral_variance;
        skewness[i] = analysis[i].spectral_skewness;
        kurtosis[i] = analysis[i].spectral_kurtosis;
        entropy[i] = analysis[i].spectral_entropy;
        flatness[i] = analysis[i].spectral_flatness;
        roll_50[i] = analysis[i].spectral_roll_off_50;
        roll_75[i] = analysis[i].spectral_roll_off_75;
        roll_90[i] = analysis[i].spectral_roll_off_90;
        roll_95[i] = analysis[i].spectral_roll_off_95;
        slope[i] = analysis[i].spectral_slope;
        slope01[i] = analysis[i].spectral_slope_0_1_khz;
        slope15[i] = analysis[i].spectral_slope_1_5_khz;
        slope05[i] = analysis[i].spectral_slope_0_5_khz;
    }
    analysis_dict.set_item(String::from("spectral_centroid"), centroid.into_pyarray(py).to_owned()).unwrap();
    analysis_dict.set_item(String::from("spectral_variance"), variance.into_pyarray(py).to_owned()).unwrap();
    analysis_dict.set_item(String::from("spectral_skewness"), skewness.into_pyarray(py).to_owned()).unwrap();
    analysis_dict.set_item(String::from("spectral_kurtosis"), kurtosis.into_pyarray(py).to_owned()).unwrap();
    analysis_dict.set_item(String::from("spectral_entropy"), entropy.into_pyarray(py).to_owned()).unwrap();
    analysis_dict.set_item(String::from("spectral_flatness"), flatness.into_pyarray(py).to_owned()).unwrap();
    analysis_dict.set_item(String::from("spectral_roll_off_50"), roll_50.into_pyarray(py).to_owned()).unwrap();
    analysis_dict.set_item(String::from("spectral_roll_off_75"), roll_75.into_pyarray(py).to_owned()).unwrap();
    analysis_dict.set_item(String::from("spectral_roll_off_90"), roll_90.into_pyarray(py).to_owned()).unwrap();
    analysis_dict.set_item(String::from("spectral_roll_off_95"), roll_95.into_pyarray(py).to_owned()).unwrap();
    analysis_dict.set_item(String::from("spectral_slope"), slope.into_pyarray(py).to_owned()).unwrap();
    analysis_dict.set_item(String::from("spectral_slope_0_1_khz"), slope01.into_pyarray(py).to_owned()).unwrap();
    analysis_dict.set_item(String::from("spectral_slope_1_5_khz"), slope15.into_pyarray(py).to_owned()).unwrap();
    analysis_dict.set_item(String::from("spectral_slope_0_5_khz"), slope05.into_pyarray(py).to_owned()).unwrap();
    analysis_dict
}

/// A Python module implemented in Rust.
#[pymodule]
fn aus_analyzer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(analyze, m)?)?;
    Ok(())
}
