//! # analyze
//! The `analyze` module contains analysis functionality.

use aus::{analysis, mp, spectrum};
use threadpool::ThreadPool;
use std::sync::mpsc;
use num::Complex;

/// Represents a STFT analysis
pub struct StftAnalysis {
    pub power_spectrogram: Vec<Vec<f64>>,
    pub mel_spectrogram: Vec<Vec<f64>>,
    pub mfccs: Vec<Vec<f64>>,
    pub analyses: Vec<FrameAnalysis2>
}

#[derive(Copy, Clone)]
pub struct FrameAnalysis2 {
    pub alpha_ratio: f64,
    pub hammarberg_index: f64,
    pub harmonicity: f64,
    pub spectral_centroid: f64,
    pub spectral_difference: f64,
    pub spectral_entropy: f64,
    pub spectral_flatness: f64,
    pub spectral_flux: f64,
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
    pub spectral_variance: f64
}

fn analyze_aux(magnitude_spectrum: &[f64], magnitude_spectrum_prev: Option<&[f64]>, rfft_freqs: &[f64], sample_rate: u32, mel_filterbank: &analysis::mel::MelFilterbank, num_mfccs: usize) -> (FrameAnalysis2, Vec<f64>, Vec<f64>, Vec<f64>) {
    let power_spectrum = analysis::make_power_spectrum(&magnitude_spectrum);
    let magnitude_spectrum_sum = magnitude_spectrum.iter().sum();
    let power_spectrum_sum = power_spectrum.iter().sum();
    let spectrum_pmf = analysis::make_spectrum_pmf(&power_spectrum, power_spectrum_sum);
    let analysis_spectral_centroid = analysis::computation::compute_spectral_centroid(&magnitude_spectrum, rfft_freqs, magnitude_spectrum_sum);
    let analysis_spectral_variance = analysis::computation::compute_spectral_variance(&spectrum_pmf, rfft_freqs, analysis_spectral_centroid);
    let analysis_spectral_skewness = analysis::computation::compute_spectral_skewness(&spectrum_pmf, rfft_freqs, analysis_spectral_centroid, analysis_spectral_variance);
    let analysis_spectral_kurtosis = analysis::computation::compute_spectral_kurtosis(&spectrum_pmf, rfft_freqs, analysis_spectral_centroid, analysis_spectral_variance);
    let analysis_spectral_entropy = analysis::computation::compute_spectral_entropy(&spectrum_pmf);
    let analysis_spectral_flatness = analysis::computation::compute_spectral_flatness(&magnitude_spectrum, magnitude_spectrum_sum);
    let analysis_spectral_roll_off_50 = analysis::computation::compute_spectral_roll_off_point(&power_spectrum, rfft_freqs, power_spectrum_sum, 0.5);
    let analysis_spectral_roll_off_75 = analysis::computation::compute_spectral_roll_off_point(&power_spectrum, rfft_freqs, power_spectrum_sum, 0.75);
    let analysis_spectral_roll_off_90 = analysis::computation::compute_spectral_roll_off_point(&power_spectrum, rfft_freqs, power_spectrum_sum, 0.9);
    let analysis_spectral_roll_off_95 = analysis::computation::compute_spectral_roll_off_point(&power_spectrum, rfft_freqs, power_spectrum_sum, 0.95);
    let analysis_spectral_slope = analysis::computation::compute_spectral_slope(&power_spectrum, power_spectrum_sum);
    let spec_difference = match magnitude_spectrum_prev {
        Some(spec) => {
        match analysis::spectral_difference(magnitude_spectrum, spec, true) {
            Ok(diff) => diff,
            Err(_) => f64::NAN
        }},
        None => f64::NAN
    };
    let spec_flux = match magnitude_spectrum_prev {
        Some(spec) => {
        match analysis::spectral_flux(magnitude_spectrum, spec, Some(aus::util::Norm::L2)) {
            Ok(diff) => diff,
            Err(_) => f64::NAN
        }},
        None => f64::NAN
    };

    // Eyben notes an author that recommends computing the slope of these spectral bands separately.
    let analysis_spectral_slope_0_1_khz = analysis::computation::compute_spectral_slope_region(&power_spectrum, &rfft_freqs, 0.0, 1000.0, sample_rate);
    let analysis_spectral_slope_1_5_khz = analysis::computation::compute_spectral_slope_region(&power_spectrum, &rfft_freqs, 1000.0, 5000.0, sample_rate);
    let analysis_spectral_slope_0_5_khz = analysis::computation::compute_spectral_slope_region(&power_spectrum, &rfft_freqs, 0.0, 5000.0, sample_rate);

    let mel_spectrum = mel_filterbank.filter(&power_spectrum);
    let mfccs = analysis::mel::mfcc_spectrum(&mel_spectrum, num_mfccs, None);
    
    (FrameAnalysis2 {
        alpha_ratio: analysis::alpha_ratio(magnitude_spectrum, rfft_freqs),
        hammarberg_index: analysis::hammarberg_index(magnitude_spectrum, rfft_freqs),
        harmonicity: analysis::harmonicity(magnitude_spectrum, true),
        spectral_centroid: analysis_spectral_centroid,
        spectral_difference: spec_difference,
        spectral_entropy: analysis_spectral_entropy,
        spectral_flatness: analysis_spectral_flatness,
        spectral_flux: spec_flux,
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
    }, power_spectrum, mel_spectrum, mfccs)

}

/// Performs a STFT analysis on an audio file. This is a modified version
/// of the analyzer included in aus::mp. It is modified to also produce
/// the STFT data.
pub fn analyze(audio: &mut Vec<f64>, fft_size: usize, hop_size: usize, sample_rate: u32, num_mels: usize, num_mfccs: usize, max_num_threads: Option<usize>) -> StftAnalysis {
    let max_available_threads = match std::thread::available_parallelism() {
        Ok(x) => x.get(),
        Err(_) => 1
    };
    let pool_size = match max_num_threads {
        Some(x) => {
            if x > max_available_threads || x == 0 {
                max_available_threads
            } else {
                x
            }
        },
        None => max_available_threads
    };

    let stft_imaginary_spectrogram: Vec<Vec<Complex<f64>>> = mp::rstft(audio, fft_size, hop_size, aus::WindowType::Hamming, max_num_threads);
    let rfft_freqs = spectrum::rfftfreq(fft_size, sample_rate);
    let (stft_magnitude_spectrogram, _) = spectrum::complex_to_polar_rstft(&stft_imaginary_spectrogram);
    let mel_filterbank = analysis::mel::MelFilterbank::new(0.0, sample_rate as f64 / 2.0, num_mels, &rfft_freqs, true);
    
    // Set up the multithreading
    let (tx, rx) = mpsc::channel();  // the message passing channel

    // Get the starting STFT frame index for each thread
    let mut thread_start_indices: Vec<usize> = vec![0; pool_size];
    let num_frames_per_thread: usize = f64::ceil(stft_magnitude_spectrogram.len() as f64 / pool_size as f64) as usize;
    for i in 0..pool_size {
        thread_start_indices[i] = num_frames_per_thread * i;
    }

    // Run the threads
    let pool = ThreadPool::new(pool_size);
    for i in 0..pool_size {
        let tx_clone = tx.clone();
        let thread_idx = i;
        let local_rfft_freqs = rfft_freqs.clone();
        
        // Copy the fragment of the magnitude spectrum for this thread
        let mut local_magnitude_spectra: Vec<Vec<f64>> = Vec::with_capacity(num_frames_per_thread);
        let start_idx = i * num_frames_per_thread;
        let end_idx = usize::min(start_idx + num_frames_per_thread, stft_magnitude_spectrogram.len());
        
        for j in start_idx..end_idx {
            let mut rfft_frame: Vec<f64> = Vec::with_capacity(stft_magnitude_spectrogram[j].len());
            for k in 0..stft_magnitude_spectrogram[j].len() {
                rfft_frame.push(stft_magnitude_spectrogram[j][k]);
            }
            local_magnitude_spectra.push(rfft_frame);
        }

        // we need to provide the previous frame for spectral difference and spectral flux computation
        let prev_spectrum = if start_idx > 0 {
            let mut spec: Vec<f64> = Vec::with_capacity(stft_magnitude_spectrogram[start_idx-1].len());
            for j in 0..stft_magnitude_spectrogram[start_idx-1].len() {
                spec.push(stft_magnitude_spectrogram[start_idx-1][j]);
            }
            Some(spec)
        } else {
            None
        };

        // copy the Mel filterbank
        let local_mel_filterbank = mel_filterbank.clone();

        // Start the thread
        pool.execute(move || {
            let mut analyses: Vec<(FrameAnalysis2, Vec<f64>, Vec<f64>, Vec<f64>)> = Vec::with_capacity(local_magnitude_spectra.len());
            
            // Analyze the first frame
            match prev_spectrum {
                Some(spec) => {
                    analyses.push(analyze_aux(&local_magnitude_spectra[0], Some(&spec), &local_rfft_freqs, sample_rate, &local_mel_filterbank, num_mfccs));
                },
                None => {
                    analyses.push(analyze_aux(&local_magnitude_spectra[0], None, &local_rfft_freqs, sample_rate, &local_mel_filterbank, num_mfccs));
                }
            }

            // Analyze the remaining frames
            for j in 1..local_magnitude_spectra.len() {
                analyses.push(analyze_aux(&local_magnitude_spectra[j], Some(&local_magnitude_spectra[j-1]), &local_rfft_freqs, sample_rate, &local_mel_filterbank, num_mfccs));
            }

            let _ = match tx_clone.send((thread_idx, analyses)) {
                Ok(x) => x,
                Err(_) => ()
            };
        });
    }

    // Drop the original sender. Once all senders are dropped, receiving will end automatically.
    drop(tx);

    // Collect the analysis vectors and sort them by thread id
    let mut results = vec![];
    for received_data in rx {
        results.push(received_data);
    }
    results.sort_by_key(|&(index, _)| index);
    
    // let all threads wrap up
    pool.join();

    // Combine the analysis vectors into one big vector
    let mut analyses: Vec<FrameAnalysis2> = Vec::new();
    let mut power_spectrogram: Vec<Vec<f64>> = Vec::new();
    let mut mel_spectrogram: Vec<Vec<f64>> = Vec::new();
    let mut mfccs: Vec<Vec<f64>> = Vec::new();
    for i in 0..results.len() {
        for j in 0..results[i].1.len() {
            analyses.push(results[i].1[j].0);
            power_spectrogram.push(results[i].1[j].1.clone());
            mel_spectrogram.push(results[i].1[j].2.clone());
            mfccs.push(results[i].1[j].3.clone());
        }
    }

    // Rotate the spectral vectors to comply with typical SciPy layout
    let mut power_spectrogram1: Vec<Vec<f64>> = Vec::new();
    let mut mel_spectrogram1: Vec<Vec<f64>> = Vec::new();
    let mut mfccs1: Vec<Vec<f64>> = Vec::new();

    // Rotate the power spectrogram
    for j in 0..stft_magnitude_spectrogram[0].len() {
        let mut rotated_power_spectral_frame: Vec<f64> = Vec::with_capacity(power_spectrogram.len());
        for i in 0..power_spectrogram.len() {
            rotated_power_spectral_frame.push(power_spectrogram[i][j]);

        }
        power_spectrogram1.push(rotated_power_spectral_frame);
    }

    // Rotate the Mel spectrum
    for j in 0..mel_spectrogram[0].len() {
        let mut rotated_spectral_frame: Vec<f64> = Vec::with_capacity(mel_spectrogram.len());
        for i in 0..mel_spectrogram.len() {
            rotated_spectral_frame.push(mel_spectrogram[i][j]);
        }
        mel_spectrogram1.push(rotated_spectral_frame);
    }

    // Rotate the MFCCs and only include the number asked for
    let mfcc_len = if num_mfccs < mfccs[0].len() {num_mfccs} else {mfccs[0].len()};
    for j in 0..mfcc_len {
        let mut rotated_spectral_frame: Vec<f64> = Vec::with_capacity(mfccs.len());
        for i in 0..mfccs.len() {
            rotated_spectral_frame.push(mfccs[i][j]);
        }
        mfccs1.push(rotated_spectral_frame);
    }

    // Return the final analysis package
    StftAnalysis {
        power_spectrogram: power_spectrogram1,
        mel_spectrogram: mel_spectrogram1,
        mfccs: mfccs1,
        analyses
    }
}
