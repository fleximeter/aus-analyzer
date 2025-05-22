//! # analyze
//! The `analyze` module contains analysis functionality.

use aus::{analysis, mp, spectrum};
use aus::analysis::{Analysis, analyzer};
use threadpool::ThreadPool;
use std::sync::mpsc;
use num::Complex;

/// Represents a STFT analysis
pub struct StftAnalysis {
    pub magnitude_spectrogram: Vec<Vec<f64>>,
    pub mel_spectrogram: Vec<Vec<f64>>,
    pub mfccs: Vec<Vec<f64>>,
    pub phase_spectrogram: Vec<Vec<f64>>,
    pub analysis: Vec<Analysis>
}

/// Performs a STFT analysis on an audio file. This is a modified version
/// of the analyzer included in aus::mp. It is modified to also produce
/// the STFT data.
pub fn analyze_audio_file(audio: &mut Vec<f64>, fft_size: usize, sample_rate: u32, num_mels: usize, num_mfccs: usize, max_num_threads: Option<usize>) -> StftAnalysis {
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

    let stft_imaginary_spectrogram: Vec<Vec<Complex<f64>>> = mp::rstft(audio, fft_size, fft_size / 2, aus::WindowType::Hamming, max_num_threads);
    let (stft_magnitude_spectrogram, stft_phase_spectrogram) = spectrum::complex_to_polar_rstft(&stft_imaginary_spectrogram);
    let stft_power_spectrogram = analysis::make_power_spectrogram(&stft_magnitude_spectrogram);
    let rfft_freqs = spectrum::rfftfreq(fft_size, sample_rate);
    let mel_filterbank = analysis::mel::MelFilterbank::new(0.0, sample_rate as f64 / 2.0, num_mels, &rfft_freqs, true);
    let mel_spectrogram = analysis::mel::make_mel_spectrogram(&stft_power_spectrogram, &mel_filterbank);
    let mfccs = analysis::mel::mfcc_spectrogram(&mel_spectrogram, 0.0);
    
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
        // let local_sample_rate = sample_rate;
        
        // Copy the fragment of the magnitude spectrum for this thread
        let mut local_magnitude_spectrum: Vec<Vec<f64>> = Vec::with_capacity(num_frames_per_thread);
        let mut local_mel_spectrum: Vec<Vec<f64>> = Vec::with_capacity(num_frames_per_thread);
        let start_idx = i * num_frames_per_thread;
        let end_idx = usize::min(start_idx + num_frames_per_thread, stft_magnitude_spectrogram.len());
        for j in start_idx..end_idx {
            let mut rfft_frame: Vec<f64> = Vec::with_capacity(stft_magnitude_spectrogram[j].len());
            let mut mel_frame: Vec<f64> = Vec::with_capacity(stft_magnitude_spectrogram[j].len());
            for k in 0..stft_magnitude_spectrogram[j].len() {
                rfft_frame.push(stft_magnitude_spectrogram[j][k]);
            }
            for k in 0..mel_spectrogram[j].len() {
                mel_frame.push(mel_spectrogram[j][k]);
            }
            local_magnitude_spectrum.push(rfft_frame);
            local_mel_spectrum.push(mel_frame);
        }

        let prev_spectrum = if start_idx > 0 {
            let mut spec: Vec<f64> = vec![0.0; stft_magnitude_spectrogram[start_idx-1].len()];
            for j in 0..stft_magnitude_spectrogram[start_idx-1].len() {
                spec[j] = stft_magnitude_spectrogram[start_idx-1][j];
            }
            Some(spec)
        } else {
            None
        };

        // Start the thread
        pool.execute(move || {
            let mut analyses: Vec<Analysis> = Vec::new();
            
            // Perform the analyses
            for j in 0..local_magnitude_spectrum.len() {
                match &prev_spectrum {
                    Some(spec) => {
                        analyses.push(analyzer(&local_magnitude_spectrum[j], Some(&spec), sample_rate, &local_rfft_freqs));
                    }, None => {
                        analyses.push(analyzer(&local_magnitude_spectrum[j], None, sample_rate, &local_rfft_freqs));
                    }
                }
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
    let mut analyses: Vec<Analysis> = Vec::new();
    for i in 0..results.len() {
        for j in 0..results[i].1.len() {
            analyses.push(results[i].1[j]);
        }
    }

    // Rotate the spectral vectors to comply with typical SciPy layout
    let mut magnitude_spectrogram1: Vec<Vec<f64>> = Vec::new();
    let mut phase_spectrogram1: Vec<Vec<f64>> = Vec::new();
    let mut mel_spectrogram1: Vec<Vec<f64>> = Vec::new();
    let mut mfccs1: Vec<Vec<f64>> = Vec::new();

    // Rotate the STFT data
    for j in 0..stft_magnitude_spectrogram[0].len() {
        let mut rotated_mag_spectral_frame: Vec<f64> = Vec::with_capacity(stft_magnitude_spectrogram.len());
        let mut rotated_phase_spectral_frame: Vec<f64> = Vec::with_capacity(stft_phase_spectrogram.len());
        for i in 0..stft_magnitude_spectrogram.len() {
            rotated_mag_spectral_frame.push(stft_magnitude_spectrogram[i][j]);
            rotated_phase_spectral_frame.push(stft_phase_spectrogram[i][j]);

        }
        magnitude_spectrogram1.push(rotated_mag_spectral_frame);
        phase_spectrogram1.push(rotated_phase_spectral_frame);
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
        magnitude_spectrogram: magnitude_spectrogram1,
        phase_spectrogram: phase_spectrogram1,
        mel_spectrogram: mel_spectrogram1,
        mfccs: mfccs1,
        analysis: analyses
    }
}
