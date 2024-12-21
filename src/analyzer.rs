//! # analyze
//! The `analyze` module contains analysis functionality.

use aus::analysis::{Analysis, analyzer};
use aus::spectrum;
use std::thread;
use threadpool::ThreadPool;
use std::sync::mpsc;
use num::Complex;

/// Represents a STFT analysis
pub struct StftAnalysis {
    pub magnitude_spectrogram: Vec<Vec<f64>>,
    pub phase_spectrogram: Vec<Vec<f64>>,
    pub analysis: Vec<Analysis>
}

/// Performs a STFT analysis on an audio file. This is a modified version
/// of the analyzer included in aus::mp. It is modified to also produce
/// the STFT data.
pub fn analyze_audio_file(audio: &mut Vec<f64>, fft_size: usize, sample_rate: u32, max_num_threads: Option<usize>) -> StftAnalysis {
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

    let stft_imaginary_spectrum: Vec<Vec<Complex<f64>>> = spectrum::rstft(audio, fft_size, fft_size / 2, aus::WindowType::Hamming);
    let (stft_magnitude_spectrum, stft_phase_spectrum) = spectrum::complex_to_polar_rstft(&stft_imaginary_spectrum);
    
    // Set up the multithreading
    let (tx, rx) = mpsc::channel();  // the message passing channel
    let num_threads: usize = match thread::available_parallelism() {
        Ok(x) => x.get(),
        Err(_) => 1
    };

    // Get the starting STFT frame index for each thread
    let mut thread_start_indices: Vec<usize> = vec![0; num_threads];
    let num_frames_per_thread: usize = f64::ceil(stft_magnitude_spectrum.len() as f64 / num_threads as f64) as usize;
    for i in 0..num_threads {
        thread_start_indices[i] = num_frames_per_thread * i;
    }

    // Run the threads
    let pool = ThreadPool::new(pool_size);
    for i in 0..num_threads {
        let tx_clone = tx.clone();
        let thread_idx = i;
        
        // Copy the fragment of the magnitude spectrum for this thread
        let mut local_magnitude_spectrum: Vec<Vec<f64>> = Vec::with_capacity(num_frames_per_thread);
        let start_idx = i * num_frames_per_thread;
        let end_idx = usize::min(start_idx + num_frames_per_thread, stft_magnitude_spectrum.len());
        for j in start_idx..end_idx {
            let mut rfft_frame: Vec<f64> = Vec::with_capacity(stft_magnitude_spectrum[j].len());
            for k in 0..stft_magnitude_spectrum[j].len() {
                rfft_frame.push(stft_magnitude_spectrum[j][k]);
            }
            local_magnitude_spectrum.push(rfft_frame);
        }

        // Copy other important variables
        let local_fft_size = fft_size;
        let local_sample_rate = sample_rate;

        // Start the thread
        pool.execute(move || {
            let mut analyses: Vec<Analysis> = Vec::with_capacity(local_magnitude_spectrum.len());
            
            // Perform the analyses
            for j in 0..local_magnitude_spectrum.len() {
                analyses.push(analyzer(&local_magnitude_spectrum[j], local_fft_size, local_sample_rate))
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

    // Return the final analysis package
    StftAnalysis {
        magnitude_spectrogram: stft_magnitude_spectrum,
        phase_spectrogram: stft_phase_spectrum,
        analysis: analyses
    }
}
