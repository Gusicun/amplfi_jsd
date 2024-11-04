import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import welch
from data import SignalDataSet
from ml4gw.waveforms import IMRPhenomD
from mlpe.injection.priors import nonspin_bbh_chirp_mass_q_parameter_sampler


def plot_time_domain(sig_data_original, sig_data_reduced, sample_rate_original, sample_rate_reduced):
    """Plot waveforms in the time domain and save the plot."""

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)

    # Load data directly from the datasets
    original_waveforms = next(iter(sig_data_original.train_dataloader()))[0]
    reduced_waveforms = next(iter(sig_data_reduced.train_dataloader()))[0]

    # Move tensors to CPU if needed
    original_waveforms = original_waveforms.cpu().numpy()
    reduced_waveforms = reduced_waveforms.cpu().numpy()

    # Time vector for the original data
    time_original = np.arange(original_waveforms.shape[-1]) / sample_rate_original
    # Time vector for the reduced data
    time_reduced = np.arange(reduced_waveforms.shape[-1]) / sample_rate_reduced

    # Select the first interferometer's waveform data (e.g., H1)
    original_waveform_h1 = original_waveforms[0][0, :]  # shape should be (8192,)
    reduced_waveform_h1 = reduced_waveforms[0][0, :]    # shape should be (4096,) for reduced data

    # Plot original waveform
    plt.plot(time_original, original_waveform_h1, label='Original Signal H1')
    # Plot reduced waveform
    plt.plot(time_reduced, reduced_waveform_h1, label='Reduced Signal H1', linestyle='--')

    plt.title('Time Domain Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    # Save plot as an image
    plt.savefig("time_domain_comparison.png")
    plt.close()


def plot_frequency_domain(sig_data_original, sig_data_reduced, sample_rate_original, sample_rate_reduced):
    """Plot waveforms in the frequency domain using FFT and save the plot."""

    original_waveforms = next(iter(sig_data_original.train_dataloader()))[0]
    reduced_waveforms = next(iter(sig_data_reduced.train_dataloader()))[0]

    # Move tensors to CPU if needed
    original_waveforms = original_waveforms.cpu().numpy()
    reduced_waveforms = reduced_waveforms.cpu().numpy()

    fft_original = fft(original_waveforms[0][0, :])
    fft_reduced = fft(reduced_waveforms[0][0, :])

    freq_original = fftfreq(len(fft_original), 1 / sample_rate_original)
    freq_reduced = fftfreq(len(fft_reduced), 1 / sample_rate_reduced)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 2)
    plt.plot(freq_original, np.abs(fft_original), label='Original FFT')
    plt.plot(freq_reduced, np.abs(fft_reduced), label='Reduced FFT', linestyle='--')

    plt.title('Frequency Domain Comparison')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()

    # Save plot as an image
    plt.savefig("frequency_domain_comparison.png")
    plt.close()


def plot_power_spectral_density(sig_data_original, sig_data_reduced, sample_rate_original, sample_rate_reduced):
    """Plot power spectral density (PSD) comparison and save the plot."""

    original_waveforms = next(iter(sig_data_original.train_dataloader()))[0]
    reduced_waveforms = next(iter(sig_data_reduced.train_dataloader()))[0]

    # Move tensors to CPU if needed
    original_waveforms = original_waveforms.cpu().numpy()
    reduced_waveforms = reduced_waveforms.cpu().numpy()

    f_original, Pxx_original = welch(original_waveforms[0][0, :], sample_rate_original, nperseg=1024)
    f_reduced, Pxx_reduced = welch(reduced_waveforms[0][0, :], sample_rate_reduced, nperseg=512)

    plt.figure(figsize=(7, 5))
    plt.semilogy(f_original, Pxx_original, label='Original PSD')
    plt.semilogy(f_reduced, Pxx_reduced, label='Reduced PSD', linestyle='--')
    plt.title('Power Spectral Density Comparison')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.legend()

    # Save plot as an image
    plt.savefig("power_spectral_density_comparison.png")
    plt.close()


def main():
    background_path = os.getenv('DATA_DIR') + "/background.h5"
    batch_size = 500
    batches_per_epoch = 100
    sample_rate = 2048
    time_duration = 4
    f_min = 20
    f_max = 200
    f_ref = 40
    prior_func = nonspin_bbh_chirp_mass_q_parameter_sampler

    # Generate original data
    sig_data_original = SignalDataSet(
        background_path,
        ifos=["H1", "L1"],
        valid_frac=0.2,
        batch_size=batch_size,
        batches_per_epoch=batches_per_epoch,
        sampling_frequency=sample_rate,
        time_duration=time_duration,
        f_min=f_min,
        f_max=f_max,
        f_ref=f_ref,
        prior_func=prior_func,
        approximant=IMRPhenomD,
    )
    sig_data_original.setup(None)

    # Generate reduced data (halved time_duration and sampling_rate)
    sig_data_reduced = SignalDataSet(
        background_path,
        ifos=["H1", "L1"],
        valid_frac=0.2,
        batch_size=batch_size,
        batches_per_epoch=batches_per_epoch,
        sampling_frequency=sample_rate // 2,
        time_duration=time_duration / 2,
        f_min=f_min,
        f_max=f_max,
        f_ref=f_ref,
        prior_func=prior_func,
        approximant=IMRPhenomD,
    )
    sig_data_reduced.setup(None)

    # Visualization and save plots
    plot_time_domain(sig_data_original, sig_data_reduced, sample_rate, sample_rate // 2)
    plot_frequency_domain(sig_data_original, sig_data_reduced, sample_rate, sample_rate // 2)
    plot_power_spectral_density(sig_data_original, sig_data_reduced, sample_rate, sample_rate // 2)


if __name__ == "__main__":
    main()
