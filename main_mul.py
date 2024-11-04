import os
import time
import torch
from data import SignalDataSet
from lightning.pytorch import Trainer, callbacks, loggers
import numpy as np
import random
from ml4gw.waveforms import IMRPhenomD, TaylorF2
from mlpe.architectures.embeddings import ResNet
from mlpe.architectures.flows import MaskedAutoRegressiveFlow
from mlpe.injection.priors import nonspin_bbh_chirp_mass_q_parameter_sampler
from scipy.signal import butter, filtfilt, decimate
from sampler import ParameterTransformer, ParameterSampler
from torch.distributions import Uniform

torch.cuda.empty_cache()

def apply_bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """
    Apply a bandpass filter to the signal.
    
    Parameters:
    - signal: The input signal.
    - lowcut: Low cutoff frequency.
    - highcut: High cutoff frequency.
    - fs: Sampling frequency of the signal.
    - order: Order of the filter.
    
    Returns:
    - Filtered signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def time_based_partial_downsampling(signal, fs, target_fs_1, target_fs_2, downsample_duration, original_duration):
    """
    Perform partial downsampling on the input signal where the first `downsample_duration` seconds are downsampled with
    `target_fs_1`, and the remaining `original_duration - downsample_duration` seconds are downsampled with `target_fs_2`.
    
    Parameters:
    - signal: The input signal.
    - fs: Original sampling frequency of the signal.
    - target_fs_1: The desired downsampled frequency for the first segment.
    - target_fs_2: The desired downsampled frequency for the second segment.
    - downsample_duration: Duration (seconds) of the first segment.
    - original_duration: Total duration of the signal.
    
    Returns:
    - downsampled_signals: Two downsampled signals.
    """
    samples_downsample_1 = int(fs * downsample_duration)  # Samples in the first part to be downsampled
    samples_original = int(fs * (original_duration - downsample_duration))  # Samples in the second part
    
    # Split the signal into two parts
    signal_part_1 = signal[:samples_downsample_1]
    signal_part_2 = signal[samples_downsample_1:samples_downsample_1 + samples_original]
    
    # Apply downsampling to the first and second parts
    decimation_factor_1 = int(fs / target_fs_1)
    decimation_factor_2 = int(fs / target_fs_2)
    
    downsampled_signal_1 = decimate(signal_part_1, decimation_factor_1)
    downsampled_signal_2 = decimate(signal_part_2, decimation_factor_2)
    
    return downsampled_signal_1, downsampled_signal_2

def multiband_downsampling(signal, fs, bands, target_sample_rates):
    """
    Perform multiband downsampling on the input signal.
    
    Parameters:
    - signal: The input signal.
    - fs: Original sampling frequency of the signal.
    - bands: List of (lowcut, highcut) tuples for each frequency band.
    - target_sample_rates: List of target sample rates for each frequency band.
    
    Returns:
    - downsampled_signals: List of downsampled signals for each frequency band.
    - downsampled_lengths: List of downsampled signal lengths for each band.
    """
    downsampled_signals = []
    downsampled_lengths = []
    
    for (lowcut, highcut), target_fs in zip(bands, target_sample_rates):
        # Apply bandpass filter for the current frequency band
        filtered_signal = apply_bandpass_filter(signal, lowcut, highcut, fs)
        
        # Calculate the decimation factor
        decimation_factor = int(fs / target_fs)
        
        # Downsample the filtered signal for lower frequencies
        if decimation_factor > 1:
            downsampled_signal = decimate(filtered_signal, decimation_factor)
        else:
            downsampled_signal = filtered_signal  # No downsampling for high frequency bands
            
        downsampled_signals.append(downsampled_signal)
        #downsampled_lengths.append(len(downsampled_signal))  # Store the length
    
    return downsampled_signals
    
def save_flow_obj(flow_obj, suffix):
    torch.save(flow_obj.state_dict(), f'flow_obj_state_{suffix}.pth')

def load_flow_obj(flow_obj, suffix):
    flow_obj.load_state_dict(torch.load(f'flow_obj_state_{suffix}.pth'))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(0)
    background_path = os.getenv('DATA_DIR') + "/background.h5"
    ifos = ["H1", "L1"]
    batch_size = 500
    batches_per_epoch = 100
    sample_rate = 2048
    target_sample_rate_1 = 1024  # Target downsampled frequency for the first segment
    target_sample_rate_2 = 2048  # Target downsampled frequency for the second segment
    time_duration = 4  # Total duration of the signal
    downsample_duration = 0.5  # Duration of the first segment to be downsampled
    f_max = 200
    f_min = 20
    f_ref = 40
    valid_frac = 0.2
    learning_rate = 1e-3

    densenet_context_dim = 100
    densenet_growth_rate = 32
    densenet_block_config = [6, 6]
    densenet_drop_rate = 0
    densenet_norm_groups = 8

    resnet_context_dim = 100
    resnet_layers = [4, 4]
    resnet_norm_groups = 8

    inference_params = [
        "chirp_mass",
        "mass_ratio",
        "luminosity_distance",
        "phase",
        "theta_jn",
        "dec",
        "psi",
        "phi",
    ]
    num_transforms = 60
    num_blocks = 5
    hidden_features = 120

    optimizer = torch.optim.AdamW
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    '''
    param_dim, n_ifos, strain_dim_1 = (
        len(inference_params),
        len(ifos),
        int(target_sample_rate_1 * downsample_duration),  # Adjust for downsampled signal in dataset 1
    )
    
    param_dim, n_ifos, strain_dim_2 = (
        len(inference_params),
        len(ifos),
        int(target_sample_rate_2 * (time_duration - downsample_duration)),  # Adjust for dataset 2
    )

    embedding_1 = ResNet(
        (n_ifos, strain_dim_1),
        context_dim=resnet_context_dim,
        layers=resnet_layers,
        norm_groups=resnet_norm_groups,
    )

    embedding_2 = ResNet(
        (n_ifos, strain_dim_2),
        context_dim=resnet_context_dim,
        layers=resnet_layers,
        norm_groups=resnet_norm_groups,
    )
    '''
    param_dim, n_ifos, strain_dim = (
        len(inference_params),
        len(ifos),
        int(sample_rate * time_duration),
    )

    embedding = ResNet(
        (n_ifos, strain_dim),
        context_dim=resnet_context_dim,
        layers=resnet_layers,
        norm_groups=resnet_norm_groups,
    )

    prior_func = nonspin_bbh_chirp_mass_q_parameter_sampler

    flow_obj = MaskedAutoRegressiveFlow(
        (param_dim, n_ifos, strain_dim),  # Ensure model can handle varying input sizes
        embedding,  # Use one embedding for both datasets
        optimizer,
        scheduler,
        inference_params,
        prior_func,
        num_transforms=num_transforms,
        num_blocks=num_blocks,
        hidden_features=hidden_features
    )

    # Initialize data loader and setup
    sig_dat = SignalDataSet(
        background_path,
        ifos,
        valid_frac,
        batch_size,
        batches_per_epoch,
        sample_rate,
        time_duration,
        f_min,
        f_max,
        f_ref,
        prior_func=nonspin_bbh_chirp_mass_q_parameter_sampler,
        approximant=IMRPhenomD,
    )
    
    print("##### Initialized data loader, calling setup ####")
    sig_dat.setup(None)
    
    # Get original waveforms from dataloader
    original_waveforms = next(iter(sig_dat.train_dataloader()))[0]
    original_waveform = original_waveforms[0][0].cpu().numpy()

    # Apply partial time-based downsampling to get two datasets
    downsampled_waveform_1, downsampled_waveform_2 = time_based_partial_downsampling(
        original_waveform,
        sample_rate,
        target_sample_rate_1,
        target_sample_rate_2,
        downsample_duration,
        time_duration
    )

    # Print signal lengths before and after downsampling
    print(f"Original signal length: {len(original_waveform)}")
    print(f"Downsampled signal length (part 1): {len(downsampled_waveform_1)}")
    print(f"Downsampled signal length (part 2): {len(downsampled_waveform_2)}")

    # Create two new SignalDataSet instances with the downsampled data
    downsampled_dataset_1 = SignalDataSet(
        background_path,
        ifos,
        valid_frac,
        batch_size,
        batches_per_epoch,
        target_sample_rate_1,  # Downsampled rate for first 3s
        downsample_duration,
        f_min,
        f_max,
        f_ref,
        prior_func=nonspin_bbh_chirp_mass_q_parameter_sampler,
        approximant=IMRPhenomD,
    )
    
    downsampled_dataset_2 = SignalDataSet(
        background_path,
        ifos,
        valid_frac,
        batch_size,
        batches_per_epoch,
        target_sample_rate_2,  # Original rate for last 1s
        time_duration - downsample_duration,
        f_min,
        f_max,
        f_ref,
        prior_func=nonspin_bbh_chirp_mass_q_parameter_sampler,
        approximant=IMRPhenomD,
    )

    print("##### Replaced datasets with downsampled data #####")
    downsampled_dataset_1.setup(None)
    downsampled_dataset_2.setup(None)
    downsampled_datasets = [downsampled_dataset_1, downsampled_dataset_2] 
    # TRAINING SETUP for both datasets
    torch.set_float32_matmul_precision("high")
    
    # Early stopping callback
    early_stop_cb = callbacks.EarlyStopping(
        "valid_loss", patience=50, check_finite=True, verbose=True
    )
    
    # Learning rate monitor callback
    lr_monitor = callbacks.LearningRateMonitor(logging_interval="epoch")
    
    # Logger setup
    outdir = os.getenv("BASE_DIR")
    logger = loggers.CSVLogger(save_dir=outdir + "/pl-logdir", name="phenomd-training-dataset1")
    
    for i, dataset in enumerate(downsampled_datasets):
        print(f"##### Training with multiband downsampled dataset {i + 1} #####")
        trainer = Trainer(
              max_epochs=10,
              log_every_n_steps=100,
              callbacks=[early_stop_cb, lr_monitor],
              logger=logger, 
              gradient_clip_val=10.0,
        )
        start_time = time.time()
        trainer.fit(model=flow_obj, datamodule=dataset)
        end_time = time.time()
        print(f"Inference time: {end_time - start_time} seconds")

    #save_flow_obj(flow_obj, f'epoch_{epoch+1}')

    # Save the trained models
    torch.save(flow_obj.state_dict(), 'multi_time_resnet.pth')
    
    # Use the last dataset (or any other dataset) for testing
    start_time = time.time()
    trainer.test(model=flow_obj, datamodule=sig_dat)  # Use the last dataset or the merged one
    end_time = time.time()
    print(f"Inference time: {end_time - start_time} seconds")


if __name__ == '__main__':
    main()
