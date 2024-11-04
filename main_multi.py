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
import scipy.signal

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
    
def create_dataset_from_band(signal, fs, lowcut, highcut, target_fs, background_path, ifos, valid_frac, batch_size, batches_per_epoch, f_min, f_max, f_ref, start_time, end_time):
    
    filtered_signal = apply_bandpass_filter(signal, lowcut, highcut, fs)
  
    #start_time, end_time = calculate_time_for_frequency_band(filtered_signal, fs, lowcut, highcut)
    print(f"time: {start_time} to {end_time}")
    
    downsampled_signal = time_based_downsampling(filtered_signal, fs, target_fs, start_time, end_time)
    print(f"Downsampled signal length: {len(downsampled_signal)}")
 
    dataset = SignalDataSet(
        background_path,
        ifos,
        valid_frac,
        batch_size,
        batches_per_epoch,
        target_fs,  
        len(downsampled_signal) / target_fs, 
        f_min,
        f_max,
        f_ref,
        prior_func=nonspin_bbh_chirp_mass_q_parameter_sampler,
        approximant=IMRPhenomD,
    )
    
    return dataset



def calculate_time_for_frequency_band(signal, fs, f_min, f_max, params=None):
   
    f, t, Zxx = scipy.signal.stft(signal, fs, nperseg=1024)

    band_indices = (f >= f_min) & (f <= f_max)

    band_power = np.mean(np.abs(Zxx[band_indices, :])**2, axis=0)
    power_threshold = 0.1 * np.max(band_power)

    significant_indices = np.where(band_power > power_threshold)[0]

    if len(significant_indices) > 0:
        start_time = t[significant_indices[0]]
        end_time = t[significant_indices[-1]]
    else:
        start_time = 0
        end_time = t[-1]
    
    return start_time, end_time

def time_based_downsampling(signal, fs, target_fs, start_time, end_time):
    start_sample = int(start_time * fs)
    end_sample = int(end_time * fs)
    signal_segment = signal[start_sample:end_sample]
    decimation_factor = int(fs / target_fs)
    downsampled_signal = decimate(signal_segment, decimation_factor)
    return downsampled_signal

def multiband_time_based_downsampling(signal, fs, bands, target_sample_rates, chirp_duration=4.0):
    downsampled_signals = []
    for (lowcut, highcut), target_fs in zip(bands, target_sample_rates):
        filtered_signal = apply_bandpass_filter(signal, lowcut, highcut, fs)
        start_time, end_time = calculate_time_for_frequency_band(filtered_signal, fs, lowcut, highcut)
        print(f"time for freq is {start_time} to {end_time}")
        downsampled_signal = time_based_downsampling(filtered_signal, fs, target_fs, start_time, end_time)
        downsampled_signals.append(downsampled_signal)
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
    time_duration = 3
    f_max = 200
    f_min = 20
    f_ref = 40
    highpass = 25
    valid_frac = 0.2
    learning_rate = 1e-3

    # Define multibanding parameters with different target sample rates
    bands = [(20, 100), (100, 200)]  # Define frequency bands
    shd=100
    target_sample_rates = [1024, 2048]  # Lower sample rate for lower frequency band
    #target_sample_rates_2 = 2048

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
        (param_dim, n_ifos, strain_dim),
        embedding,
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
    print(f"Original signal length: {len(original_waveform)}")

    # Apply multibanding downsampling
    lowcut1, highcut1 = bands[0]
    start_time1, end_time1 = calculate_time_for_frequency_band(original_waveform, sample_rate, lowcut1, highcut1)
    print(f"first ({lowcut1}-{highcut1} Hz) is : {start_time1} to {end_time1}")
    
    duration1 = end_time1 - start_time1
    if duration1 > time_duration:
        duration1 = time_duration
        end_time1 = start_time1 + duration1

    start_time2 = end_time1  
    end_time2 = time_duration 
    lowcut2, highcut2 = bands[1]

    downsampled_datasets = []

    dataset1 = create_dataset_from_band(
        original_waveform, sample_rate, lowcut1, highcut1, target_sample_rates[0],
        background_path, ifos, valid_frac, batch_size, batches_per_epoch,
        f_min, f_max, f_ref, start_time1, end_time1
    )
    dataset1.setup(None)
    dataset2 = create_dataset_from_band(
        original_waveform, sample_rate, lowcut2, highcut2, target_sample_rates[1],
        background_path, ifos, valid_frac, batch_size, batches_per_epoch,
        f_min, f_max, f_ref, start_time1, end_time2
    )
    dataset2.setup(None)
    
    #for dataset in downsampled_datasets:
    #dataset1.setup(None)
    
    print("##### Dataloader initialized #####")
    
    downsampled_datasets = [dataset1, dataset2]
    # Iterate over downsampled datasets and create a new Trainer instance for each
    '''
    for i, dataset in enumerate(downsampled_datasets):
        print(f"##### Training with multiband downsampled dataset {i + 1} #####")
        
        # Re-initialize Trainer for each dataset
        early_stop_cb = callbacks.EarlyStopping(
            "valid_loss", patience=50, check_finite=True, verbose=True
        )
        lr_monitor = callbacks.LearningRateMonitor(logging_interval="epoch")
        outdir = os.getenv("BASE_DIR")
        logger = loggers.CSVLogger(save_dir=outdir + "/pl-logdir", name=f"phenomd-60-transforms-4-4-resnet-wider-dl-dataset-{i+1}")
        
        trainer = Trainer(
            max_epochs=10,
            log_every_n_steps=100,
            callbacks=[early_stop_cb, lr_monitor],
            logger=logger,
            gradient_clip_val=10.0,
        )
    
        # Load flow_obj state if this is not the first dataset
        if i > 0:
            load_flow_obj(flow_obj, i-1)
    
        start_time = time.time()
        trainer.fit(model=flow_obj, datamodule=dataset)
        end_time = time.time()
        print(f"Training time for dataset {i + 1}: {end_time - start_time} seconds")
        
        # Save the entire flow_obj state after each dataset
        save_flow_obj(flow_obj, i)


    
    torch.save(flow_obj.state_dict(), 'org_resnet.pth')
    
    # For testing/inference, you can use one of the datasets or a merged dataset
    start_time = time.time()
    trainer.test(model=flow_obj, datamodule=sig_dat)  # Replace with the dataset you want to test
    end_time = time.time()
    print(f"Inference time: {end_time - start_time} seconds")
    '''
     
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
