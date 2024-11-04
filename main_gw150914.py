import os
import time
import torch
from data import SignalDataSet,SignalGWDataSet
from lightning.pytorch import Trainer, callbacks, loggers
from lightning.pytorch.cli import LightningCLI
import numpy as np
import random
from ml4gw.waveforms import IMRPhenomD, TaylorF2
from mlpe.architectures.embeddings import ResNet, QuantizedResNet
from mlpe.architectures.embeddings import DenseNet1D
from ml4gw.nn.resnet import ResNet1D
from mlpe.architectures.flows import MaskedAutoRegressiveFlow
from mlpe.injection.priors import nonspin_bbh_component_mass, nonspin_bbh_chirp_mass_q, nonspin_bbh_component_mass_parameter_sampler, nonspin_bbh_chirp_mass_q_parameter_sampler
from mlpe.logging import configure_logging
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, decimate
from scipy.signal import welch
from sampler import ParameterTransformer, ParameterSampler, ZippedDataset
from torch.distributions import Uniform
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps, event_segment
from scipy.signal import butter, filtfilt
import numpy as np
from bilby.gw.detector import InterferometerList
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
from bilby.gw.detector import Interferometer, PowerSpectralDensity
#from bilby.gw.detector import get_interferometer_with_open_data
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

torch.cuda.empty_cache()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_gw150914_data(sample_rate: int, duration: float):
    
    event_time = 1126259462.4  
    end_time = event_time + duration
    h1_data = TimeSeries.fetch_open_data('H1', event_time, end_time, cache=True)
    l1_data = TimeSeries.fetch_open_data('L1', event_time, end_time, cache=True)

    h1_data = h1_data.resample(sample_rate)
    l1_data = l1_data.resample(sample_rate)

    strain_data = np.stack([h1_data.value, l1_data.value])  
    return strain_data
'''


def bandpass_filter(strain_data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, strain_data)

def whiten(strain, psd, dt):
    """Whiten data by dividing by the sqrt of the power spectral density (PSD)"""
    freqs = np.fft.rfftfreq(len(strain), dt)
    strain_fft = np.fft.rfft(strain)
    psd_interp = np.interp(freqs, psd.frequencies.value, psd.value)
    whitened_fft = strain_fft / np.sqrt(psd_interp) / (np.sqrt(2.0 * dt))
    return np.fft.irfft(whitened_fft, n=len(strain))

def load_gw150914_data(sample_rate: int, duration: float, lowcut=20, highcut=512):
    """
    Load and preprocess GW150914 strain data for H1 and L1 detectors.
    """
    # Set up parameters for GW150914
    trigger_time = 1126259462.4
    detectors = ["H1", "L1"]
    roll_off = 0.4  # Tukey window roll-off duration in seconds

    # Define time segments for data and PSD estimation
    end_time = trigger_time + 2  # 2 seconds post-trigger
    start_time = end_time - duration
    psd_duration = 32 * duration
    psd_start_time = start_time - psd_duration
    psd_end_time = start_time

    strain_data = []

    for det in detectors:
        # Download analysis data for each interferometer
        data = TimeSeries.fetch_open_data(det, start_time, end_time, cache=True).resample(sample_rate)
        psd_data = TimeSeries.fetch_open_data(det, psd_start_time, psd_end_time, cache=True)

        # Bandpass filter the data
        filtered_data = bandpass_filter(data.value, lowcut, highcut, sample_rate)

        # Compute PSD and whiten the data
        psd = psd_data.psd(fftlength=duration, window=("tukey", 2 * roll_off / duration))
        whitened_data = whiten(filtered_data, psd, 1 / sample_rate)

        strain_data.append(whitened_data)

    # Stack H1 and L1 whitened strain data for output
    return np.stack(strain_data)
'''


def main():
    set_seed(0)
    #background_path = "/home/fan.zhang/PE-dev/segmented_data/0.0_3600.0_background/background.h5"
    background_path = os.getenv('DATA_DIR') + "/background.h5"
    ifos = ["H1", "L1"]
    batch_size = 50#500
    batches_per_epoch = 10#org=200
    sample_rate = 2048
    time_duration = 4
    f_max = 200
    f_min = 20
    f_ref = 40
    highpass = 25
    valid_frac = 0.2
    learning_rate = 1e-3

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
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
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
    
    # Continue with the rest of your setup and training logic...
    #prior_func = nonspin_bbh_component_mass_parameter_sampler
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

    #ckpt_path = os.getenv("BASE_DIR") + "/pl-logdir/phenomd-60-transforms-4-4-resnet-wider-dl/version_815/checkpoints/epoch=232-step=46600.ckpt"
    ckpt_path = "/home/fan.zhang/base/pl-logdir/phenomd-60-transforms-4-4-resnet-wider-dl/version_49/checkpoints/epoch=273-step=27400.ckpt"
    checkpoint = torch.load(ckpt_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    flow_obj.load_state_dict(checkpoint["state_dict"])
    # data
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
        prior_func=prior_func,
        approximant=IMRPhenomD,
    )
    
    print("##### Initialized data loader, calling setup ####")
    sig_dat.setup(None)
    
    gw150914_data = load_gw150914_data(sample_rate, time_duration)

    sig_gw = SignalGWDataSet(
        strain_data=gw150914_data,  
        ifos=ifos,
        batch_size=50,
        batches_per_epoch=10,
        sampling_frequency=sample_rate,
        time_duration=time_duration,
    )
    print("##### Initialized GW150914 data loader, calling setup ####")
    sig_gw.setup(None)

    
    print("##### Dataloader initialized #####")
    torch.set_float32_matmul_precision("high")
    early_stop_cb = callbacks.EarlyStopping(
        "train_loss", patience=50, check_finite=True, verbose=True
    )
    lr_monitor = callbacks.LearningRateMonitor(logging_interval="epoch")
    outdir = os.getenv("BASE_DIR")
    logger = loggers.CSVLogger(save_dir=outdir + "/pl-logdir", name="phenomd-60-transforms-4-4-resnet-wider-dl")
    #logger = loggers.CSVLogger(save_dir=outdir + "/pl-logdir", name="phenomd-50-transforms-2-2-resnet")
    print("##### Initializing trainer #####")
    trainer = Trainer(
        #precision=16,
        max_epochs=1000,
        log_every_n_steps=100,
        callbacks=[early_stop_cb, lr_monitor],
        logger=logger,
        gradient_clip_val=10.0,
    )
    '''
    start_time = time.time()
    trainer.fit(model=flow_obj, datamodule=sig_dat)
    end_time = time.time()
    print(f"Training time: {end_time - start_time} seconds")
    torch.save(flow_obj.state_dict(), 'gw150914_resnet.pth')
    '''
    # For testing/inference
   
    start_time = time.time()
    '''
    results = []
    for batch in sig_gw.test_dataloader():
        with torch.no_grad():
            estimated_params = flow_obj(batch[0])  
            results.append(estimated_params.cpu().numpy())

    df_results = pd.DataFrame(np.vstack(results), columns=["chirp_mass", "mass_ratio", "luminosity_distance", "phase", "theta_jn", "dec", "psi", "phi"])
    df_results.to_csv("gw150914_parameter_estimates.csv", index=False)
    print("Parameter estimates saved to gw150914_parameter_estimates.csv")
    '''
    trainer.test(model=flow_obj, datamodule=sig_gw, ckpt_path=None)
    end_time = time.time()
    print(f"Inference time: {end_time - start_time} seconds")
    
    flow_obj.save_sampling_results(output_dir="./amplfi_sampling_results")
  
    #trainer.fit(model=flow_obj, datamodule=sig_dat)
    # trainer.test(model=flow_obj, datamodule=sig_dat, ckpt_path=ckpt_path)

if __name__ == '__main__':
    main()
