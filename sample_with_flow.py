import argparse
import logging
import pickle
import random
import os
from pathlib import Path
from time import time
from typing import Callable, List
import torch
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
from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries
import bilby
import numpy as np
from bilby.core.prior import Uniform
from utils import (
    draw_samples_from_model,
    initialize_data_loader,
    load_and_initialize_flow,
    load_preprocessor_state,
    plot_mollview,
)

from mlpe.architectures import embeddings, flows
from mlpe.injection.priors import sg_uniform
from mlpe.logging import configure_logging
import pandas as pd
import random
from pathlib import Path
import logging

from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import List
import h5py

from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps, find_datasets
from gwosc.datasets import event_gps, event_segment
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import List
import bilby
from gwpy.segments import Segment


def bandpass_filter(strain, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, strain)

def whiten(strain, psd, dt):
    """Whiten data by dividing by the sqrt of the power spectral density (PSD)."""
    freqs = np.fft.rfftfreq(len(strain), dt)
    strain_fft = np.fft.rfft(strain)
    try:
        psd_freqs = psd.frequency_array  
        psd_values = psd.psd_array       
    except AttributeError:
        raise AttributeError("PSD lack of bilby PowerSpectralDensity")

    psd_interp = np.interp(freqs, psd_freqs, psd_values)

    whitened_fft = strain_fft / np.sqrt(psd_interp) / (np.sqrt(2.0 * dt))
    return np.fft.irfft(whitened_fft, n=len(strain))


def initialize_data_loader_from_gw150914(
    inference_params: List[str],
    device: str,
    sample_rate: float = None,  
):
    event = 'GW150914'
    trigger_time = event_gps(event)
    segment = event_segment(event)
    start_time, end_time = segment
    print(f"Available data segment for {event}: {start_time} - {end_time}")
 
    duration = 6   
    start_time = trigger_time - duration / 2
    end_time = trigger_time + duration / 2
  
    if start_time < segment[0]:
        start_time = segment[0]
    if end_time > segment[1]:
        end_time = segment[1]
 

    ifo_list = bilby.gw.detector.InterferometerList([])
    for det in ['H1', 'L1']:
        print(f"Downloading analysis data for {det}")
        data = TimeSeries.fetch_open_data(
            det,
            start_time,
            end_time,
            #sample_rate=sample_rate,
            cache=True,
            #version='GWTC-1-confident' 
        )
        data = data.resample(sample_rate)
        ifo = bilby.gw.detector.get_empty_interferometer(det)
        ifo.strain_data.set_from_gwpy_timeseries(data)
        print(f"Downloading PSD data for {det}")
        psd_duration = 32 
        psd_start_time = start_time - psd_duration
        psd_end_time = start_time
        psd_data = TimeSeries.fetch_open_data(
            det,
            psd_start_time,
            psd_end_time,
            #sample_rate=sample_rate,
            cache=True,
            #version='GWTC-1-confident' 
        )
        psd_data = psd_data.resample(sample_rate)
        psd_alpha = 0.25  
        psd = psd_data.psd(
            fftlength=duration,
            overlap=0,
            window=('tukey', psd_alpha),
            method='median'
        )
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=psd.frequencies.value,
            psd_array=psd.value
        )
        ifo_list.append(ifo)
    
    strain_data = []
    #original code
    for ifo in ifo_list:
        #strain_data.append(ifo.strain_data.time_domain_strain.value)
        strain_data.append(ifo.strain_data.time_domain_strain)
    '''
    for ifo in ifo_list:
        filtered_data = bandpass_filter(ifo.strain_data.time_domain_strain, 20, 400, sample_rate)
        whitened_data = whiten(filtered_data, ifo.power_spectral_density, 1 / sample_rate)
        strain_data.append(whitened_data)
    '''

    signals = np.stack(strain_data)  
    injections = torch.from_numpy(signals).to(torch.float32)


    try:
        samples = np.genfromtxt("/home/fan.zhang/bilby/samples_bilbys.csv", delimiter=',', names=True)
        params_list = []
        for param in inference_params:
            if param in samples.dtype.names:
                values = samples[param]
                param_value = np.mean(values)
            else:
                param_value = 0.0
                print(f"Parameter '{param}' not found in samples, using 0.0 as placeholder.")
            params_list.append(param_value)
        params_array = np.array(params_list)
    except FileNotFoundError:

        print("Bilby result file 'samples_bilby.csv' not found. Using placeholder values for parameters.")
        params_array = np.zeros(len(inference_params))

    params = torch.from_numpy(params_array).to(torch.float32).unsqueeze(0) 

    times = np.array([trigger_time])  
    print("Injections shape:", injections.shape)
    #print("Params shape:", params.shape)

    dataset = TensorDataset(injections)#, params)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        pin_memory=False if device == "cpu" else True,
        pin_memory_device=device,
    )

    return dataloader, params, times

def main(
    flow: Callable,
    embedding: Callable,
    model_state_path: Path,
    num_samples_draw: int = 1000,
    num_plot_corner: int = 5,
    verbose: bool = False,
    ):

    basedir = os.getenv("BASE_DIR") + "/pl-logdir/phenomd-60-transforms-4-4-resnet-wider-dl"
    ifos = ["H1", "L1"]
    sample_rate = 2048
    kernel_length = 4  # in seconds
    fduration = 0.5  # assumed based on physics, can be adjusted
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
    device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'
    '''
    event = 'GW150914'
    hdf5_file = f"{event}.h5"
    
    download_and_save_hdf5(event, hdf5_file)
    '''
    logdir = Path(basedir) / "log"
    logdir.mkdir(parents=True, exist_ok=True)
    configure_logging(logdir / "sample_with_flow.log", verbose)

    priors = sg_uniform()
    priors["phi"] = Uniform(
        name="phi", minimum=-np.pi, maximum=np.pi, latex_label="phi"
    )  # FIXME: remove when prior is moved to using torch tools
    
    n_ifos = len(ifos)
    param_dim = len(inference_params)
    strain_dim = int(sample_rate * kernel_length)
    optimizer = torch.optim.AdamW
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau

    logging.info("Initializing model and setting weights from trained state")
    num_transforms = 60
    num_blocks = 5
    hidden_features = 120
    prior_func = nonspin_bbh_chirp_mass_q_parameter_sampler

    
    flow = load_and_initialize_flow(
        flow,
        embedding,
        model_state_path,
        n_ifos,
        strain_dim,
        param_dim,
        device,
        optimizer,  
        scheduler,  
        inference_params,
        num_transforms,
        num_blocks,
        hidden_features
    )
    flow = flow.to(device) 
    embedding = embedding.to(device)
    flow.eval()  # set flow to eval mode

    input_tensor = torch.randn(n_ifos, strain_dim).unsqueeze(0).to(device)
    embedded_output = embedding(input_tensor)  

    logging.info(f"Embedding output shape: {embedded_output.shape}")

    logging.info("Initializing preprocessor and setting weights from trained state")
    preprocessor_dir = Path("/home/fan.zhang/PE-dev/projects/sandbox_cbc/train/data")#Path(basedir) / "training" / "preprocessor"
    '''
    preprocessor = load_preprocessor_state(
        preprocessor_dir, param_dim, n_ifos, fduration, sample_rate, device
    )
    '''
    time_duration = 5.0  
    preprocessor = load_preprocessor_state(
        preprocessor_dir,
        param_dim,
        n_ifos,
        time_duration,
        sample_rate,
        device,
    )
    preprocessor = preprocessor.to(device)
    logging.info("Loading GW150914 data and initializing dataloader")
    test_dataloader, params, gps_time = initialize_data_loader_from_gw150914(inference_params, device, sample_rate)
    #test_dataloader, params, gps_time = initialize_data_loader_from_gw150914(hdf5_file, inference_params, device)

    print(f"Type of preprocessor: {type(preprocessor)}")
    print(f"the params by bilby:(params)")

    logging.info(f"Drawing {num_samples_draw} samples for each test data")
    print(type(preprocessor))
    total_sampling_time = time()
    results = draw_samples_from_model(
        test_dataloader,
        params,
        flow,
        preprocessor,
        inference_params,
        num_samples_draw,
        priors,
        label="test_samples_using_flow",
    )
    print(results)
    total_sampling_time = time() - total_sampling_time

    all_posteriors = []  
    num_plotted = 0  
    num_plot_corner = 10 
    for idx, res in enumerate(results):
        if random.random() > 0.5 and num_plotted < num_plot_corner:
            corner_plot_filename = Path(basedir) / f"{num_plotted}_descaled_corner.png"
            skymap_filename = Path(basedir) / f"{num_plotted}_mollview.png"
            
            res.plot_corner(
                save=True,
                filename=corner_plot_filename,
                levels=(0.5, 0.9),
            )
            plot_mollview(
                res.posterior["phi"],
                res.posterior["dec"],
                truth=(
                    res.injection_parameters["phi"],
                    res.injection_parameters["dec"],
                ),
                outpath=skymap_filename,
            )
            num_plotted += 1

        posterior_df = res.posterior.copy()
        posterior_df["result_index"] = idx  
        all_posteriors.append(posterior_df)
    
    all_posteriors_df = pd.concat(all_posteriors, ignore_index=True)
    
    posterior_csv_path = Path(basedir) / "amplfi_samples_5000.csv"
    all_posteriors_df.to_csv(posterior_csv_path, index=False)
    logging.info(f"All posterior samples saved to {posterior_csv_path}")
  
    logging.info("Making pp-plot")
    pp_plot_dir = Path(basedir) / "pp_plots"
    pp_plot_dir.mkdir(parents=True, exist_ok=True)
    pp_plot_filename = pp_plot_dir / "pp-plot-test-set-3000.png"
    
    bilby.result.make_pp_plot(
        results,
        save=True,
        filename=pp_plot_filename,
        keys=inference_params,
    )
    logging.info(f"PP Plots saved in {pp_plot_dir}")
    '''
    for res in results:
        if random.random() > 0.5 and num_plotted < num_plot_corner:
            corner_plot_filename = Path(basedir) / f"{num_plotted}_descaled_corner.png"
            skymap_filename = Path(basedir) / f"{num_plotted}_mollview.png"
            res.plot_corner(
                save=True,
                filename=corner_plot_filename,
                levels=(0.5, 0.9),
            )
            plot_mollview(
                res.posterior["phi"],
                res.posterior["dec"],
                truth=(
                    res.injection_parameters["phi"],
                    res.injection_parameters["dec"],
                ),
                outpath=skymap_filename,
            )
            num_plotted += 1
    logging.info(f"Total sampling time: {total_sampling_time:.1f} seconds")

    logging.info("Making pp-plot")
    pp_plot_dir = Path(basedir) / "pp_plots"
    pp_plot_filename = pp_plot_dir / "pp-plot-test-set-1000.png"
    bilby.result.make_pp_plot(
        results,
        save=True,
        filename=pp_plot_filename,
        keys=inference_params,
    )
    logging.info(f"PP Plots saved in {pp_plot_dir}")

    logging.info("Saving samples obtained from flow")
    with open(Path(basedir) / "flow-samples-as-bilby-result_1000.pickle", "wb") as f:
        pickle.dump(results, f)
    '''




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run flow-based inference sampling")
    parser.add_argument('--model_state_path', type=str, required=True, help="Path to the trained model state (.pth file)")
    parser.add_argument('--num_samples_draw', type=int, default=1000, help="Number of samples to draw for each test data")
    parser.add_argument('--num_plot_corner', type=int, default=5, help="Number of corner plots to generate")
    parser.add_argument('--verbose', action='store_true', help="Increase output verbosity")

    args = parser.parse_args()

    background_path = os.getenv('DATA_DIR') + "/background.h5"
    ifos = ["H1", "L1"]
    batch_size = 500
    batches_per_epoch = 100#org=200
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
    
    param_dim, n_ifos, strain_dim = (
        len(inference_params),
        len(ifos),
        int(sample_rate * time_duration),
    )

    #input_tensor = torch.randn(n_ifos, strain_dim)
    embedding = ResNet(
        (n_ifos, strain_dim),
        context_dim=resnet_context_dim,
        layers=resnet_layers,
        norm_groups=resnet_norm_groups,
        #kernel_size=5 
    )
    
    main(
        flow=MaskedAutoRegressiveFlow,
        embedding=embedding,
        model_state_path=Path(args.model_state_path),
        num_samples_draw=args.num_samples_draw,
        num_plot_corner=args.num_plot_corner,
        verbose=args.verbose
    )
