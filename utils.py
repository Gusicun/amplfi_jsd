from pathlib import Path
from typing import Callable, List, Optional, Tuple

import bilby
import h5py
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mlpe.architectures.flows import MaskedAutoRegressiveFlow

from ml4gw.transforms import ChannelWiseScaler
from mlpe.data.transforms import Preprocessor
from mlpe.injection.utils import phi_from_ra
import torch
import h5py
from mlpe.injection.priors import nonspin_bbh_component_mass, nonspin_bbh_chirp_mass_q, nonspin_bbh_component_mass_parameter_sampler, nonspin_bbh_chirp_mass_q_parameter_sampler
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# TODO: add this function to preprocessor module class
'''
def load_preprocessor_state(
    preprocessor_dir: Path,
    param_dim: int,
    n_ifos: int,
    fduration: float,
    sample_rate: float,
    device: str,
):
    standard_scaler = ChannelWiseScaler(param_dim)
    preprocessor = Preprocessor(
        n_ifos,
        sample_rate,
        fduration,
        scaler=standard_scaler,
    )
    whitener_path = preprocessor_dir / "whitener.pt"
    scaler_path = preprocessor_dir / "scaler.pt"

    preprocessor.whitener.load_state_dict(torch.load(whitener_path))
    preprocessor.scaler.load_state_dict(torch.load(scaler_path))

    preprocessor = preprocessor.to(device)
    return preprocessor
'''
def load_preprocessor_state(
    preprocessor_dir: Path,
    param_dim: int,
    n_ifos: int,
    time_duration: float,
    sample_rate: float,
    device: str,
):
    from pathlib import Path
    import torch
    from ml4gw.transforms import ChannelWiseScaler
    from mlpe.data.transforms import Preprocessor

    #param_dim = 8
    standard_scaler = ChannelWiseScaler(param_dim)

    kernel_size = time_duration + 1

    preprocessor = Preprocessor(
        n_ifos,
        kernel_size,
        sample_rate,
        scaler=standard_scaler,
    )


    whitener_path = preprocessor_dir / "whitener.pt"
    scaler_path = preprocessor_dir / "scaler.pt"
    
    whitener_state_dict = torch.load(whitener_path, map_location=device)
    current_whitener_state_dict = preprocessor.whitener.state_dict()

    filtered_whitener_state_dict = {
        k: v for k, v in whitener_state_dict.items()
        if k in current_whitener_state_dict and v.size() == current_whitener_state_dict[k].size()
    }

    preprocessor.whitener.load_state_dict(filtered_whitener_state_dict, strict=False)

    preprocessor.scaler.load_state_dict(torch.load(scaler_path, map_location=device))

    preprocessor = preprocessor.to(device)

    preprocessor = preprocessor.to(device)
    return preprocessor

def initialize_data_loader_from_gw150914(signal_tensor: torch.Tensor, device: str):
    param_tensor = torch.zeros(signal_tensor.size(0), len(inference_params), dtype=torch.float32) 
    dataset = torch.utils.data.TensorDataset(signal_tensor.to(device), param_tensor.to(device))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, pin_memory=(device != "cpu"))

    return dataloader, param_tensor, None 


def initialize_data_loader_from_gw150914(
    inference_params: List[str],
    device: str,
    sample_rate: int = 2048,
):
    event = 'GW150914'
    gps_time = event_gps(event)
    
    start_time = gps_time - 4
    end_time = gps_time + 4
    
    h1_data = TimeSeries.fetch_open_data('H1', start_time, end_time, sample_rate=sample_rate, cache=True)
    l1_data = TimeSeries.fetch_open_data('L1', start_time, end_time, sample_rate=sample_rate, cache=True)
    
    h1_whitened = h1_data.whiten()
    l1_whitened = l1_data.whiten()

    h1_tensor = torch.tensor(h1_whitened.value, dtype=torch.float32).unsqueeze(0)  # shape (1, T)
    l1_tensor = torch.tensor(l1_whitened.value, dtype=torch.float32).unsqueeze(0)  # shape (1, T)
    
    signal_tensor = torch.cat([h1_tensor, l1_tensor], dim=0)  # shape: (2, T)

    param_values = {
        "chirp_mass": [30.0],  
        "mass_ratio": [0.8],   
        "luminosity_distance": [400.0],  
        "phase": [0.0],
        "theta_jn": [0.4],
        "dec": [-1.0],
        "psi": [0.2],
        "phi": [1.5],
    }
    
    params = []
    for param in inference_params:
        if param == "hrss":
            params.append(np.log10(param_values.get(param, [0.0])))
        else:
            params.append(param_values.get(param, [0.0]))
    params = np.vstack(params).T  # (N_samples, N_params)
    
    params_tensor = torch.from_numpy(params).to(torch.float32).to(device)

    dataset = TensorDataset(signal_tensor, params_tensor)
    dataloader = DataLoader(dataset, batch_size=1, pin_memory=(device != "cpu"))

    return dataloader, params_tensor, [gps_time]


def initialize_data_loader(
    testing_path: Path,
    inference_params: List[str],
    device: str,
):
    with h5py.File(testing_path, "r") as f:
        try:
            times = f["geocent_time"][:]
        except KeyError:
            times = None
        signals = f["injections"][:]
        params = []
        for param in inference_params:
            values = f[param][:]
            # take logarithm since hrss
            # spans large magnitude range
            if param == "hrss":
                values = np.log10(values)
            params.append(values)

        params = np.vstack(params).T
    injections = torch.from_numpy(signals).to(torch.float32)
    params = torch.from_numpy(params).to(torch.float32)
    dataset = torch.utils.data.TensorDataset(injections, params)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        pin_memory=False if device == "cpu" else True,
        batch_size=1,
        pin_memory_device=device,
    )

    return dataloader, params, times


def cast_samples_as_bilby_result(
    samples: np.ndarray,
    truth: np.ndarray,
    inference_params: List[str],
    priors: "bilby.core.prior.PriorDict",
    label: str,
):
    """Cast posterior samples as bilby Result object"""
    # samples shape (1, num_samples, num_params)
    # inference_params shape (1, num_params)
    for param in inference_params:
        if param not in priors:
            print(f"Parameter '{param}' missing in priors, adding default prior.")
            if param == "chirp_mass":
                priors[param] = bilby.core.prior.Uniform(name=param, minimum=10, maximum=100, latex_label="$\\mathcal{M}$")
            elif param == "mass_ratio":
                priors[param] = bilby.core.prior.Uniform(name=param, minimum=0.1, maximum=1, latex_label="$q$")
            elif param == "luminosity_distance":
                priors[param] = bilby.core.prior.Uniform(name=param, minimum=100, maximum=1000, latex_label="$D_L$")
            else:
                priors[param] = bilby.core.prior.Uniform(name=param, minimum=0, maximum=1, latex_label=param)


    injections = {k: float(v) for k, v in zip(inference_params, truth)}

    posterior = dict()
    for idx, k in enumerate(inference_params):
        posterior[k] = samples.T[idx].flatten()
    posterior = pd.DataFrame(posterior)

    return bilby.result.Result(
        label=label,
        injection_parameters=injections,
        posterior=posterior,
        search_parameter_keys=inference_params,
        priors=priors,
    )


def generate_corner_plots(
    results: List[bilby.core.result.Result], writedir: Path
):
    for i, result in enumerate(results):
        filename = writedir / f"corner_{i}.png"
        result.plot_corner(
            parameters=result.injection_parameters,
            save=True,
            filename=filename,
            levels=(0.5, 0.9),
        )


def generate_overlapping_corner_plots(
    results: Tuple[bilby.core.result.Result], outfile: Path
):
    for i, result in enumerate(results):
        bilby.result.plot_multiple(
            result,
            parameters=result[0].injection_parameters,
            save=True,
            filename=outfile,
            levels=(0.5, 0.9),
        )


def plot_mollview(
    ra_samples: np.ndarray,
    dec_samples: np.ndarray,
    nside: int = 32,
    truth: Optional[Tuple[float, float]] = None,
    outpath: Path = None,
):
    """Plot mollview of posterior samples

    Args:
        ra_samples: array of right ascension samples in radians (-pi, pi)
        dec_samples: array of declination samples in radians (-pi/2, pi/2)
        nside: nside parameter for healpy
        truth: tuple of true ra and dec
    """

    # mask out non physical samples;
    ra_samples_mask = (ra_samples > -np.pi) * (ra_samples < np.pi)
    dec_samples += np.pi / 2
    dec_samples_mask = (dec_samples > 0) * (dec_samples < np.pi)

    net_mask = ra_samples_mask * dec_samples_mask
    ra_samples = ra_samples[net_mask]
    dec_samples = dec_samples[net_mask]

    # calculate number of samples in each pixel
    NPIX = hp.nside2npix(nside)
    ipix = hp.ang2pix(nside, dec_samples, ra_samples)
    ipix = np.sort(ipix)
    uniq, counts = np.unique(ipix, return_counts=True)

    # create empty map and then fill in non-zero pix with counts
    m = np.zeros(NPIX)
    m[np.in1d(range(NPIX), uniq)] = counts

    plt.close()
    # plot molleweide
    fig = hp.mollview(m)
    if truth is not None:
        ra_inj, dec_inj = truth
        dec_inj += np.pi / 2
        hp.visufunc.projscatter(
            dec_inj, ra_inj, marker="x", color="red", s=150
        )

    plt.savefig(outpath)

    return fig


def load_and_initialize_flow(
    flow: Callable,
    embedding: Callable,
    model_state_path: Path,
    n_ifos: int,
    strain_dim: int,
    param_dim: int,
    device: str,
    opt: Callable,  
    sched: Callable,  
    inference_params: List[str], 
    num_transforms: int,
    num_blocks: int, 
    hidden_features: int,
):
    
    #ckpt_path = os.getenv("BASE_DIR") + "/pl-logdir/phenomd-60-transforms-4-4-resnet-wider-dl/version_815/checkpoints/epoch=232-step=46600.ckpt"
    # checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    # flow_obj.load_state_dict(checkpoint['state_dict'])
    model_state = torch.load(model_state_path, map_location=device)

    #input_tensor = torch.randn(1, n_ifos, strain_dim).to(device)  
    
    embedding = embedding.to(device)  
    #embedding_output = embedding(input_tensor)  
    

    flow_obj = MaskedAutoRegressiveFlow(
        (param_dim, n_ifos, strain_dim), 
        embedding,
        opt, 
        sched,  
        inference_params,
        num_transforms=num_transforms,
        num_blocks=num_blocks,
        hidden_features=hidden_features
    )
    
    flow_obj.load_state_dict(model_state['state_dict'], strict=False)
    #flow_obj.build_flow(model_state=model_state)
    #flow_obj.build_flow()
    flow_obj = flow_obj.to(device) 
    flow_obj.to(device)
    embedding.to(device)

    return flow_obj


def draw_samples_from_model(
    dataloader,
    params,  
    flow: torch.nn.Module,
    preprocessor: torch.nn.Module,
    inference_params: List[str],
    num_samples_draw: int,
    priors: dict,
    label: str = "testing_samples",
):
    results = []
    device = next(flow.parameters()).device
    preprocessor.to(device)

    joint_signal_batches = []  # To combine signal data from both interferometers

    for batch in dataloader:
        signal_batch = batch[0]  # Assuming shape is [2, 8192], two interferometers
        signal_batch = signal_batch.to(device)

        # Add signal to the joint list
        joint_signal_batches.append(signal_batch)

    # Stack the signal batches for joint processing, shape becomes [2, 8192] if two interferometers
    joint_signal_batch = torch.stack(joint_signal_batches, dim=1)  # Shape becomes [1, 2, 8192]

    param_batch = params.to(device)

    print(f"Type of joint_signal_batch: {type(joint_signal_batch)}")
    print(f"Shape of joint_signal_batch: {joint_signal_batch.shape}")
    print(f"Type of param_batch: {type(param_batch)}")

    # Process the joint signal and parameters through the preprocessor
    strain, scaled_param = preprocessor(joint_signal_batch, param_batch)

    with torch.no_grad():
        # Sample from the flow model based on the joint signal
        samples = flow.sample([1, num_samples_draw], context=strain)  # signal_batch.shape[0]

        # Descale the samples
        descaled_samples = preprocessor.scaler(
            samples[0].transpose(1, 0).to(device), reverse=True
        )

    # Ensure correct dimensions for the output
    descaled_samples = descaled_samples.unsqueeze(0).transpose(2, 1)

    # Cast the results into a format suitable for bilby
    descaled_res = cast_samples_as_bilby_result(
        descaled_samples.cpu().numpy()[0],  
        scaled_param.cpu().numpy()[0],
        inference_params,
        priors,
        label=label,
    )

    results.append(descaled_res)  # Append the combined result

    return results




'''
def draw_samples_from_model(
    signal,
    param,
    flow: torch.nn.Module,
    preprocessor: torch.nn.Module,
    inference_params: List[str],
    num_samples_draw: int,
    priors: dict,
    label: str = "testing_samples",
):
    print(f"Type of signal: {type(signal)}")
    print(f"Type of param: {type(param)}")
    strain, scaled_param = preprocessor(signal, param)
    with torch.no_grad():
        samples = flow.sample([1, num_samples_draw], context=strain)
        descaled_samples = preprocessor.scaler(
            samples[0].transpose(1, 0), reverse=True
        )
    descaled_samples = descaled_samples.unsqueeze(0).transpose(2, 1)
    descaled_res = cast_samples_as_bilby_result(
        descaled_samples.cpu().numpy()[0],
        param.cpu().numpy()[0],
        inference_params,
        priors,
        label=label,
    )
    return descaled_res
'''

def load_and_sort_bilby_results_from_dynesty(
    bilby_result_dir: Path,
    inference_params: List[str],
    parameters: np.ndarray,
    times: np.ndarray,
):
    bilby_results = []
    paths = sorted(list(bilby_result_dir.iterdir()))
    for idx, (path, param, time) in enumerate(zip(paths, parameters, times)):
        print(idx, time, path)
        bilby_result = bilby.core.result.read_in_result(path)
        bilby_result.injection_parameters = {
            k: float(v) for k, v in zip(inference_params, param)
        }
        bilby_result.injection_parameters["geocent_time"] = time
        bilby_result.label = f"bilby_{idx}"
        bilby_results.append(bilby_result)
    return bilby_results


def add_phi_to_bilby_results(results: List[bilby.core.result.Result]):
    """Attach phi w.r.t. GMST to the bilby results"""
    results_with_phi = []
    for res in results:
        res.posterior["phi"] = phi_from_ra(
            res.posterior["ra"], res.injection_parameters["geocent_time"]
        )
        results_with_phi.append(res)
    return results_with_phi