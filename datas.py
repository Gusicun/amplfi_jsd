from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Optional, Tuple, TypeVar

import h5py
import torch
from gwpy.timeseries import TimeSeries
import lightning.pytorch as pl
import numpy as np
import torch
from waveforms import FrequencyDomainWaveformGenerator

from ml4gw import gw
from ml4gw.dataloading import InMemoryDataset
from ml4gw.transforms import ChannelWiseScaler
from ml4gw.waveforms import IMRPhenomD, TaylorF2
from mlpe.data.transforms import Preprocessor
from mlpe.injection.priors import nonspin_bbh_chirp_mass_q_parameter_sampler
from gwpy.timeseries import TimeSeries

Tensor = TypeVar("Tensor", np.ndarray, torch.Tensor)

def load_gw150914_data(sample_rate: int, duration: float, start_time: float):
    event_time = 1126259462.4  
    end_time = event_time + duration
    h1_data = TimeSeries.fetch_open_data('H1', start_time, end_time, cache=True)
    h1_data = h1_data.resample(sample_rate)
    l1_data = TimeSeries.fetch_open_data('L1', start_time, end_time, cache=True)
    l1_data = l1_data.resample(sample_rate)

    strain_data = np.stack([h1_data.value, l1_data.value]) 
    return strain_data



def split(X: Tensor, frac: float, axis: int) -> Tuple[Tensor, Tensor]:
    """
    Split an array into two parts along the given axis
    by an amount specified by `frac`. Generic to both
    numpy arrays and torch Tensors.
    """
    size = int(frac * X.shape[axis])
    if isinstance(X, np.ndarray):
        return np.split(X, [size], axis=axis)
    else:
        splits = [size, X.shape[axis] - size]
        return torch.split(X, splits, dim=axis)


class PEInMemoryDataset(InMemoryDataset):
    def __init__(
        self,
        X: np.ndarray,
        preprocessor: Optional[Callable] = None,
        batch_size: int = 32,
        batches_per_epoch: Optional[int] = None,
        coincident: bool = True,
        shuffle: bool = True,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            X,
            kernel_size=X.shape[-1],  
            batch_size=batch_size,
            batches_per_epoch=batches_per_epoch,
            stride=1,
            coincident=coincident,
            shuffle=shuffle,
            device=device,
        )
        self.preprocessor = preprocessor
        self.device = device

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        X = super().__next__()
        if self.preprocessor:
            transformed_X = self.preprocessor(X)
        return transformed_X.to(dtype=torch.float32)



class SignalDataSet(pl.LightningDataModule):
    def __init__(
        self,
        background_path: Path,
        ifos: Sequence[str],
        valid_frac: float,
        batch_size: int,
        batches_per_epoch: int,
        sampling_frequency: float,
        time_duration: float,
        f_min: float,
        f_max: float,
        f_ref: float,
        approximant=TaylorF2,
        prior_func: callable = nonspin_bbh_chirp_mass_q_parameter_sampler,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.background_path = background_path
        self.num_ifos = len(ifos)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prior_func = prior_func  # instantiate in setup

        self.tensors, self.vertices = gw.get_ifo_geometry(*ifos)

    def load_gw150914(self):
        start_time = 1126259450  
        duration = 4  
        sample_rate = self.hparams.sampling_frequency
        return torch.from_numpy(load_gw150914_data(sample_rate, duration, start_time))

    def setup(self, stage: str) -> None:
        self.gw150914_data = self.load_gw150914().to(dtype=torch.float64)
        self.standard_scaler = ChannelWiseScaler(8)
        self.prior = self.prior_func(self.device)
        
        if Path("scaler.pt").exists():
            self.standard_scaler.load_state_dict(torch.load("scaler.pt"))
        else:
            _samples = self.prior(10000)
            _samples = torch.vstack((
                _samples["chirp_mass"],
                _samples["mass_ratio"],
                _samples["luminosity_distance"],
                _samples["phase"],
                _samples["theta_jn"],
                _samples["dec"],
                _samples["psi"],
                _samples["phi"]))
            self.standard_scaler.fit(_samples)
            torch.save(self.standard_scaler.state_dict(), "scaler.pt") 
        self.standard_scaler.to(self.device)
        
        self.preprocessor = Preprocessor(
            self.num_ifos,
            self.hparams.time_duration + 1,
            self.hparams.sampling_frequency,
            scaler=self.standard_scaler,
        )
        if Path("whitener.pt").exists():
            self.preprocessor.whitener.load_state_dict(torch.load("whitener.pt"))
        else:
            self.preprocessor.whitener.fit(1, *background, fftlength=2)
            torch.save(self.preprocessor.whitener.state_dict(), "whitener.pt") 
    
        self.preprocessor.whitener.to(self.device)
        
        self.training_dataset = PEInMemoryDataset(
            self.gw150914_data,
            #prior=self.prior,  
            preprocessor=self.preprocessor,
            #kernel_size=self.gw150914_data.shape[-1],
            batch_size=self.hparams.batch_size,
            batches_per_epoch=self.hparams.batches_per_epoch,
            coincident=False,
            shuffle=True,
            device=self.device,
        )
        
    def train_dataloader(self):
        return self.training_dataset
    '''
    def val_dataloader(self):
        return self.validation_dataset

    def test_dataloader(self):
        return self.test_dataset
    '''

