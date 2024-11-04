import os
import time
import torch
from data import SignalDataSet
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

torch.cuda.empty_cache()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(0)
    #background_path = "/home/fan.zhang/PE-dev/segmented_data/0.0_3600.0_background/background.h5"
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

    ckpt_path = os.getenv("BASE_DIR") + "/pl-logdir/phenomd-60-transforms-4-4-resnet-wider-dl/version_815/checkpoints/epoch=232-step=46600.ckpt"
    # checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    # flow_obj.load_state_dict(checkpoint['state_dict'])

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
    
    print("##### Dataloader initialized #####")
    torch.set_float32_matmul_precision("high")
    early_stop_cb = callbacks.EarlyStopping(
        "valid_loss", patience=50, check_finite=True, verbose=True
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
    start_time = time.time()
    trainer.fit(model=flow_obj, datamodule=sig_dat)
    end_time = time.time()
    print(f"Training time: {end_time - start_time} seconds")
    torch.save(flow_obj.state_dict(), 'org_resnet.pth')
    # For testing/inference
    start_time = time.time()
    trainer.test(model=flow_obj, datamodule=sig_dat, ckpt_path=None)
    end_time = time.time()
    print(f"Inference time: {end_time - start_time} seconds")
    
    #trainer.fit(model=flow_obj, datamodule=sig_dat)
    # trainer.test(model=flow_obj, datamodule=sig_dat, ckpt_path=ckpt_path)

if __name__ == '__main__':
    main()
