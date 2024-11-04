import os

import torch
import torch.nn as nn
from data import SignalDataSet
from lightning.pytorch import Trainer, callbacks, loggers
from lightning.pytorch.cli import LightningCLI
import bilby
from ml4gw.waveforms import IMRPhenomD, TaylorF2
from mlpe.architectures.embeddings import ResNet, QuantizedResNet, QuantizedBasicBlock, BasicBlock
from ml4gw.nn.resnet import ResNet1D
from mlpe.architectures.embeddings import DenseNet1D
from mlpe.architectures.flows import MaskedAutoRegressiveFlow
from mlpe.injection.priors import nonspin_bbh_component_mass, nonspin_bbh_chirp_mass_q, nonspin_bbh_component_mass_parameter_sampler, nonspin_bbh_chirp_mass_q_parameter_sampler
from mlpe.logging import configure_logging

import torch.ao.quantization as quantization


def main():
    background_path = os.getenv('DATA_DIR') + "/background.h5"
    ifos = ["H1", "L1"]
    batch_size = 500
    batches_per_epoch = 200
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
        #"mass_1",
        #"mass_2",
        "chirp_mass",
        "mass_ratio",
        "luminosity_distance",
        "phase",
        "theta_jn",
        "dec",
        "psi",
        "phi",
       # "ra",
    ]
    num_transforms = 60
    num_blocks = 5
    hidden_features = 120

    optimizer = torch.optim.AdamW
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    param_dim, n_ifos, strain_dim = (
        len(inference_params),
        len(ifos),
        int(sample_rate * time_duration),
    )


    # embedding = ResNet(
    #     (n_ifos, strain_dim),
    #     context_dim=resnet_context_dim,
    #     layers=resnet_layers,
    #     norm_groups=resnet_norm_groups,
    # )
    
    embedding = QuantizedResNet(
        (n_ifos, strain_dim),
         context_dim=resnet_context_dim,
         layers=resnet_layers,
         norm_groups=resnet_norm_groups,
    )
   
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
        hidden_features=hidden_features,
    )
    
    
    ckpt_path = os.getenv("BASE_DIR") + "/pl-logdir/phenomd-60-transforms-4-4-resnet-wider-dl/version_815/checkpoints/epoch=232-step=46600.ckpt"
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
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
        max_epochs=1,
        log_every_n_steps=100,
        callbacks=[early_stop_cb, lr_monitor],
        logger=logger,
        gradient_clip_val=10.0,
        # precision='16-mixed'
        accelerator='cpu'
    )
    
    # trainer.fit(model=flow_obj, datamodule=sig_dat)
    
    model_int8 = torch.ao.quantization.quantize_dynamic(
        flow_obj.embedding_net,  
        {torch.nn.Linear, torch.nn.Conv1d, QuantizedBasicBlock},  
        dtype=torch.qint8
    )  
    flow_obj.embedding_net=model_int8

    print(flow_obj.embedding_net)
    torch.save(flow_obj.state_dict(), 'quantized_ptdq_model_state.pth')
    trainer.test(model=flow_obj, datamodule=sig_dat, ckpt_path=None)
  
if __name__ == '__main__':
    main()


