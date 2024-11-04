import os
import time
import torch
import socket
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train.torch import TorchTrainer
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment, RayTrainReportCallback, prepare_trainer

from data import SignalDataSet
from lightning.pytorch import Trainer, callbacks, loggers
from ml4gw.waveforms import IMRPhenomD
from mlpe.architectures.embeddings import MobileNetV3_1D
from mlpe.architectures.flows import MaskedAutoRegressiveFlow
from mlpe.injection.priors import nonspin_bbh_chirp_mass_q_parameter_sampler

torch.cuda.empty_cache()

class ProxyOpt:
    def __init__(self, opt_class, **optim_kwargs):
        self.opt_class = opt_class
        self.optim_kwargs = optim_kwargs
    def __call__(self, *args, **kwargs):
        kwargs.update(self.optim_kwargs)
        return self.opt_class(*args, **kwargs)

def train_func(config):
    background_path = os.getenv('DATA_DIR') + "/background.h5"
    ifos = ["H1", "L1"]
    batch_size = config.get("batch_size", 500)  
    batches_per_epoch = 100
    sample_rate = 2048
    time_duration = 4
    f_max = 200
    f_min = 20
    f_ref = 40
    valid_frac = 0.2
    learning_rate = config.get("learning_rate", 1e-3)

    mobilenetv2_context_dim = 100  
    mobilenetv2_layers = [1, 2, 3]  
    mobilenetv2_expand_ratios = [1, 4, 4] 
    mobilenetv2_initial_channels = 32  
    mobilenetv2_width_mult = 1.0

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
    num_transforms = config.get("num_transforms", 60)  
    num_blocks = config.get("num_blocks", 5)  
    hidden_features = config.get("hidden_features", 120)  

    param_dim, n_ifos, strain_dim = (
        len(inference_params),
        len(ifos),
        int(sample_rate * time_duration),
    )

    embedding = MobileNetV3_1D(
        num_ifos=n_ifos, 
        context_dim=mobilenetv2_context_dim, 
        version="small"
    )

    prior_func = nonspin_bbh_chirp_mass_q_parameter_sampler
    
    optimizer_kwargs = dict(weight_decay=config['weight_decay'])
    if 'SGD' in config['optimizer_class'].__name__:
        optimizer_kwargs.update({'momentum': config['momentum']})

    optimizer = ProxyOpt(config['optimizer_class'], **optimizer_kwargs)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    
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

    print("##### Initializing trainer #####")
    '''
    trainer = Trainer(
        max_epochs=10,
        log_every_n_steps=100,
        callbacks=[early_stop_cb, lr_monitor],
        logger=logger,
        gradient_clip_val=10.0,
    )
    '''
    trainer = Trainer(
        devices="auto",
        accelerator="auto",
        max_epochs=10,
        log_every_n_steps=100,
        logger=logger,
        gradient_clip_val=10.0,
        strategy=RayDDPStrategy(find_unused_parameters=True),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
    )
    trainer = prepare_trainer(trainer)
    start_time = time.time()
    trainer.fit(model=flow_obj, datamodule=sig_dat)
    #valid_loss = trainer.callback_metrics.get("valid_loss")
    #tune.report(valid_loss=valid_loss)
    end_time = time.time()
    print(f"Training time: {end_time - start_time} seconds")

    # For testing/inference
    start_time = time.time()
    trainer.test(model=flow_obj, datamodule=sig_dat, ckpt_path=None)
    end_time = time.time()
    print(f"Inference time: {end_time - start_time} seconds")

def main():
    ray.init(configure_logging=False)

    search_space = {
        "num_transforms": tune.choice([50, 60, 70]),
        "num_blocks": tune.choice([4, 5, 6]),
        "hidden_features": tune.choice([100, 120, 150]),
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([400, 500, 600]),
        "optimizer_class": tune.choice([torch.optim.AdamW, torch.optim.SGD]),
        "momentum": tune.loguniform(1e-4, 1e-1),
        "weight_decay": tune.loguniform(1e-4, 1e-1),
    }

    num_epochs = 30
    num_samples = 10

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=5,
        reduction_factor=2
    )

    scaling_config = ScalingConfig(
        num_workers=1, use_gpu=True,
        resources_per_worker={"CPU": 1, "GPU": 1}
    )

    run_config = RunConfig(
        storage_path=os.getenv("SCRATCH_DIR") + "/ray_results",
        name="hyperparameter_tuning",
        checkpoint_config=CheckpointConfig(
            num_to_keep=3,
            checkpoint_score_attribute="valid_loss",
            checkpoint_score_order="min",
        ),
    )

    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="valid_loss",
            mode="min",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )
    
    results = tuner.fit()
    print("Best hyperparameters found were: ", results.get_best_result().config)

if __name__ == '__main__':
    main()
