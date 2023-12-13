from microscopy.trainers import train_configuration
from omegaconf import DictConfig
import hydra
import gc
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

def load_path(dataset_root, dataset_name, folder):
    if folder is not None:
        return os.path.join(dataset_root, dataset_name, folder)
    else:
        return None

# EM_crap_2x_03
# EM_crap_2x_05
# EM_crap_2x_09
# EM_crap_4x_03
# EM_crap_4x_05
# EM_crap_4x_09
# EM_down_2x
# EM_down_4x
# EM_old_crap_4x

config_name = "config"

@hydra.main(version_base=None, config_path="conf", config_name=config_name)
def my_app(cfg: DictConfig) -> None:
    
    dataset_combination = ["LiveFActinDataset"] #"LiveFActinDataset", "EM", "F-actin", "ER", "MT", "MT-SMLM_registered"
    model_combination = ["unet", "rcan", "wdsr", "dfcan"]  # "unet", "rcan", "dfcan", "wdsr", "wgan", "esrganplus", "cddpm"
    batch_size_combination = [4]
    num_epochs_combination = [200]
    lr_combination = [(0.001, 0.001),(0.0001, 0.0001)]


    scheduler_combination = ['Fixed', 'OneCycle', 'CosineDecay'] #'Fixed', 'ReduceOnPlateau', 'OneCycle', 'CosineDecay', 'MultiStepScheduler'
    optimizer_combination = ['adam']  #'adam', 'adamW', 'adamax', 'rms_prop', 'sgd'
    
    for dataset_name in dataset_combination:  
        cfg.dataset_name = dataset_name
        train_lr, train_hr, val_lr, val_hr, test_lr, test_hr = cfg.used_dataset.data_paths

        dataset_root = "datasets" if os.path.exists("datasets") else "../datasets"
        train_lr_path = load_path(dataset_root, dataset_name, train_lr)
        train_hr_path = load_path(dataset_root, dataset_name, train_hr)
        val_lr_path = load_path(dataset_root, dataset_name, val_lr)
        val_hr_path = load_path(dataset_root, dataset_name, val_hr)
        test_lr_path = load_path(dataset_root, dataset_name, test_lr)
        test_hr_path = load_path(dataset_root, dataset_name, test_hr)

        for model_name in model_combination: 
            for batch_size in batch_size_combination:  
                for num_epochs in num_epochs_combination:                  
                    for lr, discriminator_lr in lr_combination:
                        for scheduler in scheduler_combination:
                            for optimizer in optimizer_combination:

                                base_folder = 'results'

                                cfg.model_name = model_name
                                cfg.hyperparam.batch_size = batch_size
                                cfg.hyperparam.num_epochs = num_epochs
                                cfg.hyperparam.lr = lr
                                cfg.hyperparam.discriminator_lr = discriminator_lr

                                cfg.hyperparam.scheduler = scheduler
                                cfg.hyperparam.discriminator_lr_scheduler = scheduler
                                cfg.hyperparam.optimizer = optimizer
                                cfg.hyperparam.discriminator_optimizer = optimizer

                                cfg.model.optim.early_stop.patience = num_epochs

                                save_folder = "scale" + str(cfg.used_dataset.scale)
                                if cfg.hyperparam.additional_folder is not None:
                                    save_folder += "_" + cfg.hyperparam.additional_folder
                                
                                # save_folder = save_folder + "/" + config_name

                                saving_path = "./{}/{}/{}/{}/epc{}_btch{}_lr{}_optim-{}_lrsched-{}_seed{}_1".format(
                                    base_folder, 
                                    cfg.dataset_name,
                                    cfg.model_name,
                                    save_folder,
                                    cfg.hyperparam.num_epochs,
                                    cfg.hyperparam.batch_size,
                                    cfg.hyperparam.lr,
                                    cfg.hyperparam.optimizer,
                                    cfg.hyperparam.scheduler,
                                    cfg.hyperparam.seed
                                )


                                test_metric_path = os.path.join(saving_path, "test_metrics")
                                if (os.path.exists(test_metric_path) and len(os.listdir(test_metric_path)) > 0):
                                    # In case you want to skip existing configurations
                                    continue
                                    # In case you want to repeat existing configurations
                                    # saving_path = saving_path[:-1] + str(int(saving_path[-1]) + 1) 

                                model = train_configuration(
                                    config=cfg,
                                    train_lr_path=train_lr_path,
                                    train_hr_path=train_hr_path,
                                    val_lr_path=val_lr_path,
                                    val_hr_path=val_hr_path,
                                    test_lr_path=test_lr_path,
                                    test_hr_path=test_hr_path,
                                    saving_path=saving_path,
                                    verbose=0, # 0, 1 or 2
                                    data_on_memory=0
                                )
                                del model

                                gc.collect()
my_app()
