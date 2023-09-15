from microscopy.trainers import train_configuration
from omegaconf import DictConfig
import hydra
import gc
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

gpu_id = [1] #int(os.environ["CUDA_VISIBLE_DEVICES"])

def load_path(dataset_root, dataset_name, folder):
    if folder is not None:
        return os.path.join(dataset_root, dataset_name, folder)
    else:
        return None

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    
    dataset_combination = ["ER"] #"LiveFActinDataset", "EM", "F-actin", "ER", "MT", "MT-SMLM_registered"
    model_combination = ["wgan"]  # "unet", "rcan", "dfcan", "wdsr", "wgan", "esrganplus", "cddpm"
    batch_size_combination = [8]
    num_epochs_combination = [100]
    lr_combination = [(0.0005,0.0005)]
    scheduler_combination = ['Fixed', 'ReduceOnPlateau', 'MultiStepScheduler'] #'Fixed', 'ReduceOnPlateau', 'OneCycle', 'CosineDecay', 'MultiStepScheduler'
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

                                base_folder = 'WGAN'

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

                                if cfg.model_name in ["wgan", "esrganplus"]:
                                    number_of_critic_steps = cfg.used_model.n_critic_steps
                                    lambda_gp = cfg.used_model.lambda_gp
                                    recloss = cfg.used_model.recloss
                                    # number_of_critic_steps = 1
                                    # cfg.used_model.n_critic_steps = number_of_critic_steps
                                    # cfg.used_model.g_layers = 15
                                    # cfg.used_model.lambda_gp = 0.5
                                    # cfg.used_model.recloss = 0.5 # 100.0
                                    base_folder=f"{base_folder}/cs_{number_of_critic_steps}-lgp_{lambda_gp}-rec_{recloss}"

                                save_folder = "scale" + str(cfg.used_dataset.scale)
                                if cfg.hyperparam.additional_folder is not None:
                                    save_folder += "_" + cfg.hyperparam.additional_folder

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
                                    saving_path = saving_path[:-1] + str(int(saving_path[-1]) + 1) 

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
                                    data_on_memory=0,
                                    gpu_id=gpu_id
                                )
                                del model

                                gc.collect()
my_app()
