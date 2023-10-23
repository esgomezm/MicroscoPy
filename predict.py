from microscopy.trainers import predict_configuration
from omegaconf import DictConfig
import omegaconf
import hydra
import gc
import os
import shutil
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def load_path(dataset_root, dataset_name, folder):
    if folder is not None:
        return os.path.join(dataset_root, dataset_name, folder)
    else:
        return None

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:

    # dataset_combination = ["ER"] #"LiveFActinDataset", "EM", "MitoTracker_small", "F-actin", "ER", "MT", "MT-SMLM_all"
    # model_combination = ["unet"]  # "unet", "rcan", "dfcan", "wdsr", "wgan", "esrganplus", "cddpm"
    # batch_size_combination = [8] 
    # num_epochs_combination = [100]
    # lr_combination = [(0.001,0.001)]
    # scheduler_combination = ['ReduceOnPlateau', 'OneCycle', 'CosineDecay', 'MultiStepScheduler'] #'ReduceOnPlateau', 'OneCycle', 'CosineDecay', 'MultiStepScheduler'
    # optimizer_combination = ['adam', 'adamW' 'adamax', 'rms_prop', 'sgd']  #'adam', 'adamW' 'adamax', 'rms_prop', 'sgd'
    # base_folder = ''
    
    dataset_name = 'EM'
    saving_path = './multiple_predictions/EM/cddpm/scale4/epc2001_btch8_lr0.0005_optim-adam_lrsched-CosineDecay_seed666_1'

    actual_cfg = omegaconf.OmegaConf.load(os.path.join(saving_path, 'train_configuration.yaml'))

    actual_cfg.dataset_name = dataset_name
    train_lr, train_hr, val_lr, val_hr, test_lr, test_hr = actual_cfg.used_dataset.data_paths

    dataset_root = "datasets" if os.path.exists("datasets") else "../datasets"
    train_lr_path = load_path(dataset_root, dataset_name, train_lr)
    train_hr_path = load_path(dataset_root, dataset_name, train_hr)
    val_lr_path = load_path(dataset_root, dataset_name, val_lr)
    val_hr_path = load_path(dataset_root, dataset_name, val_hr)
    test_lr_path = load_path(dataset_root, dataset_name, test_lr)
    test_hr_path = load_path(dataset_root, dataset_name, test_hr)

    # cfg.hyperparam.seed = 666
    
    for i in range(1, 10):
        random_seed = np.random.randint(0, 1000)
        actual_cfg.hyperparam.seed = random_seed
        folder_name = f'{i:02d}_seed_{random_seed}' # same_seed' #

        print(folder_name)

        model = predict_configuration(
            config=actual_cfg,
            train_lr_path=train_lr_path,
            train_hr_path=train_hr_path,
            val_lr_path=val_lr_path,
            val_hr_path=val_hr_path,
            test_lr_path=test_lr_path,
            test_hr_path=test_hr_path,
            saving_path=saving_path,
            verbose=0,
        )
        del model
        gc.collect()

        os.makedirs(os.path.join(saving_path, folder_name))
        shutil.move(os.path.join(saving_path, 'test_metrics'), os.path.join(saving_path, folder_name))
        shutil.move(os.path.join(saving_path, 'predicted_images'), os.path.join(saving_path, folder_name))

    # for dataset_name in dataset_combination: 
    #     cfg.dataset_name = dataset_name
    #     train_lr, train_hr, val_lr, val_hr, test_lr, test_hr = cfg.used_dataset.data_paths

    #     dataset_root = "datasets" if os.path.exists("datasets") else "../datasets"
    #     train_lr_path = load_path(dataset_root, dataset_name, train_lr)
    #     train_hr_path = load_path(dataset_root, dataset_name, train_hr)
    #     val_lr_path = load_path(dataset_root, dataset_name, val_lr)
    #     val_hr_path = load_path(dataset_root, dataset_name, val_hr)
    #     test_lr_path = load_path(dataset_root, dataset_name, test_lr)
    #     test_hr_path = load_path(dataset_root, dataset_name, test_hr)

    #     for model_name in model_combination: 
    #         for batch_size in batch_size_combination:  
    #             for num_epochs in num_epochs_combination:                  
    #                 for lr, discriminator_lr in lr_combination:
    #                     for scheduler in scheduler_combination:
    #                         for optimizer in optimizer_combination:
    #                             cfg.model_name = model_name
    #                             cfg.hyperparam.batch_size = batch_size
    #                             cfg.hyperparam.num_epochs = num_epochs
    #                             cfg.hyperparam.lr = lr
    #                             cfg.hyperparam.discriminator_lr = discriminator_lr

    #                             cfg.hyperparam.scheduler = scheduler
    #                             cfg.hyperparam.discriminator_lr_scheduler = scheduler
    #                             cfg.hyperparam.optimizer = optimizer
    #                             cfg.hyperparam.discriminator_optimizer = optimizer

    #                             cfg.model.optim.early_stop.patience = num_epochs
    #                             save_folder = "scale" + str(cfg.used_dataset.scale)
    #                             if cfg.hyperparam.additional_folder is not None:
    #                                 save_folder += "_" + cfg.hyperparam.additional_folder

    #                             saving_path = "./results/{}/{}/{}/epc{}_btch{}_lr{}_optim-{}_lrsched-{}_seed{}".format(
    #                                 cfg.dataset_name,
    #                                 cfg.model_name,
    #                                 save_folder,
    #                                 cfg.hyperparam.num_epochs,
    #                                 cfg.hyperparam.batch_size,
    #                                 cfg.hyperparam.lr,
    #                                 cfg.hyperparam.optimizer,
    #                                 cfg.hyperparam.scheduler,
    #                                 cfg.hyperparam.seed
    #                             )

    #                             test_metric_path = os.path.join(saving_path, "test_metrics")
    #                             if (
    #                                 os.path.exists(test_metric_path)
    #                                 and len(os.listdir(test_metric_path)) > 0
    #                             ):
    #                                 model = predict_configuration(
    #                                     config=cfg,
    #                                     train_lr_path=train_lr_path,
    #                                     train_hr_path=train_hr_path,
    #                                     val_lr_path=val_lr_path,
    #                                     val_hr_path=val_hr_path,
    #                                     test_lr_path=test_lr_path,
    #                                     test_hr_path=test_hr_path,
    #                                     saving_path=saving_path,
    #                                     verbose=1
    #                                 )
    #                                 del model
    #                                 gc.collect()
    #                             else:
    #                                 print(f"{saving_path} - model combination is not trained, therefore the prediction cannot be done.")

my_app()
