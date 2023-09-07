from microscopy.trainers import predict_configuration
from omegaconf import DictConfig
import omegaconf
import hydra
import gc
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def load_path(dataset_root, dataset_name, folder):
    if folder is not None:
        return os.path.join(dataset_root, dataset_name, folder)
    else:
        return None

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:

    for dataset_name in os.listdir(os.path.join('results')): 
        cfg.dataset_name = dataset_name
        train_lr, train_hr, val_lr, val_hr, test_lr, test_hr = cfg.used_dataset.data_paths

        dataset_root = "datasets" if os.path.exists("datasets") else "../datasets"
        train_lr_path = load_path(dataset_root, dataset_name, train_lr)
        train_hr_path = load_path(dataset_root, dataset_name, train_hr)
        val_lr_path = load_path(dataset_root, dataset_name, val_lr)
        val_hr_path = load_path(dataset_root, dataset_name, val_hr)
        test_lr_path = load_path(dataset_root, dataset_name, test_lr)
        test_hr_path = load_path(dataset_root, dataset_name, test_hr)

        for model_name in os.listdir(os.path.join('results', dataset_name)): 
            for scale_folder in os.listdir(os.path.join('results', dataset_name, model_name)): 
                for config in os.listdir(os.path.join('results', dataset_name, model_name, scale_folder)):
                    config_path = os.path.join('results', dataset_name, model_name, scale_folder, config)

                    actual_cfg = omegaconf.OmegaConf.load(os.path.join(config_path, 'train_configuration.yaml'))

                    weights_path = os.path.join(config_path, "weights_best.h5")
                    if (
                        os.path.exists(weights_path) and not os.path.exists(os.path.join(config_path, "test_metrics")) 
                    ):
                    
                        print(f"{config_path} - will be evalauted.")
                        try :

                            model = predict_configuration(
                                config=actual_cfg,
                                train_lr_path=train_lr_path,
                                train_hr_path=train_hr_path,
                                val_lr_path=val_lr_path,
                                val_hr_path=val_hr_path,
                                test_lr_path=test_lr_path,
                                test_hr_path=test_hr_path,
                                saving_path=config_path,
                                verbose=1
                            )
                            del model
                            gc.collect()
                        except Exception as e:
                            print(f"{config_path} - model combination GAVE ERROR.")
                            print(e)
                    else:
                        if not os.path.exists(weights_path):
                            print(f"{config_path} - model combination is not trained, therefore the prediction cannot be done.")
                        elif os.path.exists(os.path.join(config_path, "test_metrics")):
                            print(f"{config_path} - has already predictions done.")
                        else:
                            print(f'{config_path} - SOMETHIN STRANGE.')
my_app()
