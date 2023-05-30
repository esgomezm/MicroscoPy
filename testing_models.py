from src.trainers import *
from src.utils import load_yaml
import gc

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "0";

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

dataset_config = load_yaml('./general_configs/dataset_configuration_placebo.yaml')
model_config = load_yaml('./general_configs/model_configuration.yaml')

train_config = None

for dataset_name in ['LiveFActinDataset', 'F-actin', 'ER', 'MT']: #'EM', 'MitoTracker_small', 'F-actin', 'ER', 'MT', 'MT-SMLM_all']:

    train_lr, train_hr, val_lr, val_hr, test_lr, test_hr = dataset_config[dataset_name]['data_paths']

    dataset_root = 'datasets'
    train_lr_path = os.path.join(dataset_root, dataset_name, train_lr) if train_lr is not None else None
    train_hr_path = os.path.join(dataset_root, dataset_name, train_hr) if train_hr is not None else None
    val_lr_path = os.path.join(dataset_root, dataset_name, val_lr) if val_lr is not None else None
    val_hr_path = os.path.join(dataset_root, dataset_name, val_hr) if val_hr is not None else None
    test_lr_path = os.path.join(dataset_root, dataset_name, test_lr) if test_lr is not None else None
    test_hr_path = os.path.join(dataset_root, dataset_name, test_hr) if test_hr is not None else None

    for model_name in ['unet', 'rcan', 'wdsr']: #['unet', 'rcan', 'dfcan', 'wdsr', 'wgan', 'esrganplus', 'cddpm']:

        test_metric_indexes = [69,  7, 36, 75, 74, 30, 12, 42, 87, 0]

        optimizer = 'Adam'  #'Adam', 'AdamW' 'Adamax', 'RMSprop', 'SGD'
        discriminator_optimizer = 'Adam'  #'Adam', 'AdamW', 'Adamax', 'RMSprop', 'SGD'
        scheduler = 'OneCycle'  #'ReduceOnPlateau', 'OneCycle', 'CosineDecay', 'MultiStepScheduler'
        discriminator_lr_scheduler = 'OneCycle'  #'ReduceOnPlateau', 'OneCycle', 'CosineDecay', 'MultiStepScheduler'

        seed = 666
        batch_size = 4
        number_of_epochs = 2
        lr = 0.001
        discriminator_lr = 0.001
        additional_folder = "visualization"

        for batch_size in [4]: #[1,2,4]:
            for number_of_epochs in [10,50,100]: #[5,10,20]:
                for lr, discriminator_lr in [(0.001,0.001), (0.005,0.005), (0.0001,0.0001), (0.0005,0.0005)]: #[0.001, 0.005, 0.0005]:

                    # Update the patience to be equal to the number of epochs
                    model_config['optim']['early_stop']['patience'] = number_of_epochs

                    validation_split = 0.1
                    data_augmentation = ['rotation', 'horizontal_flip', 'vertical_flip']
                    datagen_sampling_pdf = 0

                    train_config = {'model':model_name,
                                'dataset_name':dataset_name,
                                'optimizer':optimizer,
                                'discriminator_optimizer':discriminator_optimizer,
                                'scheduler':scheduler,
                                'discriminator_lr_scheduler':discriminator_lr_scheduler,
                                'seed':seed,
                                'batch_size':batch_size,
                                'number_of_epochs':number_of_epochs,
                                'learning_rate':lr,
                                'discriminator_learning_rate':discriminator_lr,
                                'validation_split':validation_split,
                                'data_augmentation':data_augmentation,
                                'test_metric_indexes':test_metric_indexes,
                                'datagen_sampling_pdf':datagen_sampling_pdf
                                }
                    
                    train_config['dataset_config'] = dataset_config[dataset_name]
                    train_config['model_config'] = model_config[model_name]
                    train_config['optim_config'] = model_config['optim']
                    train_config['other_config'] = model_config['others']
                    
                    try:
                        scale_factor = train_config['dataset_config']['scale']
                        save_folder = 'scale' + str(scale_factor)

                        if additional_folder:
                            save_folder += '_' + additional_folder

                        saving_path = './results/{}/{}/{}/scale{}_epc{}_btch{}_lr{}_optim-{}_lrsched-{}_seed{}'.format(
                                                                                            dataset_name, 
                                                                                            model_name,
                                                                                            save_folder, 
                                                                                            scale_factor, 
                                                                                            number_of_epochs,
                                                                                            batch_size, 
                                                                                            lr, 
                                                                                            optimizer,
                                                                                            scheduler,
                                                                                            seed)
                    
                        test_metric_path = os.path.join(saving_path, 'test_metrics')
                        if os.path.exists(test_metric_path) and len(os.listdir(test_metric_path)) > 0:
                            print(f'{saving_path} - model combination already trained.')
                        else:
                            model = train_configuration(
                                            data_name=dataset_name, 
                                            train_lr_path=train_lr_path, train_hr_path=train_hr_path, 
                                            val_lr_path=val_lr_path, val_hr_path=val_hr_path,
                                            test_lr_path=test_lr_path, test_hr_path=test_hr_path, 
                                            additional_folder=additional_folder, train_config=train_config,
                                            model_name=model_name, model_configuration=model_config,
                                            verbose=0
                                            )
                            del model
                    except Exception as e:
                      print(f'\033[91mERROR\033[0m - In config {dataset_name} {model_name} {number_of_epochs} {lr}')
                      print(e)
                    gc.collect()