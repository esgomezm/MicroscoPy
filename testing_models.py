from src.trainers import *
from src.utils import load_yaml

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "1";

dataset_config = load_yaml('./general_configs/dataset_configuration.yaml')
model_config = load_yaml('./general_configs/model_configuration.yaml')

train_config = None

for dataset_name in ['EM', 'MitoTracker_small', 'F-actin', 'ER', 'MT', 'LiveFActinDataset']: #, 'MT-SMLM_all']:

    train_lr, train_hr, val_lr, val_hr, test_lr, test_hr = dataset_config[dataset_name]['data_paths']

    dataset_root = 'datasets'
    train_lr_path = os.path.join(dataset_root, dataset_name, train_lr) if train_lr is not None else None
    train_hr_path = os.path.join(dataset_root, dataset_name, train_hr) if train_hr is not None else None
    val_lr_path = os.path.join(dataset_root, dataset_name, val_lr) if val_lr is not None else None
    val_hr_path = os.path.join(dataset_root, dataset_name, val_hr) if val_hr is not None else None
    test_lr_path = os.path.join(dataset_root, dataset_name, test_lr) if test_lr is not None else None
    test_hr_path = os.path.join(dataset_root, dataset_name, test_hr) if test_hr is not None else None

    for model_name in ['cddpm']: #['unet', 'rcan', 'dfcan', 'wdsr', 'wgan', 'esrganplus']:

        if train_config is None:
            test_metric_indexes = [69,  7, 36, 75, 74, 30, 12, 42, 87, 0]

            optimizer = 'AdamW'  #'Adam', 'AdamW' 'Adamax', 'RMSprop', 'SGD'
            discriminator_optimizer = 'AdamW'  #'Adam', 'AdamW', 'Adamax', 'RMSprop', 'SGD'
            scheduler = 'OneCycle'  #'ReduceOnPlateau', 'OneCycle', 'CosineDecay', 'MultiStepScheduler'
            discriminator_lr_scheduler = 'OneCycle'  #'ReduceOnPlateau', 'OneCycle', 'CosineDecay', 'MultiStepScheduler'

            #model_name = 'unet' # ['unet', 'rcan', 'dfcan', 'wdsr', 'wgan', 'esrganplus']
            seed = 666
            batch_size = 4
            number_of_epochs = 2
            lr = 0.001
            discriminator_lr = 0.001
            additional_folder = ""

            for batch_szie in [1,2,4]:
                for number_of_epochs in [5,10,20]:
                    for lr in [0.001, 0.005, 0.0005]:

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

                        model = train_configuration(
                                        data_name=dataset_name, 
                                        train_lr_path=train_lr_path, train_hr_path=train_hr_path, 
                                        val_lr_path=val_lr_path, val_hr_path=val_hr_path,
                                        test_lr_path=test_lr_path, test_hr_path=test_hr_path, 
                                        additional_folder=additional_folder, train_config=train_config,
                                        model_name=model_name, model_configuration=model_config,
                                        verbose=1
                                        )
        else:
            model = train_configuration(
                                    data_name=dataset_name, 
                                    train_lr_path=train_lr_path, train_hr_path=train_hr_path, 
                                    val_lr_path=val_lr_path, val_hr_path=val_hr_path,
                                    test_lr_path=test_lr_path, test_hr_path=test_hr_path, 
                                    additional_folder=additional_folder, train_config=train_config,
                                    model_name=model_name, model_configuration=model_config,
                                    verbose=1
                                    )