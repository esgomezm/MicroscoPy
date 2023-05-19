from src.trainers import *


from matplotlib import pyplot as plt
import numpy as np

def plot_images(img_list):
    for img in img_list:
        plt.figure(figsize=(10,10))
        plt.imshow(img, 'gray')
        plt.show()

def print_info(data):
    print('Shape: {}'.format(data.shape))
    print('dtype: {}'.format(data.dtype))
    print('Min: {}'.format(data.min()))    
    print('Min: {}'.format(data.max()))    
    print('Mean: {}'.format(data.mean()))    

dataset_config = {'EM': [None, 'train', None, None, None, 'test'],
                  'MitoTracker_small': [None, 'train', None, None, None, 'test'],
                  'F-actin': ['train/training_wf', 'train/training_gt', 'val/validate_wf', 'val/validate_gt', 'test/test_wf/level_01', 'test/test_gt'],
                  'ER': ['train/training_wf', 'train/training_gt', 'val/validate_wf', 'val/validate_gt', 'test/test_wf/level_01', 'test/test_gt/level_06'],
                  'MT': ['train/training_wf', 'train/training_gt', 'val/validate_wf', 'val/validate_gt', 'test/test_wf/level_01', 'test/test_gt'],
                  'LiveFActinDataset': ['train_split/wf', 'train_split/gt', 'val_split/wf', 'val_split/gt', 'test_split/wf', 'test_split/gt']
                  }

crappifier_config = {'EM': 'em_crappify', 
                     'MitoTracker_small': 'fluo_crappify',
                     'F-actin': 'fluo_SP_AG_D_sameas_preprint',
                     'ER': 'fluo_SP_AG_D_sameas_preprint',
                     'MT': 'fluo_SP_AG_D_sameas_preprint',
                     'LiveFActinDataset': 'fluo_SP_AG_D_sameas_preprint'}

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "2";

model_configuration = {'optim': {'early_stop':{'loss':'val_ssim_loss','mode':'max', 'patience':10},
                                 'adam':{'beta1':0.5,'beta2':0.9,'epsilon':1e-07},
                                 'adamax':{'beta1':0.5,'beta2':0.9,'epsilon':1e-07},
                                 'adamW':{'decay':0.004,'beta1':0.5,'beta2':0.9,'epsilon':1e-07},
                                 'sgd_momentum':0.9,
                                 'ReduceOnPlateau':{'monitor':'val_loss','factor':0.5,'patience':3},
                                 'MultiStepScheduler':{'lr_steps':[50000, 100000, 200000, 300000],
                                                       'lr_rate_decay':0.5}},
                       'rcan': {'num_filters':16,
                                'percp_coef': 1000},
                       'dfcan': {'n_ResGroup': 4, 'n_RCAB': 4},
                       'wdsr': {'num_res_blocks': 32},
                       'unet': {'init_channels': 16,
                                'depth': 4,
                                'upsample_method': 'SubpixelConv2D',
                                'maxpooling': False,
                                'percp_coef': 10},
                       'wgan': {'g_layers': 15,
                                'd_layers': 5,
                                'recloss': 100.0,
                                'lambda_gp':10},
                       'esrganplus': {'n_critic_steps':5},
                       'others': {'positional_encoding':False,
                                  'positional_encoding_channels':64}
                      }

test_metric_indexes = [69,  7, 36, 75, 74, 30, 12, 42, 87,  0]

optimizer = 'Adam'  #'Adam', 'Adamax', 'RMSprop', 'SGD'
discriminator_optimizer = 'Adam'  #'Adam', 'Adamax', 'RMSprop', 'SGD'
scheduler = 'OneCycle'  #'ReduceOnPlateau', 'OneCycle', 'CosineDecay', 'MultiStepScheduler'
discriminator_lr_scheduler = 'OneCycle'  #'ReduceOnPlateau', 'OneCycle', 'CosineDecay', 'MultiStepScheduler'

model_name = 'unet' # ['unet', 'rcan', 'dfcan', 'wdsr', 'wgan', 'esrganplus']
seed = 666
batch_size = 8
number_of_epochs = 20
lr = 0.001
discriminator_lr = 0.001
additional_folder = "prueba"

scale = 4

num_patches = 16
patch_size_x = 64
patch_size_y = 64
validation_split = 0.1
data_augmentation = ['rotation', 'horizontal_flip', 'vertical_flip']

for dataset_name in ['EM']:

    train_lr, train_hr, val_lr, val_hr, test_lr, test_hr = dataset_config[dataset_name]

    dataset_root = '../datasets'
    train_lr_path = os.path.join(dataset_root, dataset_name, train_lr) if train_lr is not None else None
    train_hr_path = os.path.join(dataset_root, dataset_name, train_hr) if train_hr is not None else None
    val_lr_path = os.path.join(dataset_root, dataset_name, val_lr) if val_lr is not None else None
    val_hr_path = os.path.join(dataset_root, dataset_name, val_hr) if val_hr is not None else None
    test_lr_path = os.path.join(dataset_root, dataset_name, test_lr) if test_lr is not None else None
    test_hr_path = os.path.join(dataset_root, dataset_name, test_hr) if test_hr is not None else None

    train_hr_path = '../datasets/TFM - dataset Electron Microscopy/train'
    test_hr_path = '../datasets/TFM - dataset Electron Microscopy/test'

    crappifier_method = crappifier_config[dataset_name]

    model_trainer = TensorflowTrainer('tensorflow', 
                 train_lr_path, train_hr_path, 
                 val_lr_path, val_hr_path,
                 test_lr_path, test_hr_path,
                 crappifier_method, model_name, scale, 
                 number_of_epochs, batch_size, 
                 lr, discriminator_lr, 
                 optimizer, scheduler, 
                 test_metric_indexes, additional_folder, 
                 model_configuration, seed,
                 num_patches, patch_size_x, patch_size_y, 
                 validation_split, data_augmentation,
                 discriminator_optimizer=discriminator_optimizer, 
                 discriminator_lr_scheduler=discriminator_lr_scheduler,
                 verbose=1, 
                )
                    
    model_trainer.prepare_data()

    print('Input data shape: {}'.format(model_trainer.input_data_shape))
    print('Output data shape: {}'.format(model_trainer.input_data_shape))

    for lr,hr in model_trainer.train_generator:
        print('LR')
        print_info(lr)
        print('\n')
        print('HR')
        print_info(hr)
        break

    model_trainer.configure_model()

    model_trainer.model.summary()

    print('data_name: {}'.format(model_trainer.data_name))
    print('train_lr_path: {}'.format(model_trainer.train_lr_path))
    print('train_hr_path: {}'.format(model_trainer.train_hr_path))
    print('crappifier_method: {}'.format(model_trainer.crappifier_method))
    print('scale_factor: {}'.format(model_trainer.scale_factor))
    print('num_patches: {}'.format(model_trainer.num_patches))
    print('lr_patch_size_x: {}'.format(model_trainer.lr_patch_size_x))
    print('lr_patch_size_y: {}'.format(model_trainer.lr_patch_size_y))
    print('validation_split: {}'.format(model_trainer.validation_split))
    print('model_name: {}'.format(model_trainer.model_name))
    print('number_of_epochs: {}'.format(model_trainer.number_of_epochs))
    print('batch_size: {}'.format(model_trainer.batch_size))
    print('learning_rate: {}'.format(model_trainer.learning_rate))
    print('discriminator_learning_rate: {}'.format(model_trainer.discriminator_learning_rate))
    print('optimizer_name: {}'.format(model_trainer.optimizer_name))
    print('discriminator_optimizer: {}'.format(model_trainer.discriminator_optimizer))
    print('lr_scheduler_name: {}'.format(model_trainer.lr_scheduler_name))
    print('discriminator_lr_scheduler: {}'.format(model_trainer.discriminator_lr_scheduler))
    print('model_configuration: {}'.format(model_trainer.model_configuration))
    print('test_metric_indexes: {}'.format(model_trainer.test_metric_indexes))
    print('additional_folder: {}'.format(model_trainer.additional_folder))
    print('seed: {}'.format(model_trainer.seed))

    model_trainer.train_model()
    
    from src.datasets import extract_random_patches_from_folder
    test_path = '../datasets/TFM - dataset Electron Microscopy/test'
    drawn_test_path = './data_example/drawn_test'

    test_filenames = sorted([os.path.join(filename) for filename in os.listdir(test_path)])[0:10]
    drawn_test_filenames = sorted([os.path.join(filename) for filename in os.listdir(drawn_test_path)])[0:10]

    lr_images, hr_images = extract_random_patches_from_folder(
                                    hr_data_path=test_path, 
                                    lr_data_path=None, 
                                    filenames=test_filenames, 
                                    scale_factor=scale, 
                                    crappifier_name='em_crappify', 
                                    lr_patch_shape=None, 
                                    num_patches=1)

    hr_images = np.expand_dims(hr_images, axis=-1)
    lr_images = np.expand_dims(lr_images, axis=-1)

    print('Test HR images')
    print_info(hr_images)
    print('\n')
    print('Test LR images')
    print_info(lr_images)
    print('\n')



    drawn_lr_images, drawn_hr_images = extract_random_patches_from_folder(
                                    hr_data_path=drawn_test_path, 
                                    lr_data_path=None, 
                                    filenames=drawn_test_filenames, 
                                    scale_factor=scale, 
                                    crappifier_name='em_crappify', 
                                    lr_patch_shape=None, 
                                    num_patches=1)

    drawn_hr_images = np.expand_dims(drawn_hr_images, axis=-1)
    drawn_lr_images = np.expand_dims(drawn_lr_images, axis=-1)

    print('Drawn HR images')
    print_info(drawn_hr_images)
    print('\n')
    print('Drawn LR images')
    print_info(drawn_lr_images)
    print('\n')

    from src.optimizer_scheduler_utils import select_optimizer, select_optimizer
    from src.model_utils import select_model


    optim = select_optimizer(library_name='tensorflow', optimizer_name=optimizer, 
                                    learning_rate=0.001, check_point=None,
                                    parameters=None, additional_configuration=model_configuration)

    model = select_model(model_name=model_name, input_shape=lr_images.shape,  
                        output_channels=1, scale_factor=scale, model_configuration=model_configuration)

    loss_funct = 'mean_absolute_error'
    eval_metric = 'mean_squared_error'

    model.compile(optimizer=optim, loss=loss_funct, metrics=[eval_metric])

    # Load old weights
    model.load_weights(os.path.join(model_trainer.saving_path, 'weights_best.h5') )   


    test_predictions = model.predict(lr_images, batch_size=1)
    print('Test predictions')
    print_info(test_predictions)
    print('\n')



    drawn_test_predictions = model.predict(drawn_lr_images, batch_size=1)
    print('Drawn predictions')
    print_info(drawn_test_predictions)
    print('\n')



    plot_images([test_predictions[0], drawn_test_predictions[0]])
