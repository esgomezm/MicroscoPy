from src.trainers import *

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

model_name = 'rcan' # ['unet', 'rcan', 'dfcan', 'wdsr', 'wgan', 'esrganplus']
seed = 666
batch_size = 8
number_of_epochs = 20
lr = 0.001
discriminator_lr = 0.0001
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

    print('data_name: {}'.fomat(model_trainer.data_name))
    print('train_lr_path: {}'.fomat(model_trainer.train_lr_path))
    print('train_hr_path: {}'.fomat(model_trainer.train_hr_path))
    print('crappifier_method: {}'.fomat(model_trainer.crappifier_method))
    print('scale_factor: {}'.fomat(model_trainer.scale_factor))
    print('num_patches: {}'.fomat(model_trainer.num_patches))
    print('lr_patch_size_x: {}'.fomat(model_trainer.lr_patch_size_x))
    print('lr_patch_size_y: {}'.fomat(model_trainer.lr_patch_size_y))
    print('validation_split: {}'.fomat(model_trainer.validation_split))
    print('model_name: {}'.fomat(model_trainer.model_name))
    print('number_of_epochs: {}'.fomat(model_trainer.number_of_epochs))
    print('batch_size: {}'.fomat(model_trainer.batch_size))
    print('learning_rate: {}'.fomat(model_trainer.learning_rate))
    print('discriminator_learning_rate: {}'.fomat(model_trainer.discriminator_learning_rate))
    print('optimizer_name: {}'.fomat(model_trainer.optimizer_name))
    print('discriminator_optimizer: {}'.fomat(model_trainer.discriminator_optimizer))
    print('lr_scheduler_name: {}'.fomat(model_trainer.lr_scheduler_name))
    print('discriminator_lr_scheduler: {}'.fomat(model_trainer.discriminator_lr_scheduler))
    print('model_configuration: {}'.fomat(model_trainer.model_configuration))
    print('test_metric_indexes: {}'.fomat(model_trainer.test_metric_indexes))
    print('additional_folder: {}'.fomat(model_trainer.additional_folder))
    print('seed: {}'.fomat(model_trainer.seed))

    model_trainer.train_model()
    model_trainer.predict_images()


    
