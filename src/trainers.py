import numpy as np
import time
import csv
import os 

from skimage import io

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint as tf_ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from . import datasets
from . import utils
from . import metrics
from . import model_utils
from . import optimizer_scheduler_utils

#######

class ModelsTrainer:
    def __init__(self, data_name, 
                 train_lr_path, train_hr_path, 
                 val_lr_path, val_hr_path,
                 test_lr_path, test_hr_path,
                 crappifier_method, model_name, scale_factor, 
                 number_of_epochs, batch_size, 
                 learning_rate, discriminator_learning_rate, 
                 optimizer_name, lr_scheduler_name, 
                 test_metric_indexes, additional_folder, 
                 model_configuration, seed,
                 num_patches, patch_size_x, patch_size_y, 
                 validation_split, data_augmentation,
                 discriminator_optimizer=None, 
                 discriminator_lr_scheduler=None,
                 verbose=0
                ):

        self.data_name = data_name

        self.train_lr_path = train_lr_path
        self.train_hr_path = train_hr_path
        train_extension_list = [os.path.splitext(e)[1] for e in os.listdir(self.train_hr_path)]
        train_extension = max(set(train_extension_list), key = train_extension_list.count)
        self.train_filenames = [x for x in os.listdir(self.train_hr_path) if x.endswith(train_extension)]

        self.val_lr_path = val_lr_path
        self.val_hr_path = val_hr_path
        val_extension_list = [os.path.splitext(e)[1] for e in os.listdir(self.val_hr_path)]
        val_extension = max(set(val_extension_list), key = val_extension_list.count)
        self.val_filenames = [x for x in os.listdir(self.val_hr_path) if x.endswith(val_extension)]

        self.test_lr_path = test_lr_path
        self.test_hr_path = test_hr_path
        test_extension_list = [os.path.splitext(e)[1] for e in os.listdir(self.test_hr_path)]
        test_extension = max(set(test_extension_list), key = test_extension_list.count)
        self.test_filenames = [x for x in os.listdir(self.test_hr_path) if x.endswith(test_extension)]

        self.crappifier_method = crappifier_method
        self.scale_factor = scale_factor
        self.num_patches = num_patches
        self.lr_patch_size_x = patch_size_x     
        self.lr_patch_size_y = patch_size_y
        
        self.validation_split = validation_split
        if 'rotation' in data_augmentation:
            self.rotation = True
        if 'horizontal_flip' in data_augmentation:
            self.horizontal_flip = True
        if 'vertical_flip' in data_augmentation:
            self.vertical_flip = True
        if len(data_augmentation) != 0 and (not self.rotation or not self.horizontal_flip or not self.vertical_flip):
            raise ValueError('Data augmentation values are not well defined.')

        self.model_name = model_name
        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        self.optimizer_name = optimizer_name
        self.discriminator_optimizer = discriminator_optimizer
        self.lr_scheduler_name = lr_scheduler_name
        self.discriminator_lr_scheduler = discriminator_lr_scheduler

        self.model_configuration = model_configuration
        
        self.test_metric_indexes = test_metric_indexes
        self.additional_folder = additional_folder
        self.seed = seed
        
        self.verbose = verbose
        
        utils.set_seed(self.seed)
        
        save_folder = 'scale' + str(self.scale_factor)

        if self.additional_folder:
            save_folder += '_' + self.additional_folder

        self.saving_path = './results/{}/{}/{}/scale{}_epc{}_btch{}_lr{}_optim-{}_lrsched-{}_seed{}'.format(
                                                                              self.data_name, 
                                                                              self.model_name,
                                                                              save_folder, 
                                                                              self.scale_factor, 
                                                                              self.number_of_epochs,
                                                                              self.batch_size, 
                                                                              self.learning_rate, 
                                                                              self.optimizer_name,
                                                                              self.lr_scheduler_name,
                                                                              self.seed)
        
        os.makedirs(self.saving_path, exist_ok=True)
    
    def launch(self):
        self.prepare_data()                     
        self.train_model()
        self.predict_images()
        self.eval_model()
        
        return self.history
    
    def prepare_data(self):                  
        raise NotImplementedError('prepare_data() not implemented.')          
            
    def train_model(self):
        raise NotImplementedError('train_model() not implemented.')
        
    def predict_images(self):
        raise NotImplementedError('predict_images() not implemented')
        
    def eval_model(self):
    	
        if self.verbose:
            utils.print_info('eval_model() - self.Y_test', self.Y_test)
            utils.print_info('eval_model() - self.predictions', self.predictions)

        print('The predictions will be evaluated')
        metrics_dict = metrics.obtain_metrics(gt_image_list=self.Y_test, predicted_image_list=self.predictions, 
                                        test_metric_indexes=self.test_metric_indexes)
            
        os.makedirs(self.saving_path + '/test_metrics', exist_ok=True)
        
        for key in metrics_dict.keys():
            print('{}: {}'.format(key, np.mean(metrics_dict[key])))
            np.save(self.saving_path + '/test_metrics/' + key + '.npy', metrics_dict[key])
            

class TensorflowTrainer(ModelsTrainer):
    
    def __init__(self, data_name, 
                 train_lr_path, train_hr_path, 
                 val_lr_path, val_hr_path,
                 test_lr_path, test_hr_path,
                 crappifier_method, model_name, scale_factor, 
                 number_of_epochs, batch_size, 
                 learning_rate, discriminator_learning_rate, 
                 optimizer_name, lr_scheduler_name, 
                 test_metric_indexes, additional_folder, 
                 model_configuration, seed,
                 num_patches, patch_size_x, patch_size_y, 
                 validation_split, data_augmentation,
                 discriminator_optimizer=None, 
                 discriminator_lr_scheduler=None,
                 verbose=0
                ):
        
        super().__init__(data_name, 
                 train_lr_path, train_hr_path, 
                 val_lr_path, val_hr_path,
                 test_lr_path, test_hr_path,
                 crappifier_method, model_name, scale_factor, 
                 number_of_epochs, batch_size, 
                 learning_rate, discriminator_learning_rate, 
                 optimizer_name, lr_scheduler_name, 
                 test_metric_indexes, additional_folder, 
                 model_configuration, seed,
                 num_patches, patch_size_x, patch_size_y, 
                 validation_split, data_augmentation,
                 discriminator_optimizer=discriminator_optimizer, 
                 discriminator_lr_scheduler=discriminator_lr_scheduler,
                 verbose=verbose
                )
    
        self.library_name ='tensorflow'
    
    def prepare_data(self):

        train_patches_wf, train_patches_gt = datasets.extract_random_patches_from_folder(
                                                hr_data_path=self.train_hr_path, 
                                                lr_data_path=self.train_lr_path, 
                                                filenames=self.train_filenames, 
                                                scale_factor=self.scale_factor, 
                                                crappifier_name=self.crappifier_method, 
                                                lr_patch_shape=(self.lr_patch_size_x, self.lr_patch_size_y), 
                                                num_patches=self.num_patches)

        X_train = np.array([np.expand_dims(x, axis=-1) for x in train_patches_wf])
        Y_train = np.array([np.expand_dims(x, axis=-1) for x in train_patches_gt])
           
        if self.verbose:
            print('Data:')
            print('HR - shape:{} max:{} min:{} mean:{} dtype:{}'.format(Y_train.shape, np.max(Y_train), np.min(Y_train),  np.mean(Y_train), Y_train.dtype))
            print('LR - shape:{} max:{} min:{} mean:{} dtype:{}'.format(X_train.shape, np.max(X_train), np.min(X_train),  np.mean(X_train), X_train.dtype))
            plt.figure(figsize=(10,5))
            plt.subplot(1,2,1)
            plt.imshow(Y_train[0])
            plt.title('Y_train - gt')
            plt.subplot(1,2,2)
            plt.imshow(X_train[0])
            plt.title('X_train - wf')
            plt.show()

        assert np.max(X_train[0]) <= 1.0 and np.max(Y_train[0]) <= 1.0
        assert np.min(X_train[0]) >= 0.0 and np.min(Y_train[0]) >= 0.0            
        assert len(X_train.shape) == 4 and len(Y_train.shape) == 4

        if self.model_configuration['others']['positional_encoding']:
            X_train = utils.concatenate_encoding(X_train, self.model_configuration['others']['positional_encoding_channels'])

        if self.val_hr_path is not None or self.val_lr_path is not None:

            val_patches_wf, val_patches_gt = datasets.extract_random_patches_from_folder(
                                                hr_data_path=self.val_hr_path, 
                                                lr_data_path=self.val_lr_path, 
                                                filenames=self.val_filenames, 
                                                scale_factor=self.scale_factor, 
                                                crappifier_name=self.crappifier_method, 
                                                lr_patch_shape=(self.lr_patch_size_x, self.lr_patch_size_y), 
                                                num_patches=self.num_patches)            

            X_val = np.array([np.expand_dims(x, axis=-1) for x in val_patches_wf])
            Y_val = np.array([np.expand_dims(x, axis=-1) for x in val_patches_gt])

            if self.model_configuration['others']['positional_encoding']:
                X_val = utils.concatenate_encoding(X_val, self.model_configuration['others']['positional_encoding_channels'])

            assert np.max(Y_val[0]) <= 1.0 and np.max(X_val[0]) <= 1.0
            assert np.min(Y_val[0]) >= 0.0 and np.min(X_val[0]) >= 0.0            
            assert len(X_val.shape) == 4 and len(Y_val.shape) == 4
        
        self.input_data_shape = X_train.shape
        self.output_data_shape = Y_train.shape

        if self.val_hr_path is None or self.val_lr_path is None:
            print('Both train and validation data will be extracted from the received train path.')
            train_generator, val_generator = datasets.get_train_val_generators(X_data=X_train,
                                                                      Y_data=Y_train,
                                                                      validation_split=self.validation_split,
                                                                      batch_size=self.batch_size,
                                                                      show_examples=False,
                                                                      rotation=self.rotation,
                                                                      horizontal_flip=self.horizontal_flip,
                                                                      vertical_flip=self.vertical_flip)
        else:
            print('There are different train and validation paths.')
            train_generator = datasets.get_generator(X_data=X_train,
                                            Y_data=Y_train,
                                            batch_size=self.batch_size,
                                            show_examples=False,
                                            rotation=self.rotation,
                                            horizontal_flip=self.horizontal_flip,
                                            vertical_flip=self.vertical_flip)
            
            val_generator = datasets.get_generator(X_data=X_val,
                                          Y_data=Y_val,
                                          batch_size=self.batch_size,
                                          show_examples=False,
                                          rotation=self.rotation,
                                          horizontal_flip=self.horizontal_flip,
                                          vertical_flip=self.vertical_flip)

        self.train_generator=train_generator
        self.val_generator=val_generator

    def train_model(self):


        self.optim = optimizer_scheduler_utils.select_optimizer(library_name=self.library_name, optimizer_name=self.optimizer_name, 
                                      learning_rate=self.learning_rate, check_point=None,
                                      parameters=None, additional_configuration=self.model_configuration)
            
        model = model_utils.select_model(model_name=self.model_name, input_shape=self.input_data_shape, output_channels=self.output_data_shape[-1], 
                             scale_factor=self.scale_factor, model_configuration=self.model_configuration)
        
        loss_funct = 'mean_absolute_error'
        eval_metric = 'mean_squared_error'
        
        model.compile(optimizer=self.optim, loss=loss_funct, metrics=[eval_metric, utils.ssim_loss])
        
        trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
        nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
        totalParams = trainableParams + nonTrainableParams
	
        if self.verbose:
            print('Trainable parameteres: {} \nNon trainable parameters: {} \nTotal parameters: {}'.format(trainableParams, 
                                                                                                            nonTrainableParams, 
                                                                                                        totalParams))
    

        lr_schedule = optimizer_scheduler_utils.select_lr_schedule(library_name=self.library_name, lr_scheduler_name=self.lr_scheduler_name, 
                                                                    data_len=self.input_data_shape[0]//self.batch_size, 
                                                                    number_of_epochs=self.number_of_epochs, learning_rate=self.learning_rate,
                                                                    monitor_loss=None, name=None, optimizer=None, frequency=None,
                                                                    additional_configuration=self.model_configuration)
        
        model_checkpoint = tf_ModelCheckpoint(os.path.join(self.saving_path, 'weights_best.h5'), 
                                       monitor='val_loss',verbose=1, 
                                       save_best_only=True, save_weights_only=True)
            
        # callback for early stopping
        earlystopper = EarlyStopping(monitor=self.model_configuration['optim']['early_stop']['loss'],
        			     patience=self.model_configuration['optim']['early_stop']['patience'], 
                                     min_delta=0.005, mode=self.model_configuration['optim']['early_stop']['mode'],
                                     verbose=1, restore_best_weights=True)
        
        start = time.time()
        
        history = model.fit(self.train_generator, validation_data=self.val_generator,
                          validation_steps=np.ceil(self.input_data_shape[0]*0.1/self.batch_size),
                          steps_per_epoch=np.ceil(self.input_data_shape[0]/self.batch_size),
                          epochs=self.number_of_epochs, 
                          callbacks=[lr_schedule, model_checkpoint, earlystopper])
        
        dt = time.time() - start
        mins, sec = divmod(dt, 60) 
        hour, mins = divmod(mins, 60) 
        print("\nTime elapsed:",hour, "hour(s)",mins,"min(s)",round(sec),"sec(s)\n")
        
        model.save_weights(os.path.join(self.saving_path, 'weights_last.h5'))
        self.history = history
        
        os.makedirs(self.saving_path + '/train_metrics', exist_ok=True)
                
        for key in history.history:
            np.save(self.saving_path + '/train_metrics/' + key + '.npy', history.history[key])
        np.save(self.saving_path + '/train_metrics/time.npy', np.array([dt]))
        

    def predict_images(self):

        lr_images, hr_images = datasets.extract_random_patches_from_folder(
                                        hr_data_path=self.test_hr_path, 
                                        lr_data_path=self.test_lr_path, 
                                        filenames=self.test_filenames, 
                                        scale_factor=self.scale_factor, 
                                        crappifier_name=self.crappifier_method, 
                                        lr_patch_shape=None, 
                                        num_patches=1)
    
        hr_images = np.expand_dims(hr_images, axis=-1)
        lr_images = np.expand_dims(lr_images, axis=-1)

        if self.verbose:
            print('HR images - shape:{} dtype:{}'.format(hr_images.shape, hr_images.dtype))
            print('LR images - shape:{} dtype:{}'.format(lr_images.shape, lr_images.dtype))
        
        if self.model_configuration['others']['positional_encoding']:
            lr_images = utils.concatenate_encoding(lr_images, self.model_configuration['others']['positional_encoding_channels'])
            
        optim = optimizer_scheduler_utils.select_optimizer(library_name=self.library_name, optimizer_name=self.optimizer_name, 
                                 learning_rate=self.learning_rate, check_point=None,
                                 parameters=None, additional_configuration=self.model_configuration)

        model = model_utils.select_model(model_name=self.model_name, input_shape=lr_images.shape, output_channels=hr_images.shape[-1],
                             scale_factor=self.scale_factor, model_configuration=self.model_configuration)
        
        loss_funct = 'mean_absolute_error'
        eval_metric = 'mean_squared_error'
        
        model.compile(optimizer=optim, loss=loss_funct, metrics=[eval_metric, utils.ssim_loss])
            
        # Load old weights
        model.load_weights( os.path.join(self.saving_path, 'weights_best.h5') )   
        
        predictions = model.predict(lr_images, batch_size=1)
    
        self.Y_test = hr_images
        self.predictions = np.clip(predictions, a_min=0, a_max=1)
        
        assert np.max(self.Y_test[0]) <= 1.0 and np.max(self.predictions[0]) <= 1.0
        assert np.min(self.Y_test[0]) >= 0.0 and np.min(self.predictions[0]) >= 0.0

        if self.verbose:
            utils.print_info('predict_images() - Y_test', self.Y_test)
            utils.print_info('predict_images() - predictions', self.predictions)

        # Save the predictions
        os.makedirs(self.saving_path + '/predicted_images', exist_ok=True)
                
        for i, image  in enumerate(predictions):
          tf.keras.preprocessing.image.save_img(self.saving_path+'/predicted_images/'+self.test_filenames[i], image, 
                                                data_format=None, file_format=None)
        print('Predicted images have been saved in: ' + self.saving_path + '/predicted_images')
        
            

class PytorchTrainer(ModelsTrainer):
    def __init__(self, data_name, 
                 train_lr_path, train_hr_path, 
                 val_lr_path, val_hr_path,
                 test_lr_path, test_hr_path,
                 crappifier_method, model_name, scale_factor, 
                 number_of_epochs, batch_size, 
                 learning_rate, discriminator_learning_rate, 
                 optimizer_name, lr_scheduler_name, 
                 test_metric_indexes, additional_folder, 
                 model_configuration, seed,
                 num_patches, patch_size_x, patch_size_y, 
                 validation_split, data_augmentation,
                 discriminator_optimizer=None, 
                 discriminator_lr_scheduler=None,
                 verbose=0, gpu_id=0
                ):
        
        super().__init__(data_name, 
                 train_lr_path, train_hr_path, 
                 val_lr_path, val_hr_path,
                 test_lr_path, test_hr_path,
                 crappifier_method, model_name, scale_factor, 
                 number_of_epochs, batch_size, 
                 learning_rate, discriminator_learning_rate, 
                 optimizer_name, lr_scheduler_name, 
                 test_metric_indexes, additional_folder, 
                 model_configuration, seed,
                 num_patches, patch_size_x, patch_size_y, 
                 validation_split, data_augmentation,
                 discriminator_optimizer=discriminator_optimizer, 
                 discriminator_lr_scheduler=discriminator_lr_scheduler,
                 verbose=verbose
                )
        
        self.gpu_id = gpu_id
        
        self.library_name ='pytorch'
        
    def prepare_data(self):                  
        pass 

    def train_model(self):

        model = model_utils.select_model(model_name=self.model_name, input_shape=None, output_channels=None,
                             scale_factor=self.scale_factor, batch_size=self.batch_size, 
                             lr_patch_size_x=self.lr_patch_size_x, lr_patch_size_y=self.lr_patch_size_y,
                             learning_rate_g=self.learning_rate, learning_rate_d=self.discriminator_learning_rate,
                             g_optimizer = self.optimizer_name, d_optimizer = self.discriminator_optimizer, 
                             g_scheduler = self.lr_scheduler_name, d_scheduler = self.discriminator_lr_scheduler,
                             epochs = self.number_of_epochs, save_basedir = self.saving_path, 
                             train_hr_path=None, train_lr_path=None, train_filenames = None,
                             val_hr_path=None, val_lr_path=None, val_filenames=None, crappifier_method=None,
                             model_configuration=self.model_configuration)
        
        if self.verbose:
            data = iter(model.train_dataloader()).next()

            print('LR patch shape: {}'.format(data['lr'][0][0].shape))
            print('HR patch shape: {}'.format(data['hr'][0][0].shape))
    
            utils.print_info('train_model() - lr', data['lr'])
            utils.print_info('train_model() - hr', data['hr'])

        logger = CSVLogger(self.saving_path + '/Quality Control', name='Logger')
    
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        checkpoints = ModelCheckpoint(monitor='val_ssim', mode='max', save_top_k=3, 
                                        every_n_train_steps=5, save_last=True, 
                                        filename="{epoch:02d}-{val_ssim:.3f}")

        trainer = Trainer(
            gpus=1, 
            max_epochs=self.number_of_epochs, 
            logger=logger, 
            callbacks=[checkpoints, lr_monitor]
        )

        start = time.time()
    
        trainer.fit(model)
        
        # Displaying the time elapsed for training
        dt = time.time() - start
        mins, sec = divmod(dt, 60) 
        hour, mins = divmod(mins, 60) 
        print("\nTime elapsed:",hour, "hour(s)",mins,"min(s)",round(sec),"sec(s)\n")
            
        logger_path = os.path.join(self.saving_path + '/Quality Control/Logger')
        all_logger_versions = [os.path.join(logger_path, dname) for dname in os.listdir(logger_path)]
        last_logger = all_logger_versions[-1]
    
        train_csv_path = last_logger + '/metrics.csv'

    
        if not os.path.exists(train_csv_path):
            print('The path does not contain a csv file containing the loss and validation evolution of the model')
        else:
            with open(train_csv_path,'r') as csvfile:
                csvRead = csv.reader(csvfile, delimiter=',')
                keys = next(csvRead)
                keys.remove('step') 

                if self.model_name == 'wgan':
                    train_metrics = {'g_lr':[], 'd_lr':[],
                                'g_loss_step':[], 'g_l1_step':[], 'g_adv_loss_step':[],
                                'd_real_step':[], 'd_fake_step':[],
                                'd_loss_step':[], 'd_wasserstein_step':[], 'd_gp_step':[],
                                'epoch':[],
                                'val_ssim':[], 'val_psnr':[],
                                'val_g_loss':[], 'val_g_l1':[],
                                'val_d_wasserstein':[],
                                'g_loss_epoch':[], 'g_l1_epoch':[], 'g_adv_loss_epoch':[],
                                'd_real_epoch':[], 'd_fake_epoch':[],
                                'd_loss_epoch':[], 'd_wasserstein_epoch':[], 'd_gp_epoch':[]
                                }
                elif self.model_name == 'esrganplus':
                    train_metrics = {'g_lr':[], 'd_lr':[],
                                 'ssim_step':[], 'g_loss_step':[], 'g_pixel_loss_step':[], 
                                'g_features_loss_step':[], 'g_adversarial_loss_step':[],
                                'd_loss_step':[], 'd_real_step':[], 'd_fake_step':[],
                                'epoch':[],
                                'val_ssim':[], 'val_psnr':[],
                                'val_g_loss':[], 'val_g_pixel_loss':[],
                                'val_g_features_loss':[], 'val_g_adversarial_loss':[],
                                 'ssim_epoch':[], 'g_loss_epoch':[], 'g_pixel_loss_epoch':[], 
                                'g_features_loss_epoch':[], 'g_adversarial_loss_epoch':[],
                                'd_loss_epoch':[], 'd_real_epoch':[], 'd_fake_epoch':[]
                                }

                for row in csvRead:
                    step = int(row[2])
                    row.pop(2)
                    for i, row_value in enumerate(row):
                        if row_value:
                            train_metrics[keys[i]].append([step, float(row_value)])

        os.makedirs(self.saving_path + '/train_metrics', exist_ok=True)
    
        for key in train_metrics:
            values_to_save = np.array([e[1] for e in train_metrics[key]])
            np.save(self.saving_path + '/train_metrics/' + key + '.npy', values_to_save)
        np.save(self.saving_path + '/train_metrics/time.npy', np.array([dt]))
        

        self.history = []
        print('Train information saved.')
        
    def predict_images(self):
        
        hr_images = np.array([io.imread(os.path.join(self.test_hr_path, fil)) for fil in self.test_filenames])
    
        model = model_utils.select_model(model_name=self.model_name, scale_factor=self.scale_factor, batch_size=self.batch_size, 
                             save_basedir = self.saving_path, model_configuration=self.model_configuration, 
                             checkpoint=os.path.join(self.saving_path,'Quality Control','best_checkpoint.pth'))

        trainer = Trainer(gpus=1)

        dataset = datasets.PytorchDataset(hr_data_path=self.test_hr_path,
                                 lr_data_path=self.test_lr_path, 
                                 filenames=self.test_filenames, 
                                 scale_factor=self.scale_factor, 
                                 crappifier_name=self.crappifier_method, 
                                 lr_patch_shape=(self.lr_patch_size_x, self.lr_patch_size_y), 
                                 num_patches=self.num_patches, 
                                 transformations=datasets.ToTensor())

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        data = iter(dataloader).next()
        predictions = trainer.predict(model, dataloaders=dataloader)
        predictions = np.array([np.expand_dims(np.squeeze(e.detach().numpy()),axis=-1) for e in predictions])
        
        if self.verbose:
            utils.print_info('predict_images() - lr', data['lr'])
            utils.print_info('predict_images() - hr', data['hr'])
            utils.print_info('predict_images() - predictions', predictions)
        
        os.makedirs(os.path.join(self.saving_path, 'predicted_images'), exist_ok=True)
                
        for i, image  in enumerate(predictions):
            tf.keras.preprocessing.image.save_img(self.saving_path+'/predicted_images/'+self.test_filenames[i], 
                                                  image, data_format=None, file_format=None)
        print('Predicted images have been saved in: ' + self.saving_path + '/predicted_images')
        
        self.Y_test = np.expand_dims(hr_images, axis=-1)
        self.predictions = predictions

        if self.verbose:
            utils.print_info('predict_images() - self.Y_test', self.Y_test)
            utils.print_info('predict_images() - self.predictions', self.predictions)
                
        print('True HR shape: {}'.format(self.Y_test.shape))
        print('Predicted HR shape: {}'.format(self.predictions.shape))
        
        print('True HR: max={} min={}'.format(np.max(self.Y_test[0]), np.min(self.Y_test[0])))
        print('Predicted HR: max={} min={}'.format(np.max(self.predictions[0]), np.min(self.predictions[0])))
                  
        assert np.max(self.Y_test[0]) <= 1.0 and np.max(self.predictions[0]) <= 1.0
        assert np.min(self.Y_test[0]) >= 0.0 and np.min(self.predictions[0]) >= 0.0
    
 
def train_configuration(data_name, 
                 train_lr_path, train_hr_path, 
                 val_lr_path, val_hr_path,
                 test_lr_path, test_hr_path,
                 crappifier_method, model_name, scale_factor, 
                 number_of_epochs, batch_size, 
                 learning_rate, discriminator_learning_rate, 
                 optimizer_name, lr_scheduler_name, 
                 test_metric_indexes, additional_folder, 
                 model_configuration, seed,
                 num_patches, patch_size_x, patch_size_y, 
                 validation_split, data_augmentation,
                 discriminator_optimizer=None, 
                 discriminator_lr_scheduler=None,
                 verbose=0, gpu_id=0
                ):
    
    if model_name in ['wgan', 'esrganplus']:
        model_trainer = PytorchTrainer(data_name, 
                 train_lr_path, train_hr_path, 
                 val_lr_path, val_hr_path,
                 test_lr_path, test_hr_path,
                 crappifier_method, model_name, scale_factor, 
                 number_of_epochs, batch_size, 
                 learning_rate, discriminator_learning_rate, 
                 optimizer_name, lr_scheduler_name, 
                 test_metric_indexes, additional_folder, 
                 model_configuration, seed,
                 num_patches, patch_size_x, patch_size_y, 
                 validation_split, data_augmentation,
                 discriminator_optimizer=discriminator_optimizer, 
                 discriminator_lr_scheduler=discriminator_lr_scheduler,
                 verbose=verbose, gpu_id=gpu_id
                )
    elif model_name in ['rcan', 'dfcan', 'wdsr', 'unet']:
        model_trainer = TensorflowTrainer(data_name, 
                 train_lr_path, train_hr_path, 
                 val_lr_path, val_hr_path,
                 test_lr_path, test_hr_path,
                 crappifier_method, model_name, scale_factor, 
                 number_of_epochs, batch_size, 
                 learning_rate, discriminator_learning_rate, 
                 optimizer_name, lr_scheduler_name, 
                 test_metric_indexes, additional_folder, 
                 model_configuration, seed,
                 num_patches, patch_size_x, patch_size_y, 
                 validation_split, data_augmentation,
                 discriminator_optimizer=discriminator_optimizer, 
                 discriminator_lr_scheduler=discriminator_lr_scheduler,
                 verbose=verbose, 
                )
    else:
        raise Exception("Not available model.") 
        
    return model_trainer.launch()
