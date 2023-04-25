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
                 datagen_sampling_pdf, 
                 train_config,
                 discriminator_optimizer=None, 
                 discriminator_lr_scheduler=None,
                 verbose=0
                ):

        self.data_name = data_name

        self.train_lr_path = train_lr_path
        self.train_hr_path = train_hr_path
        train_extension_list = [os.path.splitext(e)[1] for e in os.listdir(self.train_hr_path)]
        train_extension = max(set(train_extension_list), key = train_extension_list.count)
        self.train_filenames = sorted([x for x in os.listdir(self.train_hr_path) if x.endswith(train_extension)])

        if val_hr_path is None or val_lr_path is None:
            self.val_lr_path = train_lr_path
            self.val_hr_path = train_hr_path

            self.val_filenames = self.train_filenames[int(len(self.train_filenames)*(1 - validation_split)):]
            self.train_filenames = self.train_filenames[:int(len(self.train_filenames)*(1 - validation_split))]
        else:
            self.val_lr_path = val_lr_path
            self.val_hr_path = val_hr_path

            val_extension_list = [os.path.splitext(e)[1] for e in os.listdir(self.val_hr_path)]
            val_extension = max(set(val_extension_list), key = val_extension_list.count)
            self.val_filenames = sorted([x for x in os.listdir(self.val_hr_path) if x.endswith(val_extension)])

        self.test_lr_path = test_lr_path
        self.test_hr_path = test_hr_path
        test_extension_list = [os.path.splitext(e)[1] for e in os.listdir(self.test_hr_path)]
        test_extension = max(set(test_extension_list), key = test_extension_list.count)
        self.test_filenames = sorted([x for x in os.listdir(self.test_hr_path) if x.endswith(test_extension)])

        self.crappifier_method = crappifier_method
        self.scale_factor = scale_factor
        self.num_patches = num_patches
        self.lr_patch_size_x = patch_size_x     
        self.lr_patch_size_y = patch_size_y
        self.datagen_sampling_pdf = datagen_sampling_pdf
        
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

        print('\n' + '-'*10)
        print('{} model will be trained with the next configuration'.format(self.model_name))
        print('Dataset: {}'.format(self.data_name))
        print('\tTrain wf path: {}'.format(train_lr_path))
        print('\tTrain gt path: {}'.format(train_hr_path))
        print('\tVal wf path: {}'.format(val_lr_path))
        print('\tVal gt path: {}'.format(val_hr_path))
        print('\tTest wf path: {}'.format(test_lr_path))
        print('\tTest gt path: {}'.format(test_hr_path))
        print('Preprocessing info:')
        print('\tScale factor: {}'.format(self.scale_factor))
        print('\tCrappifier method: {}'.format(crappifier_method))
        print('\tNum patches: {}'.format(num_patches))
        print('\tPatch size: {} x {}'.format(patch_size_x, patch_size_y))
        print('Training info:')
        print('\tEpochs: {}'.format(number_of_epochs))
        print('\tBatchsize: {}'.format(batch_size))
        print('\tGen learning rate: {}'.format(learning_rate))
        print('\tDisc learning rate: {}'.format(discriminator_learning_rate))
        print('\tGen optimizer: {}'.format(optimizer_name))
        print('\tDisc optimizer: {}'.format(discriminator_optimizer))
        print('\tGen scheduler: {}'.format(lr_scheduler_name))
        print('\tDisc scheduler: {}'.format(discriminator_lr_scheduler))
        print('-'*10)

        os.makedirs(self.saving_path, exist_ok=True)

        utils.save_yaml(train_config, os.path.join(self.saving_path, 'train_configuration.yaml'))
    
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

        print('The predictions will be evaluated:')
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
                 datagen_sampling_pdf,
                 train_config,
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
                 datagen_sampling_pdf,
                 train_config,
                 discriminator_optimizer=discriminator_optimizer, 
                 discriminator_lr_scheduler=discriminator_lr_scheduler,
                 verbose=verbose
                )
    
        self.library_name ='tensorflow'
    
    def prepare_data(self):
        
        train_generator = datasets.DataGenerator(filenames=self.train_filenames, hr_data_path=self.train_hr_path, 
                                                 lr_data_path=self.train_hr_path, scale_factor=self.scale_factor, 
                                                 crappifier_name=self.crappifier_method, 
                                                 lr_patch_shape=(self.lr_patch_size_x, self.lr_patch_size_y),
                                                 num_patches=self.num_patches, datagen_sampling_pdf=self.datagen_sampling_pdf, 
                                                 validation_split=0.1, batch_size=self.batch_size, 
                                                 rotation=self.rotation, horizontal_flip=self.horizontal_flip, vertical_flip=self.vertical_flip, 
                                                 module='train', shuffle=True)
        val_generator = datasets.DataGenerator(filenames=self.val_filenames, hr_data_path=self.val_hr_path, 
                                                 lr_data_path=self.val_hr_path, scale_factor=self.scale_factor, 
                                                 crappifier_name=self.crappifier_method, 
                                                 lr_patch_shape=(self.lr_patch_size_x, self.lr_patch_size_y),
                                                 num_patches=self.num_patches, datagen_sampling_pdf=self.datagen_sampling_pdf, 
                                                 validation_split=0.1, batch_size=self.batch_size, 
                                                 rotation=self.rotation, horizontal_flip=self.horizontal_flip, vertical_flip=self.vertical_flip, 
                                                 module='train', shuffle=True)
        
        x_sample, y_sample, actual_scale_factor = train_generator.get_sample(0)
        self.input_data_shape = (x_sample.shape[0]*train_generator.__len__(),) + (x_sample.shape[1:])
        self.output_data_shape = (y_sample.shape[0]*train_generator.__len__(),) + (y_sample.shape[1:])

        if self.scale_factor is None or self.scale_factor != actual_scale_factor:
            self.scale_factor = actual_scale_factor
            utils.update_yaml(os.path.join(self.saving_path, 'train_configuration.yaml'), 
                              'actual_scale_factor', actual_scale_factor)
            if self.verbose:
                print('Actual scale factor that will be used is: {}'.format(self.scale_factor))
           
        if self.verbose:
            print('Data:')
            print('HR - shape:{} max:{} min:{} mean:{} dtype:{}'.format(self.output_data_shape, np.max(y_sample), np.min(y_sample),  np.mean(y_sample), y_sample.dtype))
            print('LR - shape:{} max:{} min:{} mean:{} dtype:{}'.format(self.input_data_shape, np.max(x_sample), np.min(x_sample),  np.mean(x_sample), x_sample.dtype))
            '''
            plt.figure(figsize=(10,5))
            plt.subplot(1,2,1)
            plt.imshow(Y_train[0])
            plt.title('Y_train - gt')
            plt.subplot(1,2,2)
            plt.imshow(X_train[0])
            plt.title('X_train - wf')
            plt.show()
            '''

        assert np.max(x_sample[0]) <= 1.0 and np.max(y_sample[0]) <= 1.0
        assert np.min(x_sample[0]) >= 0.0 and np.min(y_sample[0]) >= 0.0            
        assert len(x_sample.shape) == 4 and len(y_sample.shape) == 4
        
        utils.update_yaml(os.path.join(self.saving_path, 'train_configuration.yaml'), 
                                'input_data_shape', self.input_data_shape)
        utils.update_yaml(os.path.join(self.saving_path, 'train_configuration.yaml'), 
                            'output_data_shape', self.output_data_shape)

        self.train_generator=train_generator
        self.val_generator=val_generator

    def train_model(self):


        self.optim = optimizer_scheduler_utils.select_optimizer(library_name=self.library_name, optimizer_name=self.optimizer_name, 
                                      learning_rate=self.learning_rate, check_point=None,
                                      parameters=None, additional_configuration=self.model_configuration)
            
        model = model_utils.select_model(model_name=self.model_name, input_shape=self.input_data_shape, output_channels=self.output_data_shape[-1], 
                             scale_factor=self.scale_factor, datagen_sampling_pdf=self.datagen_sampling_pdf, model_configuration=self.model_configuration)
        
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
        
        print('Training is going to start:')
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

        predictions = []
        print('Prediction is going to start:')
        for test_filename in self.test_filenames:
            lr_images, hr_images, _ = datasets.extract_random_patches_from_folder(
                                            hr_data_path=self.test_hr_path, 
                                            lr_data_path=self.test_lr_path, 
                                            filenames=[test_filename], 
                                            scale_factor=self.scale_factor, 
                                            crappifier_name=self.crappifier_method, 
                                            lr_patch_shape=None, 
                                            num_patches=1,
                                            datagen_sampling_pdf=1)
    
            hr_images = np.expand_dims(hr_images, axis=-1)
            lr_images = np.expand_dims(lr_images, axis=-1)

            if self.model_name == 'unet':
                if self.verbose:
                    print('Padding will be added to the images.')
                    print('LR images before padding:')
                    print('LR images - shape:{} dtype:{}'.format(lr_images.shape, lr_images.dtype))

                height_padding, width_padding = utils.calculate_pad_for_Unet(lr_img_shape = lr_images[0].shape, 
                                                                            depth_Unet = self.model_configuration['unet']['depth'], 
                                                                            is_pre = True, 
                                                                            scale = self.scale_factor)
                
                if self.verbose and (height_padding == (0,0) and width_padding == (0,0)):
                    print('No padding has been needed to be added.')
                print(height_padding)
                print(width_padding)

                lr_images = utils.add_padding_for_Unet(lr_imgs = lr_images, 
                                                    height_padding = height_padding, 
                                                    width_padding = width_padding)

            if self.verbose:
                print('HR images - shape:{} dtype:{}'.format(hr_images.shape, hr_images.dtype))
                print('LR images - shape:{} dtype:{}'.format(lr_images.shape, lr_images.dtype))
            
            if self.model_configuration['others']['positional_encoding']:
                lr_images = utils.concatenate_encoding(lr_images, self.model_configuration['others']['positional_encoding_channels'])
                
            optim = optimizer_scheduler_utils.select_optimizer(library_name=self.library_name, optimizer_name=self.optimizer_name, 
                                    learning_rate=self.learning_rate, check_point=None,
                                    parameters=None, additional_configuration=self.model_configuration)

            model = model_utils.select_model(model_name=self.model_name, input_shape=lr_images.shape, output_channels=hr_images.shape[-1],
                                scale_factor=self.scale_factor, datagen_sampling_pdf=self.datagen_sampling_pdf, model_configuration=self.model_configuration)
            
            loss_funct = 'mean_absolute_error'
            eval_metric = 'mean_squared_error'
            
            model.compile(optimizer=optim, loss=loss_funct, metrics=[eval_metric, utils.ssim_loss])
                
            # Load old weights
            model.load_weights( os.path.join(self.saving_path, 'weights_best.h5') )   

            aux_prediction = model.predict(lr_images, batch_size=1)

            if self.model_name == 'unet':
                aux_prediction = utils.remove_padding_for_Unet(pad_hr_imgs = aux_prediction, 
                                                            height_padding = height_padding, 
                                                            width_padding = width_padding, 
                                                            scale = self.scale_factor)
            
            aux_prediction = np.clip(aux_prediction, a_min=0, a_max=1)

            predictions.append(aux_prediction)

        self.Y_test = hr_images
        self.predictions = predictions
        
        assert np.max(self.Y_test[0]) <= 1.0 and np.max(self.predictions[0]) <= 1.0
        assert np.min(self.Y_test[0]) >= 0.0 and np.min(self.predictions[0]) >= 0.0

        if self.verbose:
            utils.print_info('predict_images() - Y_test', self.Y_test)
            utils.print_info('predict_images() - predictions', self.predictions)

        # Save the predictions
        os.makedirs(self.saving_path + '/predicted_images', exist_ok=True)
                
        for i, image  in enumerate(predictions):
          tf.keras.preprocessing.image.save_img(self.saving_path+'/predicted_images/'+self.test_filenames[i], image[0,...], 
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
                 datagen_sampling_pdf,
                 train_config,
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
                 datagen_sampling_pdf,
                 train_config,
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
                             scale_factor=self.scale_factor, batch_size=self.batch_size, num_patches=self.num_patches,
                             lr_patch_size_x=self.lr_patch_size_x, lr_patch_size_y=self.lr_patch_size_y,
                             datagen_sampling_pdf=self.datagen_sampling_pdf,
                             learning_rate_g=self.learning_rate, learning_rate_d=self.discriminator_learning_rate,
                             g_optimizer = self.optimizer_name, d_optimizer = self.discriminator_optimizer, 
                             g_scheduler = self.lr_scheduler_name, d_scheduler = self.discriminator_lr_scheduler,
                             epochs = self.number_of_epochs, save_basedir = self.saving_path, 
                             train_hr_path=self.train_hr_path, train_lr_path=self.train_lr_path, train_filenames = self.train_filenames,
                             val_hr_path=self.val_hr_path, val_lr_path=self.val_lr_path, val_filenames=self.val_filenames, 
                             crappifier_method=self.crappifier_method, model_configuration=self.model_configuration)
        
        if self.verbose:
            data = next(iter(model.train_dataloader()))

            print('LR patch shape: {}'.format(data['lr'][0][0].shape))
            print('HR patch shape: {}'.format(data['hr'][0][0].shape))
    
            utils.print_info('train_model() - lr', data['lr'])
            utils.print_info('train_model() - hr', data['hr'])

        os.makedirs(self.saving_path + '/Quality Control', exist_ok=True)
        logger = CSVLogger(self.saving_path + '/Quality Control', name='Logger')
    
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        checkpoints = ModelCheckpoint(monitor='val_ssim', mode='max', save_top_k=3, 
                                        every_n_train_steps=5, save_last=True, 
                                        filename="{epoch:02d}-{val_ssim:.3f}")

        trainer = Trainer(accelerator="gpu", devices=1,
            max_epochs=self.number_of_epochs, 
            logger=logger, 
            callbacks=[checkpoints, lr_monitor]
        )

        print('Training is going to start:')

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
                             datagen_sampling_pdf=self.datagen_sampling_pdf,
                             checkpoint=os.path.join(self.saving_path,'best_checkpoint.pth'))

        trainer = Trainer(accelerator="gpu", devices=1)

        dataset = datasets.PytorchDataset(hr_data_path=self.test_hr_path,
                                 lr_data_path=self.test_lr_path, 
                                 filenames=self.test_filenames, 
                                 scale_factor=self.scale_factor, 
                                 crappifier_name=self.crappifier_method, 
                                 lr_patch_shape=None, 
                                 num_patches=1, 
                                 transformations=datasets.ToTensor(),
                                 datagen_sampling_pdf= self.datagen_sampling_pdf)

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        print('Prediction is going to start:')
        predictions = trainer.predict(model, dataloaders=dataloader)
        predictions = np.array([np.expand_dims(np.squeeze(e.detach().numpy()),axis=-1) for e in predictions])
        
        if self.verbose:
            data = next(iter(dataloader))
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
                
        #assert np.max(self.Y_test[0]) <= 1.0 and np.max(self.predictions[0]) <= 1.0
        #assert np.min(self.Y_test[0]) >= 0.0 and np.min(self.predictions[0]) >= 0.0
    
 
def train_configuration(data_name, 
                 train_lr_path, train_hr_path, 
                 val_lr_path, val_hr_path,
                 test_lr_path, test_hr_path,
                 additional_folder, train_config, 
                 model_name, model_configuration,
                 verbose=0, gpu_id=0
                ):
    
    crappifier_method = train_config['dataset_config']['crappifier']
    scale_factor = train_config['dataset_config']['scale']
    num_patches = train_config['dataset_config']['num_patches']
    patch_size_x = train_config['dataset_config']['patch_size_x']
    patch_size_y = train_config['dataset_config']['patch_size_y']
    
    number_of_epochs = train_config['number_of_epochs']
    batch_size = train_config['batch_size']
    learning_rate =  train_config['learning_rate']
    discriminator_learning_rate =  train_config['discriminator_learning_rate']
    optimizer =  train_config['optimizer']
    discriminator_optimizer =  train_config['discriminator_optimizer']
    scheduler =  train_config['scheduler']
    discriminator_lr_scheduler = train_config['discriminator_lr_scheduler']
    test_metric_indexes = train_config['test_metric_indexes']
    seed = train_config['seed']
    validation_split = train_config['validation_split']
    data_augmentation = train_config['data_augmentation']
    datagen_sampling_pdf = train_config['datagen_sampling_pdf']

    if model_name in ['wgan', 'esrganplus']:
        model_trainer = PytorchTrainer(data_name, 
                 train_lr_path, train_hr_path, 
                 val_lr_path, val_hr_path,
                 test_lr_path, test_hr_path,
                 crappifier_method, model_name, scale_factor, 
                 number_of_epochs, batch_size, 
                 learning_rate, discriminator_learning_rate, 
                 optimizer, scheduler, 
                 test_metric_indexes, additional_folder, 
                 model_configuration, seed,
                 num_patches, patch_size_x, patch_size_y, 
                 validation_split, data_augmentation,
                 datagen_sampling_pdf,
                 train_config=train_config,
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
                 optimizer, scheduler, 
                 test_metric_indexes, additional_folder, 
                 model_configuration, seed,
                 num_patches, patch_size_x, patch_size_y, 
                 validation_split, data_augmentation,
                 datagen_sampling_pdf,
                 train_config=train_config,
                 discriminator_optimizer=discriminator_optimizer, 
                 discriminator_lr_scheduler=discriminator_lr_scheduler,
                 verbose=verbose
                )
    else:
        raise Exception("Not available model.") 
        
    return model_trainer.launch()
