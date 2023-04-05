#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "2";

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


# In[3]:


train_path = '../datasets/TFM - dataset Electron Microscopy/train'
test_path = '../datasets/TFM - dataset Electron Microscopy/test'
drawn_test_path = './data_example/drawn_test'

train_filenames = sorted([os.path.join(filename) for filename in os.listdir(train_path)])
test_filenames = sorted([os.path.join(filename) for filename in os.listdir(test_path)])
drawn_test_filenames = sorted([os.path.join(filename) for filename in os.listdir(drawn_test_path)])

print(len(train_filenames))
print(len(test_filenames))
print(len(drawn_test_filenames))


# # See one image

# In[38]:


from src.datasets import read_image
hr_img = read_image(os.path.join(train_path, train_filenames[0]))
test_img = read_image(os.path.join(test_path, test_filenames[0]))
drawn_test_img = read_image(os.path.join(drawn_test_path, drawn_test_filenames[0]))

print_info(drawn_test_img)


# # Test `prepare_data`

# In[49]:

scale = 4

from src.datasets import extract_random_patches_from_folder

train_patches_wf, train_patches_gt = extract_random_patches_from_folder(
                                        hr_data_path=train_path, 
                                        lr_data_path=None, 
                                        filenames=train_filenames, 
                                        scale_factor=scale, 
                                        crappifier_name='em_crappify', 
                                        lr_patch_shape=(64, 64), 
                                        num_patches=16)

X_train = np.expand_dims(train_patches_wf, axis=-1)
Y_train = np.expand_dims(train_patches_gt, axis=-1)

print('X_train')
print_info(X_train)
print('\n')

print('Y_train')
print_info(Y_train)

# In[50]:

from src.datasets import get_train_val_generators 
batch_size = 8

train_generator, val_generator = get_train_val_generators(X_data=X_train,
                                                          Y_data=Y_train,
                                                          validation_split=0.1,
                                                          batch_size=batch_size,
                                                          show_examples=0,
                                                          rotation=True,
                                                          horizontal_flip=True,
                                                          vertical_flip=True)


# In[52]:


for lr,hr in train_generator:
    print('LR')
    print_info(lr)
    print('\n')
    print('HR')
    print_info(hr)
    break


# # Create a simple network (DFCAN for example)

# In[53]:


library_name = 'tensorflow'

optimizer = 'Adam'

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

from src.optimizer_scheduler_utils import select_optimizer, select_optimizer
optim = select_optimizer(library_name=library_name, optimizer_name=optimizer, 
                                learning_rate=0.001, check_point=None,
                                parameters=None, additional_configuration=model_configuration)
   

model_name = 'dfcan'

from src.model_utils import select_model
from src.utils import ssim_loss
model = select_model(model_name=model_name, input_shape=X_train.shape, output_channels=Y_train.shape[-1], 
                        scale_factor=scale, model_configuration=model_configuration)

loss_funct = 'mean_absolute_error'
eval_metric = 'mean_squared_error'

model.compile(optimizer=optim, loss=loss_funct, metrics=[eval_metric, ssim_loss])

trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
totalParams = trainableParams + nonTrainableParams

print('Trainable parameteres: {} \nNon trainable parameters: {} \nTotal parameters: {}'.format(trainableParams, 
                                                                                                        nonTrainableParams, 
                                                                                                    totalParams))


# In[54]:

model.summary()

# In[62]:


from tensorflow.keras.callbacks import ModelCheckpoint as tf_ModelCheckpoint
from tensorflow.keras.callbacks import LambdaCallback, EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

epochs = 100

scheduler = 'OneCycle'

from src.optimizer_scheduler_utils import select_lr_schedule
lr_schedule = select_lr_schedule(library_name='tensorflow', lr_scheduler_name=scheduler, 
                                    data_len=X_train.shape[0]//batch_size, 
                                    number_of_epochs=epochs, learning_rate=0.001,
                                    monitor_loss=None, name=None, optimizer=None, frequency=None,
                                    additional_configuration=model_configuration)


model_checkpoint = tf_ModelCheckpoint(os.path.join('results', 'weights_best.h5'), 
                               monitor='val_loss',verbose=1, 
                               save_best_only=True, save_weights_only=True)

# callback for early stopping
earlystopper = EarlyStopping(monitor=model_configuration['optim']['early_stop']['loss'],
                 patience=model_configuration['optim']['early_stop']['patience'], 
                             min_delta=0.005, mode=model_configuration['optim']['early_stop']['mode'],
                             verbose=1, restore_best_weights=True)
        

import time
start = time.time()

history = model.fit(train_generator, validation_data=val_generator,
                  validation_steps=np.ceil((0.1*X_train.shape[0])/batch_size),
                  steps_per_epoch=np.ceil(X_train.shape[0]/batch_size),
                  epochs=epochs, 
                  callbacks=[lr_schedule, model_checkpoint, earlystopper])
    
dt = time.time() - start
mins, sec = divmod(dt, 60) 
hour, mins = divmod(mins, 60) 
print("\nTime elapsed:",hour, "hour(s)",mins,"min(s)",round(sec),"sec(s)\n")


# # Make the prediction

# In[69]:


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


# In[70]:


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

# In[71]:


optim = select_optimizer(library_name=library_name, optimizer_name=optimizer, 
                                learning_rate=0.001, check_point=None,
                                parameters=None, additional_configuration=model_configuration)

model = select_model(model_name=model_name, input_shape=lr_images.shape,  
                     output_channels=1, scale_factor=scale, model_configuration=model_configuration)

loss_funct = 'mean_absolute_error'
eval_metric = 'mean_squared_error'

model.compile(optimizer=optim, loss=loss_funct, metrics=[eval_metric])

# Load old weights
model.load_weights( os.path.join('results', 'weights_best.h5') )   


# In[ ]:


test_predictions = model.predict(lr_images, batch_size=1)
print('Test predictions')
print_info(test_predictions)
print('\n')

# In[ ]:


drawn_test_predictions = model.predict(drawn_lr_images, batch_size=1)
print('Drawn predictions')
print_info(drawn_test_predictions)
print('\n')

# In[ ]:


plot_images([test_predictions[0], drawn_test_predictions[0]])




