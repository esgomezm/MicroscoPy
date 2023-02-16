import tensorflow as tf
import numpy as np

from tensorflow.keras.callbacks import ReduceLROnPlateau
from model import rcan, dfcan, wdsr, unet, wgan, esrganplus
from tensorflow_callbacks import OneCycleScheduler, MultiStepScheduler

######
  
def select_model(model_name, input_shape, output_channels, down_factor, batch_size, 
                 lr_patch_size_x, lr_patch_size_y, learning_rate_g, learning_rate_d,
                 g_optimizer, d_optimizer, g_scheduler, d_scheduler,
                 epochs, only_hr_images_basedir, type_of_data, save_basedir,
                 model_configuration):
        
    if model_name == 'rcan':
        return rcan.rcan(n_sub_block=int(np.log2(down_factor)), 
                     filters=model_configuration['rcan']['num_filters'], 
                     out_channels = 1)
        
    elif model_name == 'dfcan':
        return dfcan.DFCAN((input_shape[1:]), scale=down_factor, 
                            n_ResGroup = model_configuration['dfcan']['n_ResGroup'], 
                            n_RCAB = model_configuration['dfcan']['n_RCAB'])

    elif model_name == 'wdsr':
        # Custom WDSR B model (0.62M parameters)
        return wdsr.wdsr_b(scale=down_factor, num_res_blocks=model_configuration['wdsr']['num_res_blocks'])
        
    elif model_name == 'unet':
        return unet.preResUNet( output_channels=output_channels,
                            numInitChannels=model_configuration['unet']['init_channels'], 
                            image_shape = input_shape[1:], 
                            depth = model_configuration['unet']['depth'],
                            upsampling_factor = down_factor, 
                            maxpooling=model_configuration['unet']['maxpooling'],
                            upsample_method=model_configuration['unet']['upsample_method'], 
                            final_activation = 'linear')

    elif model_name == 'wgan':

        return wgan.WGANGP(
            g_layers=model_configuration['wgan']['g_layers'], 
            d_layers=model_configuration['wgan']['d_layers'], 
            batchsize=batch_size,
    		lr_patch_size_x=lr_patch_size_x,
    		lr_patch_size_y=lr_patch_size_y,
            down_factor=down_factor,
            recloss=model_configuration['wgan']['recloss'],
            lambda_gp=model_configuration['wgan']['lambda_gp'],
            learning_rate_g=learning_rate_g,
            learning_rate_d=learning_rate_d,
            validation_split = 0.1,
            epochs = epochs,
            rotation = True,
            horizontal_flip = True,
            vertical_flip = True,
            hr_imgs_basedir = '', 
            lr_imgs_basedir = '',
            only_high_resolution_data = True,
            only_hr_images_basedir = only_hr_images_basedir,
            type_of_data = type_of_data,
            save_basedir = save_basedir,
            gen_checkpoint = None,
            g_optimizer = g_optimizer,
            d_optimizer = d_optimizer,
            g_scheduler = g_scheduler,
            d_scheduler = d_scheduler
        )
    
    elif model_name == 'esrganplus':

        return esrganplus.ESRGANplus(batchsize=batch_size,
                                    lr_patch_size_x=lr_patch_size_x,
                                    lr_patch_size_y=lr_patch_size_y,
                                    down_factor=down_factor,
                                    learning_rate_d=learning_rate_d,
                                    learning_rate_g=learning_rate_g,
                                    n_critic_steps=model_configuration['esrganplus']['n_critic_steps'],
                                    validation_split=0.1,
                                    epochs=epochs,
                                    rotation = True,
                                    horizontal_flip = True,
                                    vertical_flip = True,
                                    hr_imgs_basedir = "", 
                                    lr_imgs_basedir ="",
                                    only_high_resolution_data = True,
                                    only_hr_images_basedir=only_hr_images_basedir,
                                    type_of_data=type_of_data,
                                    save_basedir=save_basedir,
                                    gen_checkpoint = None, 
                                    g_optimizer = g_optimizer,
                                    d_optimizer = d_optimizer,
                                    g_scheduler = g_scheduler,
                                    d_scheduler = d_scheduler
                        )
                        
    else:
        raise Exception("Not available model in TF configuration.")  
              

#######

def select_optimizer(library_name, optimizer_name, learning_rate, additional_configuration):
    if library_name == 'tensorflow':
        return select_tensorflow_optimizer(optimizer_name, learning_rate, additional_configuration)
    elif library_name == 'pytorch':
        return select_pytorch_optimizer(optimizer_name, learning_rate, additional_configuration)
    else:
        raise Exception("Wrong library name.")
        
def select_tensorflow_optimizer(optimizer_name, learning_rate, additional_configuration):
    if optimizer_name == 'RMSprop':
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_name == 'Adam':
        return tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                        beta_1=additional_configuration['optim']['adam']['beta1'],
                                        beta_2=additional_configuration['optim']['adam']['beta2'],
                                        epsilon=additional_configuration['optim']['adam']['epsilon'])
    elif optimizer_name == 'Adamax':
        return tf.keras.optimizers.experimental.Adamax(learning_rate=learning_rate,
				                        beta_1=additional_configuration['optim']['adamax']['beta1'],
				                        beta_2=additional_configuration['optim']['adamax']['beta2'],
				                        epsilon=additional_configuration['optim']['adamax']['epsilon'])
    elif optimizer_name == 'AdamW':
        return tf.keras.optimizers.experimental.AdamW(learning_rate=learning_rate,
				                        weight_decay=additional_configuration['optim']['adamW']['weight_decay'],
				                        beta_1=additional_configuration['optim']['adamW']['beta1'],
				                        beta_2=additional_configuration['optim']['adamW']['beta2'],
				                        epsilon=additional_configuration['optim']['adamW']['epsilon'])
    elif optimizer_name == 'SGD':
        return tf.keras.optimizers.SGD(learning_rate=learning_rate,
        				momentum=additional_configuration['optim']['sgd_momentum'])
    else:
        raise Exception("No available optimizer.")

def select_pytorch_optimizer(optimizer_name, learning_rate, additional_configuration):
    pass

#######

def select_lr_schedule(library_name, lr_scheduler_name, input_shape, batch_size, number_of_epochs, learning_rate, additional_configuration):
    if library_name == 'tensorflow':
        return select_tensorflow_lr_schedule(lr_scheduler_name, input_shape, batch_size, number_of_epochs, learning_rate, additional_configuration)
    elif library_name == 'pytorch':
        return select_pytorch_lr_schedule(lr_scheduler_name, input_shape, batch_size, number_of_epochs, learning_rate, additional_configuration)
    else:
        raise Exception("Wrong library name.")


def select_tensorflow_lr_schedule(lr_scheduler_name, input_shape, batch_size, number_of_epochs, 
                                  learning_rate, additional_configuration):
    if lr_scheduler_name == 'OneCycle':
        steps = np.ceil(input_shape[0] / batch_size) * number_of_epochs
        return OneCycleScheduler(learning_rate, steps)
    elif lr_scheduler_name == 'ReduceOnPlateau':
        return ReduceLROnPlateau(monitor=additional_configuration['optim']['ReduceOnPlateau']['monitor'],
        			factor=additional_configuration['optim']['ReduceOnPlateau']['factor'], 
        			patience=additional_configuration['optim']['ReduceOnPlateau']['patience'], 
                    		min_lr=(learning_rate/10))
    elif lr_scheduler_name == 'CosineDecay':
        decay_steps = np.ceil(input_shape[0] / batch_size) * number_of_epochs
        return tf.keras.optimizers.schedules.CosineDecay(learning_rate, decay_steps, alpha=0.0, name=None)
    elif lr_scheduler_name == 'MultiStepScheduler':
        return MultiStepScheduler(learning_rate,
        			  lr_steps=additional_configuration['optim']['MultiStepScheduler']['lr_steps'], 
        			  lr_rate_decay=additional_configuration['optim']['MultiStepScheduler']['lr_rate_decay'])
    elif lr_scheduler_name is None:
        return None
    else:
        raise Exception("Not available LR Scheduler.")  

def select_pytorch_lr_schedule(lr_scheduler_name, input_shape, batch_size, number_of_epochs, learning_rate, additional_configuration):
    pass
