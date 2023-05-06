import numpy as np

from . import model

######

def select_model(model_name=None, input_shape=None, output_channels=None, scale_factor=None, model_configuration=None,
                 batch_size=None, num_patches=None, lr_patch_size_x=None, lr_patch_size_y=None, datagen_sampling_pdf=None,
                 learning_rate_g=None, learning_rate_d=None, epochs=None, train_hr_path=None, train_lr_path=None, train_filenames = None,
                 val_hr_path=None, val_lr_path=None, val_filenames=None, crappifier_method=None,
                 save_basedir=None, g_optimizer=None, d_optimizer=None, g_scheduler=None, d_scheduler=None,
                 checkpoint=None):
        
    if model_name == 'unet':
        return model.unet.preResUNet( output_channels=output_channels,
                            numInitChannels=model_configuration['unet']['init_channels'], 
                            image_shape = input_shape[1:], 
                            depth = model_configuration['unet']['depth'],
                            upsampling_factor = scale_factor, 
                            maxpooling=model_configuration['unet']['maxpooling'],
                            upsample_method=model_configuration['unet']['upsample_method'], 
                            final_activation = 'linear')
    elif model_name == 'rcan':
        return model.rcan.rcan(n_sub_block=int(np.log2(scale_factor)), 
                     filters=model_configuration['rcan']['num_filters'], 
                     out_channels = 1)
        
    elif model_name == 'dfcan':
        return model.dfcan.DFCAN((input_shape[1:]), scale=scale_factor, 
                            n_ResGroup = model_configuration['dfcan']['n_ResGroup'], 
                            n_RCAB = model_configuration['dfcan']['n_RCAB'])

    elif model_name == 'wdsr':
        # Custom WDSR B model (0.62M parameters)
        return model.wdsr.wdsr_b(scale=scale_factor, num_res_blocks=model_configuration['wdsr']['num_res_blocks'])
        
    elif model_name =='cddpm':
        return model.cddpm.DiffusionModel(
            image_shape=input_shape[1:], widths=model_configuration['cddpm']['widths'], 
            block_depth=model_configuration['cddpm']['block_depth'], 
            min_signal_rate=0.02, max_signal_rate=0.95, 
            batch_size=batch_size, ema=0.999)

    elif model_name == 'wgan':
        return model.wgan.WGANGP(
            g_layers=model_configuration['wgan']['g_layers'],
            batchsize=batch_size,
            num_patches=num_patches,
    		lr_patch_size_x=lr_patch_size_x,
    		lr_patch_size_y=lr_patch_size_y,
            scale_factor=scale_factor,
            datagen_sampling_pdf=datagen_sampling_pdf,
            recloss=model_configuration['wgan']['recloss'],
            lambda_gp=model_configuration['wgan']['lambda_gp'],
            learning_rate_g=learning_rate_g,
            learning_rate_d=learning_rate_d,
            epochs = epochs,
            rotation = True,
            horizontal_flip = True,
            vertical_flip = True,
            train_hr_path = train_hr_path,
            train_lr_path = train_lr_path,
            train_filenames = train_filenames,
            val_hr_path = val_hr_path,
            val_lr_path = val_lr_path,
            val_filenames = val_filenames,
            crappifier_method= crappifier_method,
            save_basedir = save_basedir,
            gen_checkpoint = checkpoint,
            g_optimizer = g_optimizer,
            d_optimizer = d_optimizer,
            g_scheduler = g_scheduler,
            d_scheduler = d_scheduler,
            additonal_configuration = model_configuration
        )
    
    elif model_name == 'esrganplus':

        return model.esrganplus.ESRGANplus(batchsize=batch_size,
                                    num_patches=num_patches,
                                    lr_patch_size_x=lr_patch_size_x,
                                    lr_patch_size_y=lr_patch_size_y,
                                    scale_factor=scale_factor,
                                    datagen_sampling_pdf=datagen_sampling_pdf,
                                    learning_rate_d=learning_rate_d,
                                    learning_rate_g=learning_rate_g,
                                    n_critic_steps=model_configuration['esrganplus']['n_critic_steps'],
                                    epochs=epochs,
                                    rotation = True,
                                    horizontal_flip = True,
                                    vertical_flip = True,
                                    train_hr_path = train_hr_path,
                                    train_lr_path = train_lr_path,
                                    train_filenames = train_filenames,
                                    val_hr_path = val_hr_path,
                                    val_lr_path = val_lr_path,
                                    val_filenames = val_filenames,
                                    crappifier_method= crappifier_method,
                                    save_basedir=save_basedir,
                                    gen_checkpoint = checkpoint, 
                                    g_optimizer = g_optimizer,
                                    d_optimizer = d_optimizer,
                                    g_scheduler = g_scheduler,
                                    d_scheduler = d_scheduler,
                                    additonal_configuration = model_configuration
                        )

    else:
        raise Exception("Not available model in TF configuration.")  
              
