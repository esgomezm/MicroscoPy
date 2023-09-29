import numpy as np

from . import model

def select_model(
    model_name=None,
    input_shape=None,
    output_channels=None,
    scale_factor=None,
    model_configuration=None,
    batch_size=None,
    data_len=None,
    lr_patch_size_x=None,
    lr_patch_size_y=None,
    datagen_sampling_pdf=None,
    learning_rate_g=None,
    learning_rate_d=None,
    epochs=None,
    train_hr_path=None,
    train_lr_path=None,
    train_filenames=None,
    val_hr_path=None,
    val_lr_path=None,
    val_filenames=None,
    crappifier_method=None,
    save_basedir=None,
    g_optimizer=None,
    d_optimizer=None,
    g_scheduler=None,
    d_scheduler=None,
    checkpoint=None,
    verbose=None
):
    """
    Selects and returns a specific model based on the given parameters.

    Args:
        model_name (str, optional): The name of the model to select. Defaults to None.
        input_shape (tuple, optional): The shape of the input data. Defaults to None.
        output_channels (int, optional): The number of output channels for the selected model. Defaults to None.
        scale_factor (int, optional): The scale factor for upsampling the input data. Defaults to None.
        model_configuration (object, optional): The configuration object for the model. Defaults to None.
        batch_size (int, optional): The batch size for training the model. Defaults to None.
        lr_patch_size_x (int, optional): The patch size in the x-direction for training the model. Defaults to None.
        lr_patch_size_y (int, optional): The patch size in the y-direction for training the model. Defaults to None.
        datagen_sampling_pdf (array, optional): The sampling probability distribution function for the data generator. Defaults to None.
        learning_rate_g (float, optional): The learning rate for the generator model. Defaults to None.
        learning_rate_d (float, optional): The learning rate for the discriminator model. Defaults to None.
        epochs (int, optional): The number of epochs for training the model. Defaults to None.
        train_hr_path (str, optional): The file path for the high-resolution training data. Defaults to None.
        train_lr_path (str, optional): The file path for the low-resolution training data. Defaults to None.
        train_filenames (list, optional): The list of training file names. Defaults to None.
        val_hr_path (str, optional): The file path for the high-resolution validation data. Defaults to None.
        val_lr_path (str, optional): The file path for the low-resolution validation data. Defaults to None.
        val_filenames (list, optional): The list of validation file names. Defaults to None.
        crappifier_method (str, optional): The method used for creating low-resolution data. Defaults to None.
        save_basedir (str, optional): The base directory for saving the model. Defaults to None.
        g_optimizer (str, optional): The optimizer for the generator model. Defaults to None.
        d_optimizer (str, optional): The optimizer for the discriminator model. Defaults to None.
        g_scheduler (str, optional): The scheduler for the generator model. Defaults to None.
        d_scheduler (str, optional): The scheduler for the discriminator model. Defaults to None.
        checkpoint (bool, optional): Whether to save checkpoints during training. Defaults to None.
        verbose (int, optional): The verbosity level. Defaults to None.

    Returns:
        object: The selected model based on the given parameters.

    Raises:
        Exception: If the selected model is not available in the TensorFlow configuration.
    """

    if model_name == "unet":
        return model.unet.preResUNet(
            output_channels=output_channels,
            numInitChannels=model_configuration.init_channels,
            image_shape=input_shape[1:],
            depth=model_configuration.depth,
            upsampling_factor=scale_factor,
            maxpooling=model_configuration.maxpooling,
            upsample_method=model_configuration.upsample_method,
            final_activation="linear",
        )
    elif model_name == "rcan":
        return model.rcan.rcan(
            n_sub_block=int(np.log2(scale_factor)),
            filters=model_configuration.num_filters,
            out_channels=1,
        )

    elif model_name == "dfcan":
        return model.dfcan.DFCAN(
            (input_shape[1:]),
            scale=scale_factor,
            n_ResGroup=model_configuration.n_ResGroup,
            n_RCAB=model_configuration.n_RCAB,
        )

    elif model_name == "wdsr":
        # Custom WDSR B model (0.62M parameters)
        return model.wdsr.wdsr_b(
            scale=scale_factor,
            num_res_blocks=model_configuration.num_res_blocks,
        )

    elif model_name == "cddpm":
        return model.cddpm.DiffusionModel(
            image_shape=input_shape[1:],
            widths=model_configuration.widths,
            block_depth=model_configuration.block_depth,
            scale_factor=scale_factor,
            min_signal_rate=0.02,
            max_signal_rate=0.95,
            batch_size=batch_size,
            ema=0.999,
            embedding_max_frequency=1000.0,
            embedding_dims=32,
            verbose=verbose,
        )

    elif model_name == "wgan":
        print(model_configuration)
        return model.wgan.WGANGP(
            g_layers=model_configuration.used_model.g_layers,
            recloss=model_configuration.used_model.recloss,
            lambda_gp=model_configuration.used_model.lambda_gp,
            n_critic_steps=model_configuration.used_model.n_critic_steps,
            data_len=data_len,
            scale_factor=scale_factor,
            epochs=epochs,
            learning_rate_g=learning_rate_g,
            learning_rate_d=learning_rate_d,
            save_basedir=save_basedir,
            gen_checkpoint=checkpoint,
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
            g_scheduler=g_scheduler,
            d_scheduler=d_scheduler,
            additonal_configuration=model_configuration,
            verbose=verbose
        )

    elif model_name == "esrganplus":
        return model.esrganplus.ESRGANplus(
            datagen_sampling_pdf=datagen_sampling_pdf,
            n_critic_steps=model_configuration.used_model.n_critic_steps,
            data_len=data_len,
            epochs=epochs,
            scale_factor=scale_factor,
            learning_rate_d=learning_rate_d,
            learning_rate_g=learning_rate_g,
            save_basedir=save_basedir,
            gen_checkpoint=checkpoint,
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
            g_scheduler=g_scheduler,
            d_scheduler=d_scheduler,
            additonal_configuration=model_configuration,
            verbose=verbose
        )

    else:
        raise Exception("Not available model in TF configuration.")
