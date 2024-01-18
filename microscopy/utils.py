import numpy as np
import os
import random
import torch
import yaml

#####################################
#
# Functions that define normalization tecniques

def min_max_normalization(data, desired_accuracy=np.float32):
    """
    Normalize the given data using min-max normalization.

    Parameters:
        data (ndarray): The data to be normalized.
        desired_accuracy (type): The desired accuracy of the normalized data. Defaults to np.float32.

    Returns:
        ndarray: The normalized data.
    """
    return (data - data.min()) / (data.max() - data.min() + 1e-10).astype(
        desired_accuracy
    )

#
#####################################

#####################################
#
# Functions for general purposses (e.g. working with yamls, seeds, etc.)

class bcolors:
    # Colors for the warning messages
    W = "\033[0m"  # white (normal)
    R = "\033[31m"  # red
    WARNING = "\033[31m"

def set_seed(seed_value=42):
    """Sets the seed on multiple python modules to obtain results as
    reproducible as possible.
    Args:
    seedValue (int, optional): seed value.
    """
    import tensorflow as tf
    print(f'Setting seed to {seed_value}')

    # Set the seed for the random module
    random.seed(seed_value)

    # Set the seed for NumPy
    np.random.seed(seed_value)

    # Set the seed for PyTorch (if installed)
    if torch.__version__ >= "1.0":
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set the seed for TensorFlow (if installed)
    if tf.__version__ >= "2.0":
        tf.random.set_seed(seed_value)
    elif tf.__version__ >= "1.0":
        tf.set_random_seed(seed_value)

def print_info(image_name, data):
    """
    Prints information about the given image data.

    Args:
        image_name (str): The name of the image.
        data (list): A list of image data.

    Returns:
        None
    """

    try:
        np_data = np.array(data)
        print(
            "{} \n\nShape: {} \nType: {} \nNumpy type: {} \nMin: {} \nMax: {} \nMean: {}\n".format(
                image_name,
                np_data.shape,
                type(data),
                np_data.dtype,
                np.min(np_data),
                np.max(np_data),
                np.mean(np_data),
            )
        )
    except:
        print("{} \n\nNot same shapes".format(image_name))


def update_yaml(yaml_file_path, key_value, new_value):
    """
    Updates a value in a yaml file.

    :param yaml_file_path: The path to the yaml file.
    :type yaml_file_path: str
    :param key_value: The key representing the value to be updated.
    :type key_value: str
    :param new_value: The new value to update the given key with.
    :type new_value: Any
    :return: None
    """

    file_information = load_yaml(yaml_file_path)
    file_information[key_value] = new_value
    save_yaml(file_information, yaml_file_path)


def load_yaml(yaml_file_path):
    """
    Load and parse a YAML file.

    :param yaml_file_path: A string representing the path to the YAML file.
    :return: A dictionary representing the parsed YAML data.
    """

    with open(yaml_file_path) as file:
        file_information = yaml.full_load(file)
    return file_information


def save_yaml(dict_to_save, saving_path):
    """
    Saves a dictionary to a YAML file at the provided path.

    Args:
        dict_to_save (dict): The dictionary to save to a YAML file.
        saving_path (str): The path to save the YAML file to.

    Returns:
        None
    """
    from omegaconf import OmegaConf
    with open(saving_path, "w") as file:
        OmegaConf.save(dict_to_save, file)

#
#####################################

#####
#
# Function that define different losses

def ssim_loss(y_true, y_pred):
    """
    Calculates the Structural Similarity Index (SSIM) loss between two images.

    Parameters:
        y_true (tensor): The true image.
        y_pred (tensor): The predicted image.

    Returns:
        tensor: The SSIM loss.

    Example:
        >>> y_true = tf.constant([0.5, 0.8, 0.2, 0.3])
        >>> y_pred = tf.constant([0.6, 0.7, 0.3, 0.4])
        >>> ssim_loss(y_true, y_pred)
        <tf.Tensor: shape=(), dtype=float32, numpy=0.75>
    """

    # Printing the loss
    # tf.print('\nSSIM:')
    # tf.print(tf.image.ssim(y_true, y_pred, max_val=1.0))
    import tensorflow as tf
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

def vgg_loss(image_shape):
    """
    Generates the VGG loss function for image style transfer.

    Args:
        image_shape (tuple): The shape of the input image. Should be a tuple of (height, width).

    Returns:
        function: The VGG loss function that takes in the ground truth image and the predicted image 
        as inputs, and returns the mean squared difference between the VGG feature representations of the two images.

    Note:
        The VGG loss function is based on the VGG19 model pretrained on the ImageNet dataset. 
        It computes the mean squared difference between the VGG feature maps of the ground truth image and the predicted image. 
        The VGG19 model is frozen and not trainable during the execution of this loss function.

    Example:
        vgg_loss = vgg_loss(image_shape=(256, 256))
        loss = vgg_loss(ground_truth_image, predicted_image)
    """
    from tensorflow.keras.applications.vgg19 import VGG19
    import tensorflow.keras.backend as K
    from tensorflow.keras.models import Model
    
    vgg19 = VGG19(
        include_top=False,
        weights="imagenet",
        input_shape=(image_shape[0], image_shape[1], 3),
    )
    vgg19.trainable = False
    for l in vgg19.layers:
        l.trainable = False
    model = Model(inputs=vgg19.input, outputs=vgg19.get_layer("block5_conv4").output)
    model.trainable = False

    def vgg_loss_fixed(y_true, y_pred):
        y_true_3chan = K.concatenate([y_true, y_true, y_true], axis=-1)
        y_pred_3chan = K.concatenate([y_pred, y_pred, y_pred], axis=-1)
        return K.mean(K.square(model(y_true_3chan) - model(y_pred_3chan)))

    return vgg_loss_fixed

def perceptual_loss(image_shape, percp_coef=0.1):
    """
    Returns a loss function that combines the mean absolute error loss and the VGG loss.
    
    Parameters:
        image_shape (tuple): The shape of the input images.
        percp_coef (float, optional): The coefficient for the perceptual loss. Defaults to 0.1.
        
    Returns:
        mixed_loss (function): A loss function that combines the mean absolute error loss and the perceptual loss.
            The function takes in two tensors, y_true and y_pred, and returns the sum of the mean absolute error loss
            and the product of the perceptual loss and the percp_coef.
    """
    import tensorflow as tf
    mean_absolute_error = tf.keras.losses.MeanAbsoluteError()
    percp_loss = vgg_loss(image_shape)

    def mixed_loss(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred) + percp_coef * percp_loss(
            y_true, y_pred
        )

    return mixed_loss

#
#####################################

#####################################
#
# Function for adding embeddings to the images

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = np.stack((np.sin(sin_inp), np.cos(sin_inp)), -1)
    emb = np.reshape(emb, (*emb.shape[:-2], -1))
    return emb


def concatenate_encoding(images, channels):
    self_channels = int(2 * np.ceil(channels / 4))
    inv_freq = np.float32(
        1 / np.power(10000, np.arange(0, self_channels, 2) / np.float32(self_channels))
    )

    _, x, y, org_channels = images.shape

    pos_x = np.arange(x)
    pos_y = np.arange(y)

    sin_inp_x = np.einsum("i,j->ij", pos_x, inv_freq)
    sin_inp_y = np.einsum("i,j->ij", pos_y, inv_freq)

    emb_x = np.expand_dims(get_emb(sin_inp_x), 1)
    emb_y = np.expand_dims(get_emb(sin_inp_y), 0)

    emb_x = np.tile(emb_x, (1, y, 1))
    emb_y = np.tile(emb_y, (x, 1, 1))
    emb = np.concatenate((emb_x, emb_y), -1)
    cached_penc = np.repeat(emb[None, :, :, :org_channels], np.shape(images)[0], axis=0)
    return np.concatenate((images, cached_penc), -1)

#
#####################################

#####################################
#
# Functions for U-Net's padding

def calculate_pad_for_Unet(lr_img_shape, depth_Unet, is_pre, scale):
    assert (
        len(lr_img_shape) == 3
    ), "LR image shape should have a length of three: (cols x rows x channels)."

    lr_height = lr_img_shape[0]
    lr_width = lr_img_shape[1]

    if is_pre:
        lr_height *= scale
        lr_width *= scale

    if lr_width % 2**depth_Unet != 0 or lr_height % 2**depth_Unet != 0:
        height_gap = ((lr_height // 2**depth_Unet) + 1) * 2**depth_Unet - lr_height
        width_gap = ((lr_width // 2**depth_Unet) + 1) * 2**depth_Unet - lr_width

        if is_pre:
            height_gap //= scale
            width_gap //= scale

        height_padding = (height_gap // 2 + height_gap % 2, height_gap // 2)
        width_padding = (width_gap // 2 + width_gap % 2, width_gap // 2)

        if is_pre:
            if height_gap == 1:
                height_padding = (height_gap, 0)
            if width_gap == 1:
                width_padding = (width_gap, 0)

        return height_padding, width_padding
    else:
        return (0, 0), (0, 0)


def add_padding_for_Unet(lr_imgs, height_padding, width_padding):
    if len(lr_imgs.shape) == 4:
        pad_lr_imgs = np.pad(
            lr_imgs,
            ((0, 0), height_padding, width_padding, (0, 0)),
            mode="constant",
            constant_values=0,
        )
    elif len(lr_imgs.shape) == 3:
        pad_lr_imgs = np.pad(
            lr_imgs,
            (height_padding, width_padding, (0, 0)),
            mode="constant",
            constant_values=0,
        )

    return pad_lr_imgs

def remove_padding_for_Unet(pad_hr_imgs, height_padding, width_padding, scale):
    if len(pad_hr_imgs.shape) == 4:
        hr_height_padding_left = (
            -height_padding[1] * scale
            if height_padding[1] > 0
            else pad_hr_imgs.shape[1]
        )
    elif len(pad_hr_imgs.shape) == 3:
        hr_height_padding_left = (
            -height_padding[1] * scale
            if height_padding[1] > 0
            else pad_hr_imgs.shape[0]
        )

    if len(pad_hr_imgs.shape) == 4:
        hr_width_padding_left = (
            -width_padding[1] * scale if width_padding[1] > 0 else pad_hr_imgs.shape[2]
        )
    elif len(pad_hr_imgs.shape) == 3:
        hr_width_padding_left = (
            -width_padding[1] * scale if width_padding[1] > 0 else pad_hr_imgs.shape[1]
        )

    hr_height_padding = (height_padding[0] * scale, hr_height_padding_left)
    hr_width_padding = (width_padding[0] * scale, hr_width_padding_left)

    if len(pad_hr_imgs.shape) == 4:
        hr_imgs = pad_hr_imgs[
            :,
            hr_height_padding[0] : hr_height_padding[1],
            hr_width_padding[0] : hr_width_padding[1],
            :,
        ]
    elif len(pad_hr_imgs.shape) == 3:
        hr_imgs = pad_hr_imgs[
            hr_height_padding[0] : hr_height_padding[1],
            hr_width_padding[0] : hr_width_padding[1],
            :,
        ]

    return hr_imgs

#
#####################################