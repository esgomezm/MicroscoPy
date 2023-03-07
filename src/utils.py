import tensorflow as tf
import numpy as np
import os

import torch

from tensorflow.keras.applications.vgg19 import VGG19
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model

tf.config.experimental_run_functions_eagerly(True)
tf.config.run_functions_eagerly(True)

# Colors for the warning messages
class bcolors:
    W  = '\033[0m'  # white (normal)
    R  = '\033[31m' # red
    WARNING = '\033[31m'

def set_seed(seedValue=42):
    """Sets the seed on multiple python modules to obtain results as
    reproducible as possible.
    Args:
    seedValue (int, optional): seed value.
    """
    np.random.seed(seed=seedValue)
    tf.random.set_seed(seedValue)
    os.environ["PYTHONHASHSEED"]=str(seedValue)
    torch.manual_seed(seedValue)
    torch.cuda.manual_seed_all(seedValue)

def ssim_loss(y_true, y_pred):
        return tf.image.ssim(y_true, y_pred, max_val=1.0)
    
def vgg_loss(image_shape):
    vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=(image_shape[0], image_shape[1],3))
    vgg19.trainable = False
    for l in vgg19.layers:
        l.trainable = False
    model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
    model.trainable = False
    
    def vgg_loss_fixed( y_true, y_pred ):
        y_true_3chan = K.concatenate( [y_true,y_true,y_true], axis=-1 )
        y_pred_3chan = K.concatenate( [y_pred,y_pred,y_pred], axis=-1 )
        return K.mean(K.square(model(y_true_3chan) - model(y_pred_3chan)))

    return vgg_loss_fixed

def perceptual_loss(image_shape, percp_coef=0.1): 
    mean_absolute_error = tf.keras.losses.MeanAbsoluteError()
    percp_loss = vgg_loss(image_shape)
    
    def mixed_loss(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred) + percp_coef * percp_loss(y_true, y_pred)
    
    return mixed_loss

def print_info(image_name, data):
    try:
        np_data = np.array(data)
        print('{} \n\nShape: {} \nType: {} \nNumpy type: {} \nMin: {} \nMax: {} \nMean: {}\n'.format(image_name, 
                                                                    np_data.shape, type(data), np_data.dtype, 
                                                                    np.min(np_data), np.max(np_data), 
                                                                    np.mean(np_data)))
    except:
        print('{} \n\nNot same shapes'.format(image_name))

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
        1 / np.power(
            10000, np.arange(0, self_channels, 2) / np.float32(self_channels)
        )
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
