

import sys
sys.setrecursionlimit(10000)

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dropout, BatchNormalization, SpatialDropout2D
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, SeparableConv2D
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Concatenate, Add, concatenate, Lambda
from tensorflow.keras.layers import Input, UpSampling2D, Activation
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.nn import depth_to_space

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

######

# Multi Scale Similarity Index loss

# Default values obtained by Wang et al.
_MSSSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)

def ms_ssim_loss( max_val=1, power_factors=_MSSSIM_WEIGHTS, filter_size=11,
    filter_sigma=1.5, k1=0.01, k2=0.03 ):
  def ms_ssim_loss_fixed( y_true, y_pred ):
    return 1 - tf.image.ssim_multiscale(y_true, y_pred, max_val, power_factors,
                                        filter_size, filter_sigma, k1, k2)
  return ms_ssim_loss_fixed

# Used in mix_loss
mean_absolute_error = tf.keras.losses.MeanAbsoluteError()

# Mix loss as defined by Zhao et al. in https://arxiv.org/pdf/1511.08861.pdf
def mix_loss( alpha=0.84, max_val=1, power_factors=_MSSSIM_WEIGHTS, filter_size=11,
    filter_sigma=1.5, k1=0.01, k2=0.03 ):
    def mix_loss_fixed( y_true, y_pred ):
        ms_ssim = tf.image.ssim_multiscale( y_true, y_pred, max_val,
                                                power_factors, filter_size,
                                                filter_sigma, k1, k2 )
        return alpha*( 1 - ms_ssim ) + (1-alpha) * mean_absolute_error( y_true, y_pred )
    return mix_loss_fixed

# Method to use PSNR as metric while training
def psnr_loss( y_true, y_pred ):
    return tf.image.psnr( y_true, y_pred, max_val=1.0)

#####

# Sub-pixel layer for learnable upsampling
# From: https://github.com/twairball/keras-subpixel-conv/blob/master/subpixel.py
def SubpixelConv2D(input_shape, scale=4):
    """
    Keras layer to do subpixel convolution.
    NOTE: Tensorflow backend only. Uses tf.depth_to_space
    Ref:
        [1] Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
            Shi et Al.
            https://arxiv.org/abs/1609.05158
    :param input_shape: tensor shape, (batch, height, width, channel)
    :param scale: upsampling scale. Default=4
    :return:
    """
    # upsample using depth_to_space
    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                input_shape[1] * scale,
                input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return depth_to_space(x, scale)


    #return Lambda(subpixel, output_shape=subpixel_shape, name='subpixel')
    return Lambda(subpixel, output_shape=subpixel_shape)

def upsample(x, out_channels=16, method='Upsampling2D', upsampling_factor=2,
             input_shape=None):
    if method == 'Conv2DTranspose':
        if input_shape is None:
            x = Conv2DTranspose(out_channels, (2, 2),
                                strides=(upsampling_factor, upsampling_factor),
                                padding='same') (x)
        else:
            x = Conv2DTranspose(out_channels, (2, 2),
                                strides=(upsampling_factor, upsampling_factor),
                                padding='same', input_shape=input_shape) (x)
    elif method == 'Upsampling2D':
        x = UpSampling2D( size=(upsampling_factor, upsampling_factor) )( x )
    elif method == 'SubpixelConv2D':
        x = Conv2D(out_channels * upsampling_factor ** 2, (3, 3),
                   padding='same')(x)
        x = SubpixelConv2D( input_shape, scale=upsampling_factor )(x)
    else:
        x = UpSampling2D( size=(upsampling_factor, upsampling_factor) )( x )

    return x

def preUNet4(output_channels, filters=16, input_size = (128,128,1), upsampling_factor=2,
          spatial_dropout=False, upsample_method='UpSampling2D'):

  inputs = Input( input_size )
  
  if upsampling_factor > 1:
    s = upsample(inputs, out_channels=1, method=upsample_method,
                 upsampling_factor=upsampling_factor,
                 input_shape=(input_size[0], input_size[1], input_size[2]))
    conv1 = Conv2D(filters, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(s)
  else:
    conv1 = Conv2D(filters, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    
  conv1 = SpatialDropout2D(0.1)(conv1) if spatial_dropout else Dropout(0.1) (conv1)
  conv1 = Conv2D(filters, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
  pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)
  
  conv2 = Conv2D(filters*2, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
  conv2 = SpatialDropout2D(0.1)(conv2) if spatial_dropout else Dropout(0.1) (conv2)
  conv2 = Conv2D(filters*2, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
  pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)
  
  conv3 = Conv2D(filters*4, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
  conv3 = SpatialDropout2D(0.2)(conv3) if spatial_dropout else Dropout(0.2) (conv3)
  conv3 = Conv2D(filters*4, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
  pool3 = AveragePooling2D(pool_size=(2, 2))(conv3)
  
  conv4 = Conv2D(filters*8, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
  conv4 = SpatialDropout2D(0.2)(conv4) if spatial_dropout else Dropout(0.2)(conv4)
  conv4 = Conv2D(filters*8, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
  pool4 = AveragePooling2D(pool_size=(2, 2))(conv4)

  conv5 = Conv2D(filters*16, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
  conv5 = SpatialDropout2D(0.3)(conv5) if spatial_dropout else Dropout(0.3)(conv5)
  conv5 = Conv2D(filters*16, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
  
  up6 = Conv2DTranspose(filters*8, (2, 2), strides=(2, 2), padding='same') (conv5)
  merge6 = concatenate([conv4,up6], axis = 3)
  conv6 = Conv2D(filters*8, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
  conv6 = SpatialDropout2D(0.2)(conv6) if spatial_dropout else Dropout(0.2)(conv6)
  conv6 = Conv2D(filters*8, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

  up7 = Conv2DTranspose(filters*4, (2, 2), strides=(2, 2), padding='same') (conv6)
  merge7 = concatenate([conv3,up7], axis = 3)
  conv7 = Conv2D(filters*4, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
  conv7 = SpatialDropout2D(0.2)(conv7) if spatial_dropout else Dropout(0.2)(conv7)
  conv7 = Conv2D(filters*4, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

  up8 = Conv2DTranspose(filters*2, (2, 2), strides=(2, 2), padding='same') (conv7)
  merge8 = concatenate([conv2,up8], axis = 3)
  conv8 = Conv2D(filters*2, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
  conv8 = SpatialDropout2D(0.1)(conv8) if spatial_dropout else Dropout(0.1)(conv8)
  conv8 = Conv2D(filters*2, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

  up9 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same') (conv8)
  merge9 = concatenate([conv1,up9], axis = 3)
  conv9 = Conv2D(filters, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
  conv9 = SpatialDropout2D(0.1)(conv9) if spatial_dropout else Dropout(0.1)(conv9)
  conv9 = Conv2D(filters, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

  #outputs = Conv2D(1, (1, 1), activation='sigmoid') (conv9)
  outputs = Conv2D(output_channels, (1, 1)) (conv9)
  
  model = Model(inputs=[inputs], outputs=[outputs])
  return model

def postUNet4(output_channels, filters=16, input_size = (128,128,1), upsampling_factor=2,
          spatial_dropout=False, upsample_method='UpSampling2D'):

  inputs = Input( input_size )
  
  conv1 = Conv2D(filters, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
  conv1 = SpatialDropout2D(0.1)(conv1) if spatial_dropout else Dropout(0.1) (conv1)
  conv1 = Conv2D(filters, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
  pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)
  
  conv2 = Conv2D(filters*2, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
  conv2 = SpatialDropout2D(0.1)(conv2) if spatial_dropout else Dropout(0.1) (conv2)
  conv2 = Conv2D(filters*2, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
  pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)
  
  conv3 = Conv2D(filters*4, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
  conv3 = SpatialDropout2D(0.2)(conv3) if spatial_dropout else Dropout(0.2) (conv3)
  conv3 = Conv2D(filters*4, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
  pool3 = AveragePooling2D(pool_size=(2, 2))(conv3)
  
  conv4 = Conv2D(filters*8, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
  conv4 = SpatialDropout2D(0.2)(conv4) if spatial_dropout else Dropout(0.2)(conv4)
  conv4 = Conv2D(filters*8, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
  pool4 = AveragePooling2D(pool_size=(2, 2))(conv4)

  conv5 = Conv2D(filters*16, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
  conv5 = SpatialDropout2D(0.3)(conv5) if spatial_dropout else Dropout(0.3)(conv5)
  conv5 = Conv2D(filters*16, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
  
  up6 = Conv2DTranspose(filters*8, (2, 2), strides=(2, 2), padding='same') (conv5)
  merge6 = concatenate([conv4,up6], axis = 3)
  conv6 = Conv2D(filters*8, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
  conv6 = SpatialDropout2D(0.2)(conv6) if spatial_dropout else Dropout(0.2)(conv6)
  conv6 = Conv2D(filters*8, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

  up7 = Conv2DTranspose(filters*4, (2, 2), strides=(2, 2), padding='same') (conv6)
  merge7 = concatenate([conv3,up7], axis = 3)
  conv7 = Conv2D(filters*4, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
  conv7 = SpatialDropout2D(0.2)(conv7) if spatial_dropout else Dropout(0.2)(conv7)
  conv7 = Conv2D(filters*4, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

  up8 = Conv2DTranspose(filters*2, (2, 2), strides=(2, 2), padding='same') (conv7)
  merge8 = concatenate([conv2,up8], axis = 3)
  conv8 = Conv2D(filters*2, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
  conv8 = SpatialDropout2D(0.1)(conv8) if spatial_dropout else Dropout(0.1)(conv8)
  conv8 = Conv2D(filters*2, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

  up9 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same') (conv8)
  merge9 = concatenate([conv1,up9], axis = 3)
  conv9 = Conv2D(filters, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
  conv9 = SpatialDropout2D(0.1)(conv9) if spatial_dropout else Dropout(0.1)(conv9)
  conv9 = Conv2D(filters, (3,3), activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

  if upsampling_factor > 1:
    conv9 = upsample( conv9, out_channels=1, method=upsample_method,
                        upsampling_factor=upsampling_factor,
                        input_shape=(input_size[0], input_size[1], input_size[2]))

  #outputs = Conv2D(1, (1, 1), activation='sigmoid') (conv9)
  outputs = Conv2D(output_channels, (1, 1)) (conv9)

  model = Model(inputs=[inputs], outputs=[outputs])
  return model

# == Residual U-Net ==

def residual_block(x, dim, filter_size, activation='elu', 
                   kernel_initializer='he_normal', dropout_value=0.2, bn=False,
                   first_conv_strides=1, separable_conv=False, firstBlock=False):

    # Create shorcut
    if firstBlock == False:
        shortcut = Conv2D(dim, activation=None, kernel_size=(1, 1), 
                      strides=first_conv_strides)(x)
    else:
        shortcut = Conv2D(dim, activation=None, kernel_size=(1, 1), 
                      strides=1)(x)
    
    # Main path
    if firstBlock == False:
        x = BatchNormalization()(x) if bn else x
        x = Activation( activation )(x)
    if separable_conv == False:
        if firstBlock == True:
            x = Conv2D(dim, filter_size, strides=1, activation=None,
                kernel_initializer=kernel_initializer, padding='same') (x)
        else:
            x = Conv2D(dim, filter_size, strides=first_conv_strides, activation=None,
                kernel_initializer=kernel_initializer, padding='same') (x)
    else:
        x = SeparableConv2D(dim, filter_size, strides=first_conv_strides, 
                            activation=None, kernel_initializer=kernel_initializer,
                            padding='same') (x)
    x = SpatialDropout2D( dropout_value ) (x) if dropout_value else x
    x = BatchNormalization()(x) if bn else x
    x = Activation( activation )(x)
      
    if separable_conv == False:
        x = Conv2D(dim, filter_size, activation=None,
                kernel_initializer=kernel_initializer, padding='same') (x)
    else:
        x = SeparableConv2D(dim, filter_size, activation=None,
                kernel_initializer=kernel_initializer, padding='same') (x)

    # Add shortcut value to main path
    x = Add()([shortcut, x])
    return x

def level_block(x, depth, dim, fs, ac, k, d, bn, fcs, sc, fb, mp):

    if depth > 0:
        r = residual_block(x, dim, fs, ac, k, d, bn, fcs, sc, fb)
        x = MaxPooling2D((2, 2)) (r) if mp else r
        x = level_block(x, depth-1, (dim*2), fs, ac, k, d, bn, fcs, sc, False, mp) 
        x = Conv2DTranspose(dim, (2, 2), strides=(2, 2), padding='same') (x)
        x = Concatenate()([r, x])
        x = residual_block(x, dim, fs, ac, k, d, bn, 1, sc, False)
    else:
        x = residual_block(x, dim, fs, ac, k, d, bn, fcs, sc, False)
    return x


def preResUNet(image_shape, output_channels, activation='elu', kernel_initializer='he_normal',
            dropout_value=0.2, batchnorm=False, maxpooling=True, separable=False,
            numInitChannels=16, depth=4, upsampling_factor=2,
            upsample_method='UpSampling2D', final_activation=None):

    """Create the pre-upsampling ResU-Net for super-resolution
       Args:
            image_shape (array of 3 int): dimensions of the input image.
            activation (str, optional): Keras available activation type.
            kernel_initializer (str, optional): Keras available kernel 
            initializer type.
            dropout_value (real value, optional): dropout value
            batchnorm (bool, optional): use batch normalization
            maxpooling (bool, optional): use max-pooling between U-Net levels 
            (otherwise use stride of 2x2).
            separable (bool, optional): use SeparableConv2D instead of Conv2D
            numInitChannels (int, optional): number of channels at the
            first level of U-Net
            depth (int, optional): number of U-Net levels
            upsampling_factor (int, optional): initial image upsampling factor
            upsample_method (str, optional): upsampling method to use
            ('UpSampling2D', 'Conv2DTranspose', or 'SubpixelConv2D')
            final_activation (str, optional): activation function for the last
            layer
       Returns:
            model (Keras model): model containing the ResUNet created.
    """

    inputs = Input((None, None, image_shape[2]))

    if upsampling_factor > 1:
        s = upsample( inputs, out_channels=numInitChannels, method=upsample_method,
                    upsampling_factor=upsampling_factor,
                    input_shape=(image_shape[0], image_shape[1], image_shape[2]))

        conv_strides = (1,1) if maxpooling else (2,2)

        x = level_block(s, depth, numInitChannels, 3, activation, kernel_initializer,
                        dropout_value, batchnorm, conv_strides, separable, True,
                        maxpooling)

        #outputs = Conv2D(1, (1, 1), activation='sigmoid') (x)

        x = Add()([s,x]) # long shortcut
    else:
        conv_strides = (1,1) if maxpooling else (2,2)

        x = level_block(inputs, depth, numInitChannels, 3, activation, kernel_initializer,
                        dropout_value, batchnorm, conv_strides, separable, True,
                        maxpooling)
    outputs = Conv2D(output_channels, (1, 1), activation=final_activation) (x)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model

def postResUNet(image_shape, output_channels, activation='elu', kernel_initializer='he_normal',
            dropout_value=0.2, batchnorm=False, maxpooling=True, separable=False,
            numInitChannels=16, depth=4, upsampling_factor=2,
            upsample_method='UpSampling2D', final_activation=None ):

    """Create the post-upsampling ResU-Net for super-resolution
       Args:
            image_shape (array of 3 int): dimensions of the input image.
            activation (str, optional): Keras available activation type.
            kernel_initializer (str, optional): Keras available kernel 
            initializer type.
            dropout_value (real value, optional): dropout value
            batchnorm (bool, optional): use batch normalization
            maxpooling (bool, optional): use max-pooling between U-Net levels 
            (otherwise use stride of 2x2).
            separable (bool, optional): use SeparableConv2D instead of Conv2D
            numInitChannels (int, optional): number of channels at the
            first level of U-Net
            depth (int, optional): number of U-Net levels
            upsampling_factor (int, optional): initial image upsampling factor
            upsample_method (str, optional): upsampling method to use
            ('UpSampling2D', 'Conv2DTranspose', or 'SubpixelConv2D')
            final_activation (str, optional): activation function for the last
            layer
       Returns:
            model (Keras model): model containing the ResUNet created.
    """

    inputs = Input((None, None, image_shape[2]))

    conv_strides = (1,1) if maxpooling else (2,2)

    x = level_block(inputs, depth, numInitChannels, 3, activation, kernel_initializer,
                    dropout_value, batchnorm, conv_strides, separable, True,
                    maxpooling)
    if upsampling_factor > 1:
        x = upsample( x, out_channels=numInitChannels, method=upsample_method,
                    upsampling_factor=upsampling_factor,
                    input_shape=(image_shape[0], image_shape[1], image_shape[2]))

    #outputs = Conv2D(1, (1, 1), activation='sigmoid') (x)
    outputs = Conv2D(output_channels, (1, 1), activation=final_activation) (x)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model