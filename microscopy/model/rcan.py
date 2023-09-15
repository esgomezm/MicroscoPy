# Model based on: https://github.com/yulunzhang/RCAN

import sys
sys.setrecursionlimit(10000)

import warnings

warnings.filterwarnings("ignore")
import tensorflow as tf

from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Activation,
    Add,
    Lambda,
    GlobalAveragePooling2D,
    Multiply,
    Dense,
    Reshape,
)
from tensorflow.keras.models import Model

import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

######

# Multi Scale Similarity Index loss

# Default values obtained by Wang et al.
_MSSSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def ms_ssim_loss(
    max_val=1,
    power_factors=_MSSSIM_WEIGHTS,
    filter_size=11,
    filter_sigma=1.5,
    k1=0.01,
    k2=0.03,
):
    def ms_ssim_loss_fixed(y_true, y_pred):
        return 1 - tf.image.ssim_multiscale(
            y_true, y_pred, max_val, power_factors, filter_size, filter_sigma, k1, k2
        )

    return ms_ssim_loss_fixed


class Mish(tf.keras.layers.Layer):
    """
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X_input = Input(input_shape)
        >>> X = Mish()(X_input)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return inputs * tf.math.tanh(tf.math.softplus(inputs))

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}

    def compute_output_shape(self, input_shape):
        return input_shape


def sub_pixel_conv2d(scale=2, **kwargs):
    return Lambda(lambda x: tf.nn.depth_to_space(x, scale), **kwargs)


def upsample(input_tensor, filters, use_mish=False):
    x = Conv2D(filters=filters * 4, kernel_size=3, strides=1, padding="same")(
        input_tensor
    )
    x = sub_pixel_conv2d(scale=2)(x)
    if use_mish:
        x = Mish()(x)
    else:
        x = Activation("relu")(x)
    return x


def ca(input_tensor, filters, reduce=16, use_mish=False):
    x = GlobalAveragePooling2D()(input_tensor)
    x = Reshape((1, 1, filters))(x)
    if use_mish:
        x = Dense(filters / reduce, kernel_initializer="he_normal", use_bias=False)(x)
        x = Mish()(x)
    else:
        x = Dense(
            filters / reduce,
            activation="relu",
            kernel_initializer="he_normal",
            use_bias=False,
        )(x)
    x = Dense(
        filters, activation="sigmoid", kernel_initializer="he_normal", use_bias=False
    )(x)
    x = Multiply()([x, input_tensor])
    return x


def rcab(input_tensor, filters, scale=0.1, use_mish=False):
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same")(input_tensor)
    if use_mish:
        x = Mish()(x)
    else:
        x = Activation("relu")(x)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same")(x)
    x = ca(x, filters, use_mish=use_mish)
    if scale:
        x = Lambda(lambda t: t * scale)(x)
    x = Add()([x, input_tensor])

    return x


def rg(input_tensor, filters, n_rcab=20, use_mish=False):
    x = input_tensor
    for _ in range(n_rcab):
        x = rcab(x, filters, use_mish=use_mish)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same")(x)
    x = Add()([x, input_tensor])

    return x


def rir(input_tensor, filters, n_rg=10, use_mish=False):
    x = input_tensor
    for _ in range(n_rg):
        x = rg(x, filters=filters, use_mish=use_mish)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same")(x)
    x = Add()([x, input_tensor])

    return x


def rcan(filters=64, n_sub_block=2, out_channels=1, use_mish=False):
    inputs = Input(shape=(None, None, out_channels))

    x = x_1 = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same")(inputs)
    x = rir(x, filters=filters, use_mish=use_mish)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same")(x)
    x = Add()([x_1, x])

    for _ in range(n_sub_block):
        x = upsample(x, filters, use_mish=use_mish)
    x = Conv2D(filters=out_channels, kernel_size=3, strides=1, padding="same")(x)

    return Model(inputs=inputs, outputs=x)
