import sys

sys.setrecursionlimit(10000)

import warnings

warnings.filterwarnings("ignore")

import tensorflow as tf

from tensorflow.keras.layers import Add, Input, Lambda
from tensorflow.keras.models import Model

import tensorflow_addons as tfa


import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

######


def subpixel_conv2d(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)


def wdsr_a(
    scale,
    num_filters=32,
    num_res_blocks=8,
    res_block_expansion=4,
    res_block_scaling=None,
    out_channels=1,
):
    return wdsr(
        scale,
        num_filters,
        num_res_blocks,
        res_block_expansion,
        res_block_scaling,
        res_block_a,
        out_channels,
    )


def wdsr_b(
    scale,
    num_filters=32,
    num_res_blocks=8,
    res_block_expansion=6,
    res_block_scaling=None,
    out_channels=1,
):
    return wdsr(
        scale,
        num_filters,
        num_res_blocks,
        res_block_expansion,
        res_block_scaling,
        res_block_b,
        out_channels,
    )


def wdsr(
    scale,
    num_filters,
    num_res_blocks,
    res_block_expansion,
    res_block_scaling,
    res_block,
    out_channels=1,
):
    x_in = Input(shape=(None, None, out_channels))
    x = x_in
    # main branch
    m = conv2d_weightnorm(num_filters, 3, padding="same")(x)
    for i in range(num_res_blocks):
        m = res_block(
            m,
            num_filters,
            res_block_expansion,
            kernel_size=3,
            scaling=res_block_scaling,
        )
    m = conv2d_weightnorm(
        out_channels * scale**2, 3, padding="same", name=f"conv2d_main_scale_{scale}"
    )(m)
    m = Lambda(subpixel_conv2d(scale))(m)

    # skip branch
    s = conv2d_weightnorm(
        out_channels * scale**2, 5, padding="same", name=f"conv2d_skip_scale_{scale}"
    )(x)
    s = Lambda(subpixel_conv2d(scale))(s)

    x = Add()([m, s])
    # final convolution with sigmoid activation ?
    # x = Conv2D(out_channels, 3, padding='same', activation='sigmoid')(x)

    return Model(x_in, x, name="wdsr")


def res_block_a(x_in, num_filters, expansion, kernel_size, scaling):
    x = conv2d_weightnorm(
        num_filters * expansion, kernel_size, padding="same", activation="relu"
    )(x_in)
    x = conv2d_weightnorm(num_filters, kernel_size, padding="same")(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def res_block_b(x_in, num_filters, expansion, kernel_size, scaling):
    linear = 0.8
    x = conv2d_weightnorm(
        num_filters * expansion, 1, padding="same", activation="relu"
    )(x_in)
    x = conv2d_weightnorm(int(num_filters * linear), 1, padding="same")(x)
    x = conv2d_weightnorm(num_filters, kernel_size, padding="same")(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def conv2d_weightnorm(filters, kernel_size, padding="same", activation=None, **kwargs):
    return tfa.layers.WeightNormalization(
        tf.keras.layers.Conv2D(
            filters, kernel_size, padding=padding, activation=activation, **kwargs
        ),
        data_init=False,
    )


######
