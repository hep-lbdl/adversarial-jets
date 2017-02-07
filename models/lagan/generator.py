#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
file: generators.py
description: generator submodel for [arXiv/1701.05927]
author: Luke de Oliveira (lukedeoliveira@lbl.gov)
"""
import keras.backend as K
from keras.layers import (Input, Dense, Reshape, Flatten, Embedding, merge,
                          Dropout, BatchNormalization, Activation)

from keras.layers.advanced_activations import LeakyReLU

from keras.layers.local import LocallyConnected2D
from keras.layers.convolutional import UpSampling2D, Conv2D, ZeroPadding2D

from keras.models import Model, Sequential

K.set_image_dim_ordering('tf')


def generator(latent_size, return_intermediate=False):

    loc = Sequential([
        # DCGAN-style project & reshape,
        Dense(128 * 7 * 7, input_dim=latent_size),
        Reshape((7, 7, 128)),

        # block 1: (None, 7, 7, 128) => (None, 14, 14, 64),
        Conv2D(64, 5, 5, border_mode='same', init='he_uniform'),
        LeakyReLU(),
        BatchNormalization(),
        UpSampling2D(size=(2, 2)),

        # block 2: (None, 14, 14, 64) => (None, 28, 28, 6),
        ZeroPadding2D((2, 2)),
        LocallyConnected2D(6, 5, 5, init='he_uniform'),
        LeakyReLU(),
        BatchNormalization(),
        UpSampling2D(size=(2, 2)),

        # block 3: (None, 28, 28, 6) => (None, 25, 25, 1),
        LocallyConnected2D(6, 3, 3, init='he_uniform'),
        LeakyReLU(),
        LocallyConnected2D(1, 2, 2, bias=False, init='glorot_normal'),
        Activation('relu')
    ])

    # this is the z space commonly refered to in GAN papers
    latent = Input(shape=(latent_size, ))

    # this will be our label
    image_class = Input(shape=(1, ), dtype='int32')
    emb = Flatten()(Embedding(2, latent_size, input_length=1,
                              init='glorot_normal')(image_class))

    # hadamard product between z-space and a class conditional embedding
    h = merge([latent, emb], mode='mul')

    fake_image = loc(h)

    return Model(input=[latent, image_class], output=fake_image)
