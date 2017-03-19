#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
file: fcn.py
description: discrimination submodel for [arXiv/1701.05927]
author: Luke de Oliveira (lukedeoliveira@lbl.gov)
"""

import keras.backend as K
from keras.layers import (Input, Dense, Reshape, Flatten, Lambda, merge,
                          Dropout, BatchNormalization, Activation, Embedding)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import (UpSampling2D, Conv2D, ZeroPadding2D,
                                        AveragePooling2D)
from keras.layers.local import LocallyConnected2D

from keras.models import Model, Sequential

from .ops import minibatch_discriminator, minibatch_output_shape, Dense3D


K.set_image_dim_ordering('tf')


def discriminator():

    image = Input(shape=(25, 25, 1))

    x = Flatten()(image)
    x = Dense(1024)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)

    x = Dense(512)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)

    x = Dense(128)(x)
    x = LeakyReLU()(x)
    h = Dropout(0.2)(x)

    dnn = Model(image, h)

    image = Input(shape=(25, 25, 1))

    dnn_out = dnn(image)

    # nb of features to obtain
    nb_features = 20

    # dim of kernel space
    vspace_dim = 10

    # creates the kernel space for the minibatch discrimination
    K_x = Dense3D(nb_features, vspace_dim)(dnn_out)

    minibatch_featurizer = Lambda(minibatch_discriminator,
                                  output_shape=minibatch_output_shape)

    # concat the minibatch features with the normal ones
    features = merge([
        minibatch_featurizer(K_x),
        dnn_out
    ], mode='concat')

    # fake output tracks binary fake / not-fake, and the auxiliary requires
    # reconstruction of latent features, in this case, labels
    fake = Dense(1, activation='sigmoid', name='generation')(features)
    aux = Dense(1, activation='sigmoid', name='auxiliary')(features)

    return Model(input=image, output=[fake, aux])


def generator(latent_size, return_intermediate=False):

    loc = Sequential([
        # DCGAN-style project & reshape,
        Dense(256, input_dim=latent_size),
        LeakyReLU(),
        Dense(512),
        LeakyReLU(),
        Dense(625),
        LeakyReLU(),
        Dense(625),
        Activation('relu'),
        Reshape((25, 25, 1))
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
