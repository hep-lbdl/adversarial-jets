import keras.backend as K
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, merge, Dropout, ZeroPadding2D, LocallyConnected2D, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Convolution2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
import numpy as np

K.set_image_dim_ordering('tf')


def basic_generator(latent_size):
    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 1, 28, 28)
    cnn = Sequential()

    cnn.add(Dense(1024, input_dim=latent_size, activation='relu'))
    cnn.add(Dense(128 * 7 * 7, activation='relu'))
    cnn.add(Reshape((128, 7, 7)))

    # upsample to (..., 14, 14)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(256, 5, 5, border_mode='same',
                          activation='relu', init='glorot_normal'))

    # upsample to (..., 28, 28)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(128, 5, 5, border_mode='same',
                          activation='relu', init='glorot_normal'))

    # take a channel axis reduction
    cnn.add(Convolution2D(1, 2, 2, border_mode='same',
                          activation='tanh', init='glorot_normal'))

    # this is the z space commonly refered to in GAN papers
    latent = Input(shape=(latent_size, ))

    # this will be our label
    image_class = Input(shape=(1,), dtype='int32')

    # 10 classes in MNIST
    cls = Flatten()(Embedding(10, latent_size,
                              init='glorot_normal')(image_class))

    # hadamard product between z-space and a class conditional embedding
    h = merge([latent, cls], mode='mul')

    fake_image = cnn(h)

    return Model(input=[latent, image_class], output=fake_image)


def locally_connected_generator(latent_size, return_intermediate=False):

    cnn = Sequential()

    cnn.add(Dense(1024, input_dim=latent_size))
    cnn.add(LeakyReLU())
    # cnn.add(Dropout(0.3))
    cnn.add(Dense(128 * 7 * 7))
    cnn.add(LeakyReLU())
    # cnn.add(Dropout(0.3))
    cnn.add(Reshape((7, 7, 128)))

    # upsample to (..., 14, 14)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(256, 5, 5, border_mode='same',
                          activation='tanh', init='glorot_normal'))
    cnn.add(LeakyReLU())

    # upsample to (..., 28, 28)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(128, 4, 4, border_mode='valid',
                          activation='tanh', init='glorot_normal'))

    cnn.add(Convolution2D(1, 5, 5, border_mode='same', bias=False,
                          init='glorot_normal', activation='relu'))

    loc = Sequential()

    loc.add(Dense(512, input_dim=latent_size, init='glorot_normal'))
    loc.add(LeakyReLU())

    loc.add(Dense(1024, init='glorot_normal'))
    loc.add(LeakyReLU())
    # loc.add(BatchNormalization(mode=2, axis=1))

    # loc.add(Dense(1024, init='glorot_normal'))
    # loc.add(LeakyReLU())
    # loc.add(Dropout(0.3))

    loc.add(Dense(25 ** 2, activation='relu', init='glorot_normal'))
    loc.add(Reshape((25, 25, 1)))

    # this is the z space commonly refered to in GAN papers
    latent = Input(shape=(latent_size, ))

    # this will be our label
    image_class = Input(shape=(1, ), dtype='int32')
    cls = Flatten()(Embedding(2, latent_size, input_length=1,
                              init='glorot_normal')(image_class))

    # hadamard product between z-space and a class conditional embedding
    # h = merge([latent, cls], mode='concat', concat_axis=-1)
    h = merge([latent, cls], mode='mul')

    cnn_img, loc_img = cnn(h), loc(h)

    # # initialize this to flat prior between streams
    # pointwise_reduce = LocallyConnected2D(1, 1, 1, bias=False,
    #                                       weights=[np.ones((625, 2, 1)) / 2])

    # # concat over the channel axis
    # fake_image = pointwise_reduce(
    #     merge([cnn_img, loc_img], mode='concat', concat_axis=-1))

    fake_image = merge([cnn_img, loc_img], mode='ave')

    # fake_image = merge([cnn_img, loc_img], mode='ave')

    if not return_intermediate:
        return Model(input=[latent, image_class], output=fake_image)
    return Model(input=[latent, image_class], output=fake_image), (latent, image_class), (cnn_img, loc_img)
