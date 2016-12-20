import keras.backend as K
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, merge, Dropout, ZeroPadding2D, LocallyConnected2D, BatchNormalization, Activation
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

    z = Input(shape=(latent_size, ))

    x = Dense(1024)(z)
    x = LeakyReLU()(x)

    x = Dense(128 * 7 * 7)(x)
    x = LeakyReLU()(x)

    x = Reshape((7, 7, 128))(x)

    # upsample to (..., 14, 14)
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(128, 5, 5, border_mode='same', init='he_uniform')(x)
    skip = LeakyReLU()(x)

    # start the second res block
    x = Convolution2D(128, 5, 5, border_mode='same', init='he_uniform')(skip)

    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Convolution2D(128, 4, 4, border_mode='same', init='he_uniform')(x)

    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(x)

    x = merge([skip, x], mode='sum')
    x = LeakyReLU()(x)

    # upsample to (..., 28, 28)
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(64, 3, 3, border_mode='valid', init='he_uniform')(x)
    skip = LeakyReLU()(x)

    # start the second res block
    x = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(skip)

    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(x)

    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(x)

    x = merge([skip, x], mode='sum')
    x = LeakyReLU()(x)

    cnn_out = LocallyConnected2D(1, 2, 2, border_mode='valid', bias=False,
                                 init='glorot_normal', activation='relu')(x)

    cnn = Model(input=z, output=cnn_out)

    z = Input(shape=(latent_size, ))

    x = Dense(25 ** 2, input_dim=latent_size, init='he_uniform')(z)
    skip1 = LeakyReLU()(x)

    x = Dense(25 ** 2, init='he_uniform')(skip1)
    x = merge([skip1, x], mode='sum')
    skip2 = LeakyReLU()(x)

    x = Dense(25 ** 2, init='he_uniform')(skip2)
    x = merge([skip2, x], mode='sum')
    skip2 = LeakyReLU()(x)
    # h3 = merge([h3, h2, h1], mode='sum')

    x = Dense(25 ** 2, init='he_uniform')(skip2)
    x = merge([skip2, x], mode='sum')
    skip2 = LeakyReLU()(x)
    # h4 = merge([h4, h3, h2, h1], mode='sum')

    x = Dense(25 ** 2, init='he_uniform')(skip2)
    x = merge([skip1, skip2, x], mode='sum')

    loc_out = Reshape((25, 25, 1))(Activation('relu')(x))

    loc = Model(input=z, output=loc_out)

    # loc = Sequential()

    # loc.add(Dense(512, input_dim=latent_size, init='he_uniform'))
    # loc.add(LeakyReLU())

    # loc.add(Dense(1024, init='he_uniform'))
    # loc.add(LeakyReLU())
    # # loc.add(BatchNormalization(mode=2, axis=1))

    # # loc.add(Dense(1024, init='he_uniform'))
    # # loc.add(LeakyReLU())
    # # loc.add(Dropout(0.3))

    # loc.add(Dense(25 ** 2, init='he_uniform'))
    # loc.add(LeakyReLU())
    # loc.add(Reshape((25, 25, 1)))

    # loc.add(Convolution2D(1, 2, 2, border_mode='same', init='he_uniform',
    #                       activation='relu'))
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
